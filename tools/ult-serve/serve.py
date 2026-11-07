#!/usr/bin/env python3
"""ult-serve — OpenAI-compatible endpoint for the ULT (GA106) engine.

Bridges xera-rsi's OpenAI chat-completions client to the megakernel:
  GET  /v1/models             -> loaded model (+ any other profiles)
  GET  /health                -> {"status":"ok"}
  POST /v1/chat/completions   -> per-model template render -> Engine
                                 (SSE streaming from the mapped token
                                 counter mid-kernel — same mechanics as
                                 GA106's chat.py, including the
                                 drain/reset/prefill/reset order)

What this adds over the engine's own serve.py, for agent use:
  - reads max_completion_tokens (xera-rsi sends that field first)
  - injects tool schemas and parses <tool_call>{...}</tool_call> output
    back into OpenAI tool_calls (small models: best-effort)
  - splits R1/LFM3 <think>...</think> into reasoning_content, so the
    harness shows reasoning separately
  - --tools-limit caps the injected schema list (prompt size ↔ speed)

Run:
  python serve.py --model qwen --port 8000
  python serve.py --model lfm3 --port 8000

Env: GA106_HOME (engine source tree, default E:/GA106),
     ULT_QUANT (default q4), GA106_LFM3_VARIANT (default ult_exact).
"""
import json
import os
import sys
import time
import queue
import threading
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GA106 = os.environ.get("GA106_HOME", r"E:\GA106")
sys.path.insert(0, GA106)

from tokenizers import Tokenizer           # noqa: E402
from ga106.engine import Engine            # noqa: E402
from ga106.engine_lfm3 import EngineLFM3   # noqa: E402
from ga106 import cuda as _c               # noqa: E402

MAXPOS = 8192
_lock = threading.Lock()          # one persistent kernel at a time

PROFILES = {
    "qwen": {
        "name": "ult-qwen-1.5b",
        "tok": r"E:\DeepSeek-R1-Distill-Qwen-1.5B\tokenizer.json",
        "eos": 151643,
        "template": "r1",
        "engine": lambda: Engine(
            os.path.join(GA106, r"models\qwen1.5b\model.safetensors"),
            quant=os.environ.get("GA106_QUANT", "q4")),
    },
    "lfm3": {
        "name": "ult-lfm3-1.7b",
        "tok": os.path.join(GA106, r"models\lfm3-1.7b\tokenizer.json"),
        "eos": 124900,
        "template": "chatml",
        "engine": None,          # filled in main() (needs variant env)
    },
}


def lfm3_engine():
    variant = os.environ.get("GA106_LFM3_VARIANT", "ult_exact")
    model = os.path.join(GA106, r"models\lfm3-1.7b\model.safetensors")
    ult = os.path.join(GA106, r"kernels\lfm3.ult")
    if variant == "q4":
        return EngineLFM3(model, quant="q4")
    if variant == "ult_exact":
        return EngineLFM3.from_ult(model, ult,
                                   options={"fuse_conv_o": 1,
                                            "threads": 384})
    if variant == "ult_hfma":
        return EngineLFM3.from_ult(model, ult,
                                   options={"hfma": 1, "fuse_conv_o": 1,
                                            "threads": 384})
    raise ValueError("GA106_LFM3_VARIANT must be q4|ult_exact|ult_hfma")


PROFILES["lfm3"]["engine"] = lfm3_engine

# ---------- templates (mirrors GA106 chat.py) ----------

TOOL_PREAMBLE = (
    "You can call tools by emitting a line in your reply:\n"
    '<tool_call>{"name":"<tool>","arguments":{...json...}}</tool_call>\n'
    "Emit one tool_call per line, nothing else on those lines.\n"
    "Available tools:\n")


def tool_block(tools):
    schemas = json.dumps(
        [{"name": t.get("function", {}).get("name"),
          "description": t.get("function", {}).get("description", ""),
          "parameters": t.get("function", {}).get("parameters", {})}
         for t in tools], ensure_ascii=False)
    return TOOL_PREAMBLE + schemas


def render_prompt(profile, messages, tools):
    sys_extra = tool_block(tools) if tools else ""
    if profile["template"] == "chatml":
        out = "<|startoftext|>"
        first = True
        for m in messages:
            role, content = m.get("role", "user"), m.get("content", "")
            if role == "tool":
                role, content = "user", f"<result>{content}</result>"
            if role == "assistant" and m.get("tool_calls"):
                calls = "".join(
                    '<tool_call>{"name":%s,"arguments":%s}</tool_call>\n'
                    % (json.dumps(tc["function"]["name"]),
                       tc["function"].get("arguments", "{}"))
                    for tc in m["tool_calls"])
                content = (content + "\n" + calls).strip()
            if role == "assistant":
                body = content.split("</think>")[-1].strip()
                out += f"<|im_start|>assistant\n<think></think>{body}" \
                       "<|im_end|>\n"
            else:
                if role == "system" and first and sys_extra:
                    content += "\n\n" + sys_extra
                    sys_extra = ""
                out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            first = False
        if sys_extra:
            out += f"<|im_start|>system\n{sys_extra}<|im_end|>\n"
        return out + "<|im_start|>assistant\n<think>"

    # r1 template (DeepSeek distill): raw system, User/Assistant turns,
    # think primed
    out = "<|begin▁of▁sentence|>"
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        if role == "system":
            if sys_extra:
                content += "\n\n" + sys_extra
                sys_extra = ""
            out += content
        elif role == "assistant":
            body = content.split("</think>")[-1].strip()
            out += f"<|Assistant|>{body}<|end▁of▁sentence|>"
        elif role == "tool":
            out += f"<|User|><result>{content}</result>"
        else:
            out += f"<|User|>{content}"
    if sys_extra:
        out += sys_extra
    return out + "<|Assistant|><think>\n"


# ---------- tool_call / think parsing ----------

import re                                   # noqa: E402
TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                         re.DOTALL)


def split_tool_calls(text):
    calls = []
    for m in TOOLCALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("tool")
        if not name:
            continue
        args = obj.get("arguments") or obj.get("args") or {}
        calls.append({
            "id": f"call_{len(calls)}", "type": "function",
            "function": {"name": name,
                         "arguments": args if isinstance(args, str)
                         else json.dumps(args)}})
    return TOOLCALL_RE.sub("", text).strip(), calls


def split_think(text):
    """-> (reasoning, content). Prompt already opened <think>; generated
    text runs to </think> then the answer."""
    if "</think>" in text:
        think, _, rest = text.partition("</think>")
        return think.strip(), rest.strip()
    return "", text.strip()


# ---------- engine mechanics (from GA106 chat.py — do not reorder) ----------

def generate(profile, tok, eng, ids_in, ntok, on_tok=None):
    """Prefill sequentially, then one decode launch; polls the mapped
    token counter. on_tok(list-of-new-ids) fires as tokens land.
    Returns (generated_ids, elapsed_s)."""
    eos_id = profile["eos"]
    with _lock:
        # drain any orphan kernel from a disconnected request — it keeps
        # writing tokens[] while running
        t_drain = time.perf_counter()
        while not _c.stream_query(eng.stream):
            if time.perf_counter() - t_drain > 30:
                raise TimeoutError("previous generation still active")
            time.sleep(0.001)
        eng._tok_host[0] = 0
        eng._tok_host[8192 + 4] = -1
        _c.memset_dev(eng.d_bar, 0, 64)
        _c.memset_dev(eng.d_plog, 0, (4 * 56 + 64 + 4096) * 4)
        _c.sync()

        for i, tk in enumerate(ids_in[:-1]):
            eng._launch(tk, i, 1)
            eng._wait()

        # prefill launches bump tokens[0]/[1] — reset so the poll only
        # counts DECODE tokens (the stale-slot bug chat.py documents)
        eng._tok_host[0] = 0
        _c.sync()

        t0 = time.perf_counter()
        eng._launch(ids_in[-1], len(ids_in) - 1, ntok)
        got = 0
        eos = False
        while got < ntok:
            c = eng._tok_host[0]
            if got < c:
                new_ids = []
                while got < c and got < ntok:
                    got += 1
                    tid = eng._tok_host[got]
                    if tid == eos_id:
                        eos = True
                        continue
                    if not eos:
                        new_ids.append(tid)
                if new_ids and on_tok:
                    on_tok(new_ids)
            elif got < ntok:
                time.sleep(0.0005)
        _c.stream_sync(eng.stream)
    return got, time.perf_counter() - t0


# ---------- http ----------

def load_engine(slot):
    """Init engine + tokenizer + warmup. Kernel stalls under GPU
    contention are transient — retry a few times before giving up."""
    for attempt in range(3):
        try:
            slot["eng"] = slot["profile"]["engine"]()
            slot["tok"] = Tokenizer.from_file(slot["profile"]["tok"])
            bos = "<|begin▁of▁sentence|>" if slot["name"] == "qwen" \
                else "<|startoftext|>"
            slot["eng"].generate(slot["tok"].encode(bos + "hi").ids, 4)
            slot["failed"] = None
            return True
        except Exception as e:
            slot["eng"] = None
            slot["failed"] = e
            sys.stderr.write(
                f"ult-serve: {slot['name']} load attempt "
                f"{attempt + 1} failed: {e}\n")
            time.sleep(3)
    return False


def make_handler(cfg):
    def resolve(req_model):
        """request model id -> loaded (or lazily loadable) engine slot."""
        for name, slot in cfg["engines"].items():
            if req_model in (name, slot["profile"]["name"]):
                return slot
        return cfg["engines"][cfg["default"]]

    def ensure_loaded(slot):
        if slot["eng"] is not None or slot["failed"] is not None:
            return
        with _lock:
            if slot["eng"] is None and slot["failed"] is None:
                t = time.perf_counter()
                sys.stderr.write(
                    f"ult-serve: lazy-loading {slot['profile']['name']} "
                    "(first request may take a minute)...\n")
                load_engine(slot)
                sys.stderr.write(
                    f"ult-serve: {slot['name']} ready "
                    f"({time.perf_counter() - t:.1f}s)\n")

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _sse(self, chunk):
            data = chunk if isinstance(chunk, str) else json.dumps(chunk)
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()

        def _chunk(self, model_id, delta, finish=None, usage=None):
            c = {"id": "ult", "object": "chat.completion.chunk",
                 "created": int(time.time()), "model": model_id,
                 "choices": [{"index": 0, "delta": delta,
                              "finish_reason": finish}]}
            if usage:
                c["usage"] = usage
            return c

        def do_GET(self):
            _c.ctx_set_current()
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if self.path in ("/v1/models", "/models"):
                # advertise every profile — lazy-loads on first request,
                # so /model in the TUI can switch freely
                data = [{"id": s["profile"]["name"], "object": "model",
                         "created": int(time.time()), "owned_by": "ult"}
                        for s in cfg["engines"].values()]
                return self._json(200, {"object": "list", "data": data})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            _c.ctx_set_current()
            if self.path not in ("/v1/chat/completions",
                                 "/chat/completions"):
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
            except Exception:
                return self._json(400, {"error": "bad json"})

            slot = resolve(req.get("model") or "")
            if slot["failed"] is not None:
                return self._json(500, {"error":
                    f"engine failed to load: {slot['failed']}"})
            ensure_loaded(slot)
            if slot["eng"] is None:
                return self._json(500, {"error":
                    f"engine failed to load: {slot['failed']}"})
            profile, tok, eng = slot["profile"], slot["tok"], slot["eng"]
            model_id = profile["name"]

            messages = req.get("messages") or []
            tools = req.get("tools") or req.get("functions")
            if tools and cfg["tools_limit"] >= 0:
                tools = tools[: cfg["tools_limit"]] or None
            ntok = int(req.get("max_completion_tokens")
                       or req.get("max_tokens") or 512)
            stream = bool(req.get("stream"))

            prompt = render_prompt(profile, messages, tools)
            ids_in = tok.encode(prompt).ids
            sys.stderr.write(
                f"ult-serve: {model_id} prompt={len(ids_in)}tok "
                f"gen≤{ntok} stream={stream}\n")
            # truncate the middle of the prompt (keep head + tail)
            ntok = max(1, min(ntok, MAXPOS - len(ids_in) - 8))
            if len(ids_in) + ntok > MAXPOS:
                budget = MAXPOS - ntok - 8
                if budget < 128:
                    return self._json(413, {
                        "error": "prompt too long for MAXPOS"})
                keep = budget // 3
                ids_in = ids_in[:keep] + ids_in[len(ids_in) - (budget - keep):]
                sys.stderr.write(
                    "ult-serve: prompt truncated to fit MAXPOS\n")

            if stream and not tools:
                return self._stream(slot, ids_in, ntok)
            try:
                gen_ids = []
                generate(profile, tok, eng,
                         ids_in, ntok, on_tok=gen_ids.extend)
            except Exception as e:
                return self._json(500, {"error": f"engine: {e}"})
            text = tok.decode(gen_ids, skip_special_tokens=True)
            reasoning, clean = split_think(text)
            clean, calls = split_tool_calls(clean)
            msg = {"role": "assistant", "content": clean}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = calls
            resp = {"id": "ult", "object": "chat.completion",
                    "created": int(time.time()), "model": model_id,
                    "choices": [{"index": 0, "message": msg,
                                 "finish_reason":
                                 "tool_calls" if calls else "stop"}],
                    "usage": {"prompt_tokens": len(ids_in),
                              "completion_tokens": len(gen_ids),
                              "total_tokens": len(ids_in) + len(gen_ids)}}
            if stream:
                return self._stream_buffered(resp)
            return self._json(200, resp)

        def _stream_head(self, model_id):
            self.close_connection = True      # body ends at close
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self._sse(self._chunk(model_id,
                {"role": "assistant", "content": ""}))

        def _stream(self, slot, ids_in, ntok):
            """Live SSE: decode-suffix deltas, think->reasoning split."""
            model_id = slot["profile"]["name"]
            self._stream_head(model_id)
            tok = slot["tok"]
            state = {"ids": [], "think_done": False,
                     "sent_reason": 0, "sent_content": 0}

            def emit(new_ids):
                s = state
                s["ids"].extend(new_ids)
                text = tok.decode(s["ids"], skip_special_tokens=True)
                if not s["think_done"]:
                    if "</think>" not in text:
                        delta = text[s["sent_reason"]:]
                        if delta:
                            self._sse(self._chunk(model_id,
                                {"reasoning_content": delta}))
                            s["sent_reason"] = len(text)
                        return
                    s["think_done"] = True
                    think, _, rest = text.partition("</think>")
                    delta = think[s["sent_reason"]:]
                    if delta:
                        self._sse(self._chunk(model_id,
                            {"reasoning_content": delta}))
                    s["sent_reason"] = len(think)
                    s["sent_content"] = 0
                    text = rest
                delta = text[s["sent_content"]:]
                if delta:
                    self._sse(self._chunk(model_id, {"content": delta}))
                    s["sent_content"] = len(text)

            try:
                got, dt = generate(slot["profile"], tok, slot["eng"],
                                   ids_in, ntok, on_tok=emit)
                n_gen = len(state["ids"])
            except Exception as e:
                self._sse(self._chunk(model_id,
                                      {"content": f"[ult error: {e}]"}))
                n_gen = 0
            usage = {"prompt_tokens": len(ids_in),
                     "completion_tokens": n_gen,
                     "total_tokens": len(ids_in) + n_gen}
            self._sse(self._chunk(model_id, {}, finish="stop",
                                  usage=usage))
            self._sse("[DONE]")

        def _stream_buffered(self, resp):
            """tools + stream: parse first, then emit buffered chunks."""
            model_id = resp["model"]
            self._stream_head(model_id)
            msg = resp["choices"][0]["message"]
            if msg.get("reasoning_content"):
                self._sse(self._chunk(model_id,
                    {"reasoning_content": msg["reasoning_content"]}))
            if msg.get("content"):
                self._sse(self._chunk(model_id,
                                      {"content": msg["content"]}))
            for i, tc in enumerate(msg.get("tool_calls") or []):
                tc["index"] = i
                self._sse(self._chunk(model_id, {"tool_calls": [tc]}))
            self._sse(self._chunk(model_id, {},
                                  finish=resp["choices"][0]
                                  ["finish_reason"],
                                  usage=resp["usage"]))
            self._sse("[DONE]")

    return H


def main():
    ap = argparse.ArgumentParser(description="ULT OpenAI-compatible server")
    ap.add_argument("--model", default="all",
                    help="engine profile(s): qwen, lfm3, all, or comma list")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tools-limit", type=int, default=8,
                    help="cap injected tool schemas (-1 = all)")
    args = ap.parse_args()

    wanted = [s.strip() for s in args.model.split(",") if s.strip()]
    if wanted == ["all"]:
        wanted = list(PROFILES)
    for name in wanted:
        if name not in PROFILES:
            ap.error(f"unknown model '{name}' (choices: {list(PROFILES)})")

    cfg = {"engines": {name: {"name": name, "profile": PROFILES[name],
                              "eng": None, "tok": None, "failed": None}
                       for name in wanted},
           "default": wanted[0],
           "tools_limit": args.tools_limit}

    # eagerly load every requested engine — a lazy load inside a request
    # looks like a hang (NVRTC compile + upload can take minutes). A
    # stalled kernel under GPU contention retries; a dead engine is
    # marked failed instead of killing the server.
    for name, slot in cfg["engines"].items():
        t = time.perf_counter()
        print(f"ult-serve: loading {slot['profile']['name']}...",
              flush=True)
        if load_engine(slot):
            print(f"ult-serve: {name} ready "
                  f"({time.perf_counter() - t:.1f}s)", flush=True)
        else:
            print(f"ult-serve: {name} FAILED — will 500 on use",
                  flush=True)

    names = [s["profile"]["name"] for s in cfg["engines"].values()]
    print(f"ult-serve [{cfg['default']}] "
          f"http://{args.host}:{args.port}/v1  models={names}", flush=True)
    class QuietServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            # client disconnects mid-stream are routine (interrupted
            # turns) — don't dump a traceback for them
            pass

    QuietServer((args.host, args.port),
                make_handler(cfg)).serve_forever()


if __name__ == "__main__":
    main()
