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

def make_handler(cfg):
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

        def _chunk(self, delta, finish=None, usage=None):
            c = {"id": "ult", "object": "chat.completion.chunk",
                 "created": int(time.time()), "model": cfg["model_id"],
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
                data = [{"id": cfg["model_id"], "object": "model",
                         "created": int(time.time()), "owned_by": "ult"}]
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

            messages = req.get("messages") or []
            tools = req.get("tools") or req.get("functions")
            if tools and cfg["tools_limit"] >= 0:
                tools = tools[: cfg["tools_limit"]] or None
            ntok = int(req.get("max_completion_tokens")
                       or req.get("max_tokens") or 512)
            stream = bool(req.get("stream"))

            prompt = render_prompt(cfg["profile"], messages, tools)
            ids_in = cfg["tok"].encode(prompt).ids
            # fixed MAXPOS window: shrink generation budget first, then
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
                return self._stream(ids_in, ntok)
            try:
                gen_ids = []
                generate(cfg["profile"], cfg["tok"], cfg["eng"],
                         ids_in, ntok, on_tok=gen_ids.extend)
            except Exception as e:
                return self._json(500, {"error": f"engine: {e}"})
            text = cfg["tok"].decode(gen_ids, skip_special_tokens=True)
            reasoning, clean = split_think(text)
            clean, calls = split_tool_calls(clean)
            msg = {"role": "assistant", "content": clean}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = calls
            resp = {"id": "ult", "object": "chat.completion",
                    "created": int(time.time()), "model": cfg["model_id"],
                    "choices": [{"index": 0, "message": msg,
                                 "finish_reason":
                                 "tool_calls" if calls else "stop"}],
                    "usage": {"prompt_tokens": len(ids_in),
                              "completion_tokens": len(gen_ids),
                              "total_tokens": len(ids_in) + len(gen_ids)}}
            if stream:
                return self._stream_buffered(resp)
            return self._json(200, resp)

        def _stream_head(self):
            self.close_connection = True      # body ends at close
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self._sse(self._chunk(
                {"role": "assistant", "content": ""}))

        def _stream(self, ids_in, ntok):
            """Live SSE: decode-suffix deltas, think->reasoning split."""
            self._stream_head()
            tok = cfg["tok"]
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
                            self._sse(self._chunk(
                                {"reasoning_content": delta}))
                            s["sent_reason"] = len(text)
                        return
                    s["think_done"] = True
                    think, _, rest = text.partition("</think>")
                    delta = think[s["sent_reason"]:]
                    if delta:
                        self._sse(self._chunk(
                            {"reasoning_content": delta}))
                    s["sent_reason"] = len(think)
                    s["sent_content"] = 0
                    text = rest
                delta = text[s["sent_content"]:]
                if delta:
                    self._sse(self._chunk({"content": delta}))
                    s["sent_content"] = len(text)

            try:
                got, dt = generate(cfg["profile"], cfg["tok"], cfg["eng"],
                                   ids_in, ntok, on_tok=emit)
                n_gen = len(state["ids"])
            except Exception as e:
                self._sse(self._chunk({"content": f"[ult error: {e}]"}))
                n_gen = 0
            usage = {"prompt_tokens": len(ids_in),
                     "completion_tokens": n_gen,
                     "total_tokens": len(ids_in) + n_gen}
            self._sse(self._chunk({}, finish="stop", usage=usage))
            self._sse("[DONE]")

        def _stream_buffered(self, resp):
            """tools + stream: parse first, then emit buffered chunks."""
            self._stream_head()
            msg = resp["choices"][0]["message"]
            if msg.get("reasoning_content"):
                self._sse(self._chunk(
                    {"reasoning_content": msg["reasoning_content"]}))
            if msg.get("content"):
                self._sse(self._chunk({"content": msg["content"]}))
            for i, tc in enumerate(msg.get("tool_calls") or []):
                tc["index"] = i
                self._sse(self._chunk({"tool_calls": [tc]}))
            self._sse(self._chunk({},
                                  finish=resp["choices"][0]
                                  ["finish_reason"],
                                  usage=resp["usage"]))
            self._sse("[DONE]")

    return H


def main():
    ap = argparse.ArgumentParser(description="ULT OpenAI-compatible server")
    ap.add_argument("--model", default="qwen",
                    choices=list(PROFILES), help="engine profile")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tools-limit", type=int, default=8,
                    help="cap injected tool schemas (-1 = all)")
    args = ap.parse_args()

    profile = PROFILES[args.model]
    tok = Tokenizer.from_file(profile["tok"])
    eng = profile["engine"]()
    cfg = {"profile": profile, "tok": tok, "eng": eng,
           "model_id": profile["name"],
           "tools_limit": args.tools_limit}

    # warmup: first launch after upload can emit garbage — burn one
    # short generation so the first request is clean (chat.py does this)
    bos = "<|begin▁of▁sentence|>" if args.model == "qwen" \
        else "<|startoftext|>"
    eng.generate(tok.encode(bos + "hi").ids, 4)

    print(f"ult-serve [{args.model}:{eng.quant}] "
          f"http://{args.host}:{args.port}/v1  model={cfg['model_id']}",
          flush=True)
    ThreadingHTTPServer((args.host, args.port),
                        make_handler(cfg)).serve_forever()


if __name__ == "__main__":
    main()
