#!/usr/bin/env python3
"""ult-serve — OpenAI-compatible endpoint for the ULT (GA106) engine.

Architecture (changed after a deadlock took down a desktop GPU):

  serve.py (dispatcher, --port 8321)
    |- serve.py --worker qwen --port <ephemeral>   (own CUDA context)
    |- serve.py --worker lfm3 --port <ephemeral>   (own CUDA context)

Why per-model subprocesses: a deadlocked persistent kernel spins all SMs
forever and CANNOT be preempted — only destroying its CUDA context
(killing the process) frees the GPU. In-process retries stack zombies and
hang the display (WDDM TDR). A worker that stalls exits itself; the
dispatcher respawns it with a fresh context. Stalls/corruption in one
model can never poison the other.

Endpoints (dispatcher):
  GET  /health                -> {"status":"ok","models":{...}}
  GET  /v1/models             -> configured model ids
  POST /v1/chat/completions   -> proxied to the model's worker

Worker-only internals mirror GA106 chat.py: drain the orphan kernel,
reset mapped counters after prefill (stale-slot bug), sequential prefill,
single decode launch, poll the mapped token counter, stream SSE.

Env: GA106_HOME (default E:/GA106), ULT_QUANT (default q4),
     GA106_LFM3_VARIANT (q4|ult_exact|ult_hfma, default ult_exact),
     ULT_PREFIX_CACHE=0 to disable KV prefix caching (default on —
     turns after the first skip all but the new tail of the prompt),
     ULT_LFM3_BATCHED_PREFILL=0 to disable the one-launch mma prefill
     on LFM3 cold prompts (default on),
     ULT_MAX_GEN (cap on generated tokens, default 1024 — the kernel
     always fills ntok slots even after EOS, so uncapped requests burn
     GPU for nothing),
     ULT_GEN_TIMEOUT (per-request wall clock, default 300s).
"""
import json
import os
import sys
import time
import socket
import threading
import subprocess
import argparse
import http.client
import atexit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GA106 = os.environ.get("GA106_HOME", r"E:\GA106")
sys.path.insert(0, GA106)

MAXPOS = 8192
MAX_GEN = int(os.environ.get("ULT_MAX_GEN", "1024"))
GEN_TIMEOUT = float(os.environ.get("ULT_GEN_TIMEOUT", "900"))
STALL_TIMEOUT = float(os.environ.get("ULT_STALL_TIMEOUT", "90"))
WORKER_READY_TIMEOUT = float(os.environ.get("ULT_WORKER_READY", "420"))
_lock = threading.Lock()          # one persistent kernel at a time

PROFILES = {
    "qwen": {
        "name": "ult-qwen-1.5b",
        "tok": r"E:\DeepSeek-R1-Distill-Qwen-1.5B\tokenizer.json",
        "eos": 151643,
        "template": "r1",
    },
    "lfm3": {
        "name": "ult-lfm3-1.7b",
        "tok": os.path.join(GA106, r"models\lfm3-1.7b\tokenizer.json"),
        "eos": 124900,
        "template": "chatml",
    },
}

def log(*a):
    try:
        print(*a, file=sys.stderr, flush=True)
    except OSError:
        pass


def make_engine(name):
    if name == "qwen":
        from ga106.engine import Engine
        return Engine(os.path.join(GA106, r"models\qwen1.5b\model.safetensors"),
                      quant=os.environ.get("GA106_QUANT", "q4"))
    if name == "lfm3":
        from ga106.engine_lfm3 import EngineLFM3
        variant = os.environ.get("GA106_LFM3_VARIANT", "ult_exact")
        model = os.path.join(GA106, r"models\lfm3-1.7b\model.safetensors")
        ult = os.path.join(GA106, r"kernels\lfm3.ult")
        if variant == "q4":
            return EngineLFM3(model, quant="q4")
        if variant == "ult_hfma":
            return EngineLFM3.from_ult(model, ult,
                                       options={"hfma": 1, "fuse_conv_o": 1,
                                                "threads": 384})
        if variant == "ult_exact":
            return EngineLFM3.from_ult(model, ult,
                                       options={"fuse_conv_o": 1,
                                                "threads": 384})
        raise ValueError("GA106_LFM3_VARIANT must be q4|ult_exact|ult_hfma")
    raise ValueError(f"unknown engine '{name}'")


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


# ================= worker (single engine, own CUDA context) =================

def run_worker(model_name, host, port, tools_limit):
    """Load one engine and serve /v1 for it. On any kernel stall we
    os._exit — the deadlocked kernel can only be killed by destroying
    the CUDA context, and the dispatcher will respawn us cleanly."""
    from tokenizers import Tokenizer
    from ga106 import cuda as _c

    profile = PROFILES[model_name]
    model_id = profile["name"]

    def die(code=2):
        log(f"ult-worker[{model_name}]: exiting ({code}) so the "
            "dispatcher respawns with a fresh CUDA context")
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except OSError:
            pass
        os._exit(code)

    eng = make_engine(model_name)
    tok = Tokenizer.from_file(profile["tok"])

    # KV/conv-state prefix cache: positions are absolute and the kv/conv
    # buffers persist across requests, so a chat turn only needs to
    # prefill tokens past the longest shared prefix — a 1600-token agent
    # prompt re-prefills ~20 tokens, not 1600. Set ULT_PREFIX_CACHE=0 to
    # force full prefill every request.
    prefix_cache = os.environ.get("ULT_PREFIX_CACHE", "1") != "0"
    # "safe" = the last request finished decoding cleanly, so an orphan
    # found running now is just a post-EOS chunk tail — it only wrote
    # positions past our recorded prefix, which stays valid. If a request
    # died mid-prefill/decode, KV may be corrupt -> drop the cache.
    kv_state = {"ids": [], "safe": True}

    # warmup: the first launch after upload can emit garbage — burn one
    # short generation so the first user message is clean
    bos = "<|begin▁of▁sentence|>" if model_name == "qwen" else "<|startoftext|>"
    try:
        eng.generate(tok.encode(bos + "hi").ids, 4)
    except Exception as e:
        log(f"ult-worker[{model_name}]: warmup failed: {e}")
        die(2)

    # pre-build the batched-prefill kernel at startup (W5 repack + NVRTC
    # is ~40s once, cached on disk) so the first cold request stays fast
    lfm3_batched = (model_name == "lfm3"
                    and os.environ.get("ULT_LFM3_BATCHED_PREFILL", "1")
                        != "0"
                    and hasattr(eng, "_init_batched_prefill"))
    if lfm3_batched:
        try:
            eng._init_batched_prefill()
        except Exception as e:
            lfm3_batched = False
            log(f"ult-worker[{model_name}]: batched prefill init "
                f"failed, sequential fallback: {e}")
    log(f"ult-worker[{model_name}]: ready on :{port}")

    eos_id = profile["eos"]

    def poisoned_exit():
        threading.Timer(0.5, die, args=(2,)).start()

    def generate(ids_in, ntok, on_tok=None):
        """Prefill sequentially, then one decode launch; poll the mapped
        token counter. Mirrors chat.py — do not reorder. Returns
        (n_kernel_tokens, elapsed_s)."""
        with _lock:
            # drain any orphan kernel from a disconnected request — it
            # keeps writing tokens[] while running. A drained orphan also
            # wrote KV we can't attribute -> prefix cache is invalid.
            t_drain = time.perf_counter()
            was_busy = not _c.stream_query(eng.stream)
            while not _c.stream_query(eng.stream):
                if time.perf_counter() - t_drain > 30:
                    raise TimeoutError("previous generation still active")
                time.sleep(0.001)
            if was_busy and not kv_state["safe"]:
                kv_state["ids"] = []
            kv_state["safe"] = False   # until this request finishes
            eng._tok_host[0] = 0
            eng._tok_host[MAXPOS + 4] = -1
            _c.memset_dev(eng.d_bar, 0, 64)
            _c.memset_dev(eng.d_plog, 0, (4 * 56 + 64 + 4096) * 4)
            _c.sync()

            # skip prefill for the longest cached prefix — kv/conv state
            # at those positions is still valid from the last request
            c = 0
            if prefix_cache:
                cached = kv_state["ids"]
                lim = min(len(cached), len(ids_in) - 1)
                while c < lim and cached[c] == ids_in[c]:
                    c += 1
            skipped = c
            batched = (skipped == 0 and lfm3_batched
                       and len(ids_in) > 64)
            if batched:
                # one mma-chunked launch for the whole prompt instead of
                # ~1600 sequential ones; leaves kv/conv in the same
                # state (upstream: "same state ... up to f16 rounding")
                log(f"ult-worker[{model_name}]: batched prefill "
                    f"({len(ids_in) - 1} tok)")
                eng._prefill_batched(ids_in)
            else:
                if skipped:
                    log(f"ult-worker[{model_name}]: prefix hit — "
                        f"prefilling {len(ids_in) - 1 - skipped}/"
                        f"{len(ids_in) - 1} tok")
                for i in range(skipped, len(ids_in) - 1):
                    eng._launch(ids_in[i], i, 1)
                    eng._wait()

            # prefill launches bump tokens[0]/[1] — reset so the poll
            # only counts DECODE tokens (the stale-slot bug)
            eng._tok_host[0] = 0
            _c.sync()

            t0 = time.perf_counter()
            stall_deadline = t0 + STALL_TIMEOUT   # renewed on progress
            hard_deadline = t0 + GEN_TIMEOUT      # absolute ceiling
            pos = len(ids_in) - 1
            start_tok = ids_in[-1]
            emitted = 0
            raw = []            # every emitted id — KV was written for it
            eos = False
            # the kernel ALWAYS generates exactly ntok tokens — no EOS
            # early-stop. Chunk the decode so an EOS at token 5 doesn't
            # burn a 1024-slot kernel; the tail of the last chunk keeps
            # running as a bounded orphan the next request drains.
            CHUNK = 128
            while emitted < ntok and not eos:
                want = min(CHUNK, ntok - emitted)
                eng._tok_host[0] = 0       # count only this chunk
                _c.sync()
                eng._launch(start_tok, pos, want)
                got = 0
                while got < want:
                    c = eng._tok_host[0]
                    if got < c:
                        new_ids = []
                        while got < c and got < want:
                            got += 1
                            tid = eng._tok_host[got]
                            # never decode a garbage slot — a poisoned
                            # context must error, not emit text
                            if not 0 <= tid < 200000:
                                raise RuntimeError(
                                    f"garbage token id {tid} at slot "
                                    f"{got} (context likely corrupted)")
                            raw.append(tid)
                            if tid == eos_id:
                                eos = True
                                continue
                            if not eos:
                                new_ids.append(tid)
                        if new_ids and on_tok:
                            on_tok(new_ids)
                        stall_deadline = time.perf_counter() \
                            + STALL_TIMEOUT
                        if eos or _c.stream_query(eng.stream):
                            break
                    elif got < want:
                        now = time.perf_counter()
                        if now > stall_deadline or now > hard_deadline:
                            hb = eng._tok_host[MAXPOS + 4]
                            raise TimeoutError(
                                f"decode stalled at {emitted + got}/"
                                f"{ntok} tokens; heartbeat={hb}")
                        time.sleep(0.0005)
                emitted += got
                if eos or got < want:
                    # EOS: leave the rest of the chunk running as an
                    # orphan — don't stream_sync or we wait for all of it
                    break
                _c.stream_sync(eng.stream)
                if not raw:
                    break
                start_tok = raw[-1]      # last emitted = next start token
                pos += got               # positions are absolute
            if not eos:
                _c.stream_sync(eng.stream)
            # every emitted slot occupies an absolute position — the
            # processed prefix is valid kv/conv state next time
            kv_state["ids"] = (ids_in + raw)[:MAXPOS - 1]
            kv_state["safe"] = True
        return emitted, time.perf_counter() - t0

    class WH(BaseHTTPRequestHandler):
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
                 "created": int(time.time()), "model": model_id,
                 "choices": [{"index": 0, "delta": delta,
                              "finish_reason": finish}]}
            if usage:
                c["usage"] = usage
            return c

        def _fail_after_response(self, code, obj):
            """Send the error, then suicide — a stalled kernel is only
            killed by destroying this process's CUDA context."""
            try:
                self._json(code, obj)
                self.wfile.flush()
            except OSError:
                pass
            poisoned_exit()

        def do_GET(self):
            _c.ctx_set_current()
            if self.path == "/health":
                return self._json(200, {"status": "ok", "model": model_id})
            if self.path in ("/v1/models", "/models"):
                data = [{"id": model_id, "object": "model",
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
            if tools and tools_limit >= 0:
                tools = tools[:tools_limit] or None
            ntok = int(req.get("max_completion_tokens")
                       or req.get("max_tokens") or 512)
            stream = bool(req.get("stream"))

            prompt = render_prompt(profile, messages, tools)
            ids_in = tok.encode(prompt).ids
            # the kernel always emits all ntok slots even after EOS —
            # clamp so an agent's 16k max_tokens doesn't burn 25s/turn
            ntok = max(1, min(ntok, MAX_GEN, MAXPOS - len(ids_in) - 8))
            if len(ids_in) + ntok > MAXPOS:
                budget = MAXPOS - ntok - 8
                if budget < 128:
                    return self._json(413, {
                        "error": "prompt too long for MAXPOS"})
                keep = budget // 3
                ids_in = ids_in[:keep] + ids_in[len(ids_in) - (budget - keep):]
                log(f"ult-worker[{model_name}]: prompt truncated to fit "
                    "MAXPOS")
            log(f"ult-worker[{model_name}]: prompt={len(ids_in)}tok "
                f"gen<={ntok} stream={stream}")

            if stream:
                # live-stream even with tools: raw text lands as it is
                # generated (no 300s of silence), parsed tool_calls are
                # appended as a proper delta at the end
                return self._stream(ids_in, ntok, tools_mode=bool(tools))
            try:
                gen_ids = []
                generate(ids_in, ntok, on_tok=gen_ids.extend)
            except Exception as e:
                return self._fail_after_response(500,
                                               {"error": f"engine: {e}"})
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
            return self._json(200, resp)

        def _stream_head(self):
            self.close_connection = True      # body ends at close
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self._sse(self._chunk({"role": "assistant", "content": ""}))

        def _stream(self, ids_in, ntok, tools_mode=False):
            """Live SSE: decode-suffix deltas, think->reasoning split.
            tools_mode: also emit parsed tool_calls at the end."""
            self._stream_head()
            state = {"ids": [], "think_done": False,
                     "sent_reason": 0, "sent_content": 0}

            def emit(new_ids):
                s = state
                s["ids"].extend(new_ids)
                text = tok.decode(s["ids"], skip_special_tokens=True)
                # decoded length can shrink when a special token is
                # dropped mid-stream — never slice beyond the end
                s["sent_reason"] = min(s["sent_reason"], len(text))
                s["sent_content"] = min(s["sent_content"], len(text))
                if not s["think_done"]:
                    if "</think>" not in text:
                        delta = text[s["sent_reason"]:]
                        if delta:
                            self._sse(self._chunk({"reasoning_content":
                                                   delta}))
                            s["sent_reason"] = len(text)
                        return
                    s["think_done"] = True
                    think, _, rest = text.partition("</think>")
                    delta = think[s["sent_reason"]:]
                    if delta:
                        self._sse(self._chunk({"reasoning_content":
                                               delta}))
                    s["sent_reason"] = len(think)
                    s["sent_content"] = 0
                    text = rest
                delta = text[s["sent_content"]:]
                if delta:
                    self._sse(self._chunk({"content": delta}))
                    s["sent_content"] = len(text)

            try:
                got, dt = generate(ids_in, ntok, on_tok=emit)
                n_gen = len(state["ids"])
            except Exception as e:
                try:
                    self._sse(self._chunk({"content":
                                           f"[ult error: {e}]"}))
                    self._sse(self._chunk({}, finish="error"))
                    self._sse("[DONE]")
                    self.wfile.flush()
                except OSError:
                    pass
                poisoned_exit()     # stalled/poisoned ctx must die
                return
            usage = {"prompt_tokens": len(ids_in),
                     "completion_tokens": n_gen,
                     "total_tokens": len(ids_in) + n_gen}
            finish = "stop"
            if tools_mode and state["ids"]:
                text = tok.decode(state["ids"], skip_special_tokens=True)
                _, clean = split_think(text)
                _, calls = split_tool_calls(clean)
                if calls:
                    finish = "tool_calls"
                    for i, tc in enumerate(calls):
                        tc["index"] = i
                        self._sse(self._chunk({"tool_calls": [tc]}))
            self._sse(self._chunk({}, finish=finish, usage=usage))
            self._sse("[DONE]")

    class QuietServer(ThreadingHTTPServer):
        # Windows SO_REUSEADDR permits a second process to bind the same
        # port — a silent split-brain. Refuse it instead.
        allow_reuse_address = False
        daemon_threads = True

        def handle_error(self, request, client_address):
            # client disconnects mid-stream are routine
            pass

    try:
        QuietServer((host, port), WH).serve_forever()
    finally:
        die(0)


# ================= dispatcher =================

class Worker:
    """One model = one subprocess = one CUDA context."""
    def __init__(self, name, tools_limit):
        self.name = name
        self.tools_limit = tools_limit
        self.proc = None
        self.port = None
        self.ready = threading.Event()
        self.loading = False
        self.restarts = 0
        self.lock = threading.Lock()

    def _free_port(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p

    def spawn(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return
            self.ready.clear()
            self.loading = True
            self.port = self._free_port()
            self.proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__),
                 "--worker", self.name,
                 "--port", str(self.port),
                 "--tools-limit", str(self.tools_limit)])
            self.restarts += 1
            threading.Thread(target=self._watch, daemon=True).start()
            log(f"ult-serve: spawned {self.name} worker pid="
                f"{self.proc.pid} :{self.port}")

    def _watch(self):
        proc = self.proc
        deadline = time.perf_counter() + WORKER_READY_TIMEOUT
        while proc.poll() is None and time.perf_counter() < deadline:
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.port,
                                                  timeout=2)
                conn.request("GET", "/health")
                r = conn.getresponse()
                r.read()
                conn.close()
                if r.status == 200:
                    self.ready.set()
                    self.loading = False
                    log(f"ult-serve: {self.name} worker ready")
                    proc.wait()
                    log(f"ult-serve: {self.name} worker exited "
                        f"(rc={proc.returncode})")
                    return
            except Exception:
                pass
            time.sleep(1.5)
        self.loading = False
        if proc.poll() is None:
            log(f"ult-serve: {self.name} worker never became ready")
        else:
            log(f"ult-serve: {self.name} worker died during load "
                f"(rc={proc.returncode})")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def kill(self):
        with self.lock:
            p = self.proc
            self.proc = None
        if p and p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass
        self.ready.clear()


def make_dispatcher_handler(models, tools_limit):
    workers = {name: Worker(name, tools_limit) for name in models}
    for w in workers.values():
        w.spawn()

    def cleanup():
        for w in workers.values():
            w.kill()
    atexit.register(cleanup)

    def resolve(req_model):
        for name, w in workers.items():
            if req_model in (name, PROFILES[name]["name"]):
                return w
        return workers[models[0]]

    class DH(BaseHTTPRequestHandler):
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

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {
                    "status": "ok",
                    "models": {n: ("ready" if w.ready.is_set()
                                   else "loading" if w.alive()
                                   else "dead")
                               for n, w in workers.items()}})
            if self.path in ("/v1/models", "/models"):
                data = [{"id": PROFILES[n]["name"], "object": "model",
                         "created": int(time.time()), "owned_by": "ult"}
                        for n in workers]
                return self._json(200, {"object": "list", "data": data})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/v1/chat/completions",
                                 "/chat/completions"):
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                req = json.loads(body)
            except Exception:
                return self._json(400, {"error": "bad json"})

            w = resolve(req.get("model") or "")
            if not w.ready.is_set():
                if w.alive() and w.loading:
                    return self._json(503, {
                        "error": f"{PROFILES[w.name]['name']} is still "
                                 "loading (NVRTC compile — retry shortly)"})
                # dead — one respawn in flight at a time
                if not w.loading:
                    w.spawn()
                return self._json(503, {
                    "error": f"{PROFILES[w.name]['name']} crashed and is "
                             "restarting — retry shortly"})

            # one kernel at a time across the whole GPU
            with _lock:
                try:
                    conn = http.client.HTTPConnection(
                        "127.0.0.1", w.port, timeout=GEN_TIMEOUT)
                    conn.request("POST", "/v1/chat/completions",
                                 body=body,
                                 headers={"Content-Type":
                                          "application/json"})
                    resp = conn.getresponse()
                except Exception as e:
                    return self._json(502, {"error": f"worker: {e}"})

                if resp.status != 200:
                    payload = resp.read()
                    conn.close()
                    # a worker that stalled exits itself; force respawn
                    if not w.alive():
                        w.spawn()
                    return self._relay(resp.status, payload)

                want_stream = bool(req.get("stream")) and \
                    "text/event-stream" in \
                    (resp.getheader("Content-Type") or "")
                try:
                    if want_stream:
                        self.close_connection = True
                        self.send_response(200)
                        self.send_header("Content-Type",
                                         "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        while True:
                            chunk = resp.read1(8192)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    else:
                        payload = resp.read()
                        conn.close()
                        return self._relay(200, payload)
                except (TimeoutError, OSError, http.client.HTTPException) as e:
                    # mid-stream stall: the worker's kernel may be
                    # deadlocked — kill it; respawn happens on next use
                    log(f"ult-serve: {w.name} stream failed: {e} — "
                        "killing worker")
                    w.kill()
                    w.spawn()
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

        def _relay(self, status, payload):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    class QuietServer(ThreadingHTTPServer):
        allow_reuse_address = False   # never two dispatchers on one port
        daemon_threads = True

        def handle_error(self, request, client_address):
            pass

    return DH, QuietServer


def main():
    ap = argparse.ArgumentParser(description="ULT OpenAI-compatible server")
    ap.add_argument("--model", default="all",
                    help="engine profile(s): qwen, lfm3, all, or comma list")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tools-limit", type=int, default=8,
                    help="cap injected tool schemas (-1 = all)")
    ap.add_argument("--worker", default=None,
                    help=argparse.SUPPRESS)   # internal: worker mode
    args = ap.parse_args()

    if args.worker:
        if args.worker not in PROFILES:
            ap.error(f"unknown model '{args.worker}'")
        return run_worker(args.worker, args.host, args.port,
                          args.tools_limit)

    wanted = [s.strip() for s in args.model.split(",") if s.strip()]
    if wanted == ["all"]:
        wanted = list(PROFILES)
    for name in wanted:
        if name not in PROFILES:
            ap.error(f"unknown model '{name}' (choices: {list(PROFILES)})")

    # fail fast before spawning workers — a second dispatcher on the same
    # port is a silent split-brain (two contexts, random routing)
    probe = socket.socket()
    try:
        probe.bind((args.host, args.port))
    except OSError:
        probe.close()
        print(f"ult-serve: {args.host}:{args.port} is already in use — "
              "is another instance running?", file=sys.stderr, flush=True)
        sys.exit(1)
    probe.close()

    DH, QuietServer = make_dispatcher_handler(wanted, args.tools_limit)
    names = [PROFILES[n]["name"] for n in wanted]
    print(f"ult-serve [{wanted[0]}] http://{args.host}:{args.port}/v1  "
          f"models={names}", flush=True)
    print("ult-serve: each model runs in its own process; a stalled "
          "kernel kills only that worker and it respawns", flush=True)
    QuietServer((args.host, args.port), DH).serve_forever()


if __name__ == "__main__":
    main()
