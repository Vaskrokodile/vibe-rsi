"""Minimal OpenAI-compatible mock for exercising xera-rsi end-to-end.

Behavior:
- GET /v1/models -> one model
- POST /v1/chat/completions:
    * if last message is role=tool -> final answer text
    * elif last user message asks for a file -> stream a write_file tool_call,
      then (next request) a read_file, then finish with text
    * else -> streamed plain text echo
"""
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

MODELS = ("mock-1", "mock-2", "mock-code")
REQUESTED_MODELS = []


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


def chunk(delta, finish=None):
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def tool_call_msg(name, args, cid="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def full_msg(content=None, tool_calls=None):
    m = {"role": "assistant", "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


PROPOSE_JSON = json.dumps({
    "candidates": [
        {"id": "c1", "rationale": "tighten tool guidance for file ops",
         "files": [{"path": "tool_guidance.md", "op": "append",
                    "content": "- Prefer glob before grep when you know the filename."}]},
        {"id": "c2", "rationale": "domain skill for the task at hand",
         "files": [{"path": "skills/xera-task.md", "op": "write",
                    "content": "# task skill\nWhen doing the demo task, write files atomically."}]},
    ]
})

COMPOSE_JSON = json.dumps({
    "adopt": "c1",
    "rationale": "c1's probe showed fewer wasted tool calls",
    "final_files": [
        {"path": "tool_guidance.md", "op": "append",
         "content": "- Prefer glob before grep when you know the filename."}
    ],
    "lesson": "Glob-first guidance reduced search rounds in the probe.",
})


def plan(messages):
    """Decide what the mock should return."""
    sys_text = " ".join(
        str(m.get("content") or "") for m in messages if m.get("role") == "system")
    if "harness editor" in sys_text:
        return ("final", PROPOSE_JSON)
    if "harness composer" in sys_text:
        return ("final", COMPOSE_JSON)

    last = messages[-1]
    if last.get("role") == "tool":
        name = "tool"
        return ("final", f"Done — the tool ran successfully ({name}). "
                          "xera-rsi executed the call, fed me the result, "
                          "and here is my closing response.")
    for m in reversed(messages):
        if m.get("role") == "user":
            text = m.get("content") or ""
            if "hello_xera" in text:
                return ("tool", "write_file",
                        {"path": "hello_xera.txt", "content": "hello from xera-rsi\n"})
            if "shell it" in text:
                return ("tool", "run_shell",
                        {"command": "echo hello-from-shell && ver", "timeout_ms": 30000})
            if "read it back" in text:
                return ("tool", "read_file", {"path": "hello_xera.txt"})
            return ("final", f"echo: {text[:200]}")
    return ("final", "?")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps({"object": "list", "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "mock"}
                for model in MODELS
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        REQUESTED_MODELS.append(req.get("model"))
        msgs = req.get("messages", [])
        stream = req.get("stream", False)
        kind, *rest = plan(msgs)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if kind == "final":
                self.wfile.write(sse(chunk(
                    {"reasoning_content": "thinking: what does the user want? "
                                          "the prompt is simple — answer directly."})))
                self.wfile.flush()
                text = rest[0]
                for piece in re.findall(r".{1,20}", text, flags=re.S):
                    self.wfile.write(sse(chunk({"content": piece})))
                    self.wfile.flush()
                self.wfile.write(sse(chunk({}, "stop")))
            else:
                name, args = rest
                self.wfile.write(sse(chunk({"content": "Let me do that.\n"})))
                tc_delta = {
                    "tool_calls": [{
                        "index": 0, "id": "call_1", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }]
                }
                self.wfile.write(sse(chunk(tc_delta)))
                self.wfile.write(sse(chunk({}, "tool_calls")))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        # non-stream
        if kind == "final":
            msg = full_msg(content=rest[0])
            finish = "stop"
        else:
            name, args = rest
            msg = full_msg(content="Let me do that.", tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }])
            finish = "tool_calls"
        body = json.dumps({
            "id": "chatcmpl-mock", "object": "chat.completion",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8787), H).serve_forever()
