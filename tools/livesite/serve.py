#!/usr/bin/env python3
"""xera-rsi live observability server.

Serves the 3D conversation visualizer + statistics site and streams the
JSONL event logs that xera-rsi writes to ~/.xera-rsi/live/.

Zero dependencies (stdlib only). Usage:

    python3 tools/livesite/serve.py [--port 8377] [--live-dir DIR]

Then open http://localhost:8377
"""

import argparse
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
DEFAULT_LIVE_DIR = Path.home() / ".xera-rsi" / "live"
SESSION_FILE_RE = re.compile(r"^[A-Za-z0-9._-]+\.jsonl$")
SID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

LIVE_DIR = DEFAULT_LIVE_DIR


def iter_jsonl(path):
    """Yield parsed JSON objects from a .jsonl file, skipping bad lines."""
    try:
        with open(path, "rb") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except Exception:
                    continue
    except OSError:
        return


def is_session_file(name):
    return SESSION_FILE_RE.match(name) and name != "improvements.jsonl"


def list_sessions():
    """Metadata for every session file in the live dir."""
    out = []
    if not LIVE_DIR.is_dir():
        return out
    for p in sorted(LIVE_DIR.iterdir()):
        if not (p.is_file() and is_session_file(p.name)):
            continue
        sid = p.stem
        meta = {
            "id": sid,
            "model": "",
            "cwd": "",
            "name": "",
            "harness": "",
            "first_ts": 0,
            "last_ts": 0,
            "events": 0,
            "size": 0,
        }
        try:
            meta["size"] = p.stat().st_size
        except OSError:
            pass
        for ev in iter_jsonl(p):
            ts = ev.get("ts") or 0
            if ts:
                if not meta["first_ts"]:
                    meta["first_ts"] = ts
                meta["last_ts"] = ts
            meta["events"] += 1
            if ev.get("type") == "session_start":
                meta["model"] = ev.get("model") or ""
                meta["cwd"] = ev.get("cwd") or ""
                meta["name"] = ev.get("name") or ""
                meta["harness"] = ev.get("harness") or ""
        if meta["events"]:
            out.append(meta)
    out.sort(key=lambda s: s["first_ts"])
    return out


def read_events(sid, offset):
    """Read raw JSONL lines for a session starting at a byte offset."""
    if not SID_RE.match(sid or ""):
        return None
    path = LIVE_DIR / f"{sid}.jsonl"
    if not path.is_file():
        return {"off": 0, "events": [], "done": True}
    size = path.stat().st_size
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        offset = 0
    if offset > size:
        offset = 0  # file rotated/truncated — resend from start
    events = []
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(4 << 20)  # up to 4MB per poll
        new_off = f.tell()
    for line in data.splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return {"off": new_off, "events": events, "done": False}


def aggregate_stats():
    """Cross-session statistics + Harness² improvement history."""
    sessions = list_sessions()
    per_session = []
    totals = {
        "sessions": 0,
        "turns": 0,
        "tool_calls": 0,
        "tool_results_ok": 0,
        "tool_results_err": 0,
        "skills_used": 0,
        "subagents": 0,
        "reasoning_chars": 0,
        "reasoning_blocks": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_calls": 0,
        "assistant_msgs": 0,
    }
    tool_hist = {}
    skill_hist = {}
    authors = {}
    for s in sessions:
        path = LIVE_DIR / f"{s['id']}.jsonl"
        st = {
            "id": s["id"],
            "model": s["model"],
            "cwd": s["cwd"],
            "name": s["name"],
            "first_ts": s["first_ts"],
            "last_ts": s["last_ts"],
            "turns": 0,
            "tool_calls": 0,
            "skills": [],
            "subagents": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_chars": 0,
            "assistant_msgs": 0,
            "authors": {},
        }
        for ev in iter_jsonl(path):
            t = ev.get("type")
            if t == "user":
                st["turns"] += 1
                totals["turns"] += 1
                a = ev.get("author") or "you"
                st["authors"][a] = st["authors"].get(a, 0) + 1
                authors[a] = authors.get(a, 0) + 1
            elif t == "assistant":
                st["assistant_msgs"] += 1
                totals["assistant_msgs"] += 1
            elif t == "reasoning":
                n = len(ev.get("text") or "")
                st["reasoning_chars"] += n
                totals["reasoning_chars"] += n
                totals["reasoning_blocks"] += 1
            elif t == "tool_call":
                st["tool_calls"] += 1
                totals["tool_calls"] += 1
                n = ev.get("name") or "?"
                tool_hist[n] = tool_hist.get(n, 0) + 1
            elif t == "tool_result":
                if ev.get("ok"):
                    totals["tool_results_ok"] += 1
                else:
                    totals["tool_results_err"] += 1
            elif t == "skill":
                n = ev.get("name") or "?"
                st["skills"].append(n)
                totals["skills_used"] += 1
                skill_hist[n] = skill_hist.get(n, 0) + 1
            elif t == "subagent":
                st["subagents"] += 1
                totals["subagents"] += 1
            elif t == "usage":
                p = ev.get("prompt") or 0
                c = ev.get("completion") or 0
                st["prompt_tokens"] += p
                st["completion_tokens"] += c
                totals["prompt_tokens"] += p
                totals["completion_tokens"] += c
                totals["llm_calls"] += 1
        per_session.append(st)
        totals["sessions"] += 1

    improvements = []
    for ev in iter_jsonl(LIVE_DIR / "improvements.jsonl"):
        improvements.append(ev)
    improvements = improvements[-500:]
    composes = [e for e in improvements if e.get("phase") == "compose"]
    lessons = [e.get("lesson") for e in composes if e.get("lesson")]

    nsess = max(1, totals["sessions"])
    return {
        "generated_ms": int(time.time() * 1000),
        "live_dir": str(LIVE_DIR),
        "totals": totals,
        "averages": {
            "tool_calls_per_session": round(totals["tool_calls"] / nsess, 2),
            "skills_per_session": round(totals["skills_used"] / nsess, 2),
            "turns_per_session": round(totals["turns"] / nsess, 2),
            "prompt_tokens_per_session": round(totals["prompt_tokens"] / nsess, 1),
            "completion_tokens_per_session": round(totals["completion_tokens"] / nsess, 1),
        },
        "tool_histogram": sorted(tool_hist.items(), key=lambda kv: -kv[1]),
        "skill_histogram": sorted(skill_hist.items(), key=lambda kv: -kv[1]),
        "authors": authors,
        "sessions": per_session,
        "improvements": {
            "total_events": len(improvements),
            "runs_started": len([e for e in improvements if e.get("phase") == "start"]),
            "composes": len(composes),
            "dry_runs": len([e for e in improvements if e.get("phase") == "dry_run"]),
            "lessons": lessons,
            "events": improvements[-50:],
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name, ctype):
        p = HERE / name
        if not p.is_file():
            self._send(404, "not found", "text/plain")
            return
        self._send(200, p.read_bytes(), ctype)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/" or u.path == "/index.html":
            self._static("index.html", "text/html; charset=utf-8")
        elif u.path == "/app.js":
            self._static("app.js", "application/javascript; charset=utf-8")
        elif u.path == "/api/sessions":
            self._send(200, json.dumps(list_sessions()))
        elif u.path == "/api/events":
            sid = (q.get("id") or [""])[0]
            off = (q.get("off") or ["0"])[0]
            res = read_events(sid, off)
            if res is None:
                self._send(400, '{"error":"bad session id"}')
            else:
                self._send(200, json.dumps(res))
        elif u.path == "/api/stats":
            self._send(200, json.dumps(aggregate_stats()))
        elif u.path == "/api/health":
            self._send(200, '{"ok":true,"live_dir":"%s"}' % LIVE_DIR)
        else:
            self._send(404, "not found", "text/plain")


def main():
    global LIVE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8377)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--live-dir", default=os.environ.get("XERA_LIVE_DIR", str(DEFAULT_LIVE_DIR)))
    args = ap.parse_args()
    LIVE_DIR = Path(args.live_dir).expanduser()
    print(f"xera-rsi live site")
    print(f"  live dir : {LIVE_DIR}")
    print(f"  url      : http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
