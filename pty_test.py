"""Drive xera-rsi TUI under a ConPTY and emulate the screen.

Usage: python -u pty_test.py [basic|shell]
"""
import os
import re
import sys
import time
import queue
import threading

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from winpty import PTY

EXE = r"C:\Users\kovax\xera-rsi\zig-out\bin\xera-rsi.exe"
CWD = r"C:\Users\kovax\xera-rsi\test_ws"
COLS, ROWS = 110, 32

LOG = open(r"C:\Users\kovax\xera-rsi\pty_log.txt", "w", encoding="utf-8")
rxq = queue.Queue()


def log(msg):
    LOG.write(msg + "\n")
    LOG.flush()
    print(msg, flush=True)


def read_all(pty, dur=0.5):
    """Read for a fixed duration (spinner output never stops while busy)."""
    out = []
    end = time.time() + dur
    while time.time() < end:
        try:
            d = pty.read(blocking=False)
            if d:
                out.append(d)
            else:
                time.sleep(0.02)
        except Exception as e:
            sys.stderr.write(f"[read err: {e!r}]\n")
            break
    return "".join(out)


class Screen:
    """Tiny VT emulator: handles CUP, EL, ED, text. Styles ignored."""

    def __init__(self, cols=COLS, rows=ROWS):
        self.cols, self.rows = cols, rows
        self.grid = [[" "] * cols for _ in range(rows)]
        self.r = self.c = 0

    def feed(self, data: str):
        i = 0
        while i < len(data):
            ch = data[i]
            if ch == "\x1b":
                m = re.match(r"\x1b\[([0-9;?]*)([A-Za-z~@])", data[i:])
                if m:
                    self._csi(m.group(1), m.group(2))
                    i += m.end()
                    continue
                m2 = re.match(r"\x1b[()][A-Z0-9]", data[i:])
                if m2:
                    i += m2.end()
                    continue
                i += 2
                continue
            if ch == "\r":
                self.c = 0
                i += 1
                continue
            if ch == "\n":
                self.r = min(self.r + 1, self.rows - 1)
                i += 1
                continue
            if ch == "\x07" or ch < " ":
                i += 1
                continue
            if self.c >= self.cols:
                self.c = 0
                self.r = min(self.r + 1, self.rows - 1)
            import unicodedata
            w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if w == 2 and self.c + 1 < self.cols:
                self.grid[self.r][self.c + 1] = " "
            if self.r < self.rows:
                self.grid[self.r][self.c] = ch
            self.c += w
            i += 1

    def _csi(self, params: str, fin: str):
        ps = params.lstrip("?")
        nums = [int(x) for x in ps.split(";") if x.isdigit()]
        if fin in ("H", "f"):
            self.r = (nums[0] - 1) if len(nums) > 0 else 0
            self.c = (nums[1] - 1) if len(nums) > 1 else 0
            self.r = max(0, min(self.r, self.rows - 1))
            self.c = max(0, min(self.c, self.cols - 1))
        elif fin == "K":
            mode = nums[0] if nums else 0
            if mode in (0, 2):
                for x in range(self.c, self.cols):
                    self.grid[self.r][x] = " "
            if mode in (1, 2):
                for x in range(0, self.c):
                    self.grid[self.r][x] = " "
        elif fin == "J":
            mode = nums[0] if nums else 0
            if mode == 2:
                self.grid = [[" "] * self.cols for _ in range(self.rows)]
        elif fin == "A":
            self.r = max(0, self.r - (nums[0] if nums else 1))
        elif fin == "B":
            self.r = min(self.rows - 1, self.r + (nums[0] if nums else 1))
        elif fin == "C":
            self.c = min(self.cols - 1, self.c + (nums[0] if nums else 1))
        elif fin == "D":
            self.c = max(0, self.c - (nums[0] if nums else 1))
        elif fin == "G":
            self.c = (nums[0] - 1) if nums else 0

    def text(self):
        return "\n".join("".join(row).rstrip() for row in self.grid)


def show(scr, label):
    log(f"\n=== {label} ===\n{scr.text()}")


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "basic"
    mock_server = None
    model_test_error = None
    if scenario == "model":
        from http.server import HTTPServer
        from mock_llm import H, REQUESTED_MODELS

        mock_server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=mock_server.serve_forever, daemon=True).start()
        os.environ["XERA_TEST_BASE_URL"] = f"http://127.0.0.1:{mock_server.server_port}/v1"
    os.environ.update({
        "XERA_BASE_URL": os.environ.get("XERA_TEST_BASE_URL", "http://127.0.0.1:8787/v1"),
        "XERA_API_KEY": "mock",
        "XERA_MODEL": "mock-1",
    })
    pty = PTY(COLS, ROWS)
    log("[pty created]")
    pty.spawn(EXE, cwd=CWD)
    log("[spawned]")
    scr = Screen()
    if scenario == "anim":
        # catch the matrix reveal mid-flight — snapshots as frames arrive
        scr.feed(read_all(pty, 0.25))
        show(scr, "anim ~0.25s")
        scr.feed(read_all(pty, 0.35))
        show(scr, "anim ~0.6s")
        scr.feed(read_all(pty, 0.4))
        show(scr, "anim ~1.0s")
        scr.feed(read_all(pty, 0.8))
        show(scr, "anim settled")
    else:
        scr.feed(read_all(pty, 1.2))
        show(scr, "initial screen")

    if scenario in ("basic", "shell"):
        prompt = "please create the hello_xera file" if scenario == "basic" else "shell it please"
        log("[write prompt]")
        pty.write(prompt)
        scr.feed(read_all(pty, 0.8))
        show(scr, "typed prompt")
        log("[write enter]")
        pty.write("\r")
        scr.feed(read_all(pty, 2.0))
        show(scr, "after submit (approval expected)")
        log("[write y]")
        pty.write("y")
        scr.feed(read_all(pty, 4.0))
        show(scr, "after approve")

    if scenario == "model":
        pty.write("/model")
        scr.feed(read_all(pty, 0.4))
        pty.write("\r")
        scr.feed(read_all(pty, 1.0))
        show(scr, "model picker loaded")
        pty.write("mock-code")
        scr.feed(read_all(pty, 0.5))
        show(scr, "model picker filtered")
        pty.write("\r")
        scr.feed(read_all(pty, 1.0))
        show(scr, "model selected")
        pty.write("hello after model switch\r")
        scr.feed(read_all(pty, 2.0))
        show(scr, "prompt uses selected model")
        if not REQUESTED_MODELS or REQUESTED_MODELS[-1] != "mock-code":
            model_test_error = f"completion used {REQUESTED_MODELS[-1] if REQUESTED_MODELS else 'no model'}, expected mock-code"
        else:
            log("[verified request model: mock-code]")

    if scenario == "slash":
        # '/' opens the dropdown
        pty.write("/")
        scr.feed(read_all(pty, 0.6))
        show(scr, "slash open")
        # type more to narrow to /usage
        pty.write("usa")
        scr.feed(read_all(pty, 0.6))
        show(scr, "narrowed to /usa*")
        # enter selects the single match
        pty.write("\r")
        scr.feed(read_all(pty, 1.0))
        show(scr, "after dropdown enter")
        # tab completes a command name
        pty.write("/ag")
        scr.feed(read_all(pty, 0.5))
        pty.write("\t")
        scr.feed(read_all(pty, 0.5))
        show(scr, "tab-completed /agents")
        pty.write("\r")
        scr.feed(read_all(pty, 0.8))
        # other commands via direct typing
        for c in ("/skills", "/harness"):
            pty.write(c + "\r")
            scr.feed(read_all(pty, 1.0))
            show(scr, f"after {c}")

    log("[write ctrl-c]")
    pty.write("\x03")
    time.sleep(0.5)
    scr.feed(read_all(pty, 0.6))
    show(scr, "after ctrl-c (should exit)")
    log(f"alive: {pty.isalive()}")
    if pty.isalive():
        pty.write("/exit\r")
        time.sleep(0.5)
        log(f"alive after /exit: {pty.isalive()}")
    if pty.isalive():
        import signal
        os.kill(pty.pid, signal.SIGTERM)
    if mock_server is not None:
        mock_server.shutdown()
        mock_server.server_close()
    if model_test_error is not None:
        raise AssertionError(model_test_error)


if __name__ == "__main__":
    main()
