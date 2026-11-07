"""Verify keys that go through the new ESC parser still behave.

Covers: arrow keys, home/end, delete, pageup, shift-tab, F-keys,
bracketed paste (multi-line + literal ESC inside), and plain typing.

Usage: python -u verify_keys.py
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pty_test import PTY, EXE, CWD, COLS, ROWS, Screen, read_all

PROMPT_ROW_MARK = "❯"


def composer_text(scr):
    """Pull the text following ❯ in the composer row (padding stripped)."""
    for ln in scr.text().split("\n"):
        if PROMPT_ROW_MARK in ln:
            after = ln.split(PROMPT_ROW_MARK, 1)[1]
            return after.replace("│", "").strip()
    return "<not found>"


def session(fn):
    os.environ.update({
        "XERA_BASE_URL": os.environ.get("XERA_BASE_URL", "http://127.0.0.1:8787/v1"),
        "XERA_API_KEY": "mock",
        "XERA_MODEL": "mock-1",
    })
    pty = PTY(COLS, ROWS)
    pty.spawn(EXE, cwd=CWD)
    scr = Screen()
    scr.feed(read_all(pty, 3.2))
    try:
        return fn(pty, scr)
    finally:
        pty.write("\x03")
        time.sleep(0.4)


def case_typing(pty, scr):
    pty.write("hello")
    scr.feed(read_all(pty, 0.5))
    return composer_text(scr)


def case_arrows(pty, scr):
    pty.write("abc")
    scr.feed(read_all(pty, 0.4))
    pty.write("\x1b[D")  # left
    pty.write("\x1b[D")  # left
    pty.write("X")
    scr.feed(read_all(pty, 0.5))
    return composer_text(scr)


def case_home_end(pty, scr):
    pty.write("abc")
    scr.feed(read_all(pty, 0.4))
    pty.write("\x1b[H")  # home
    pty.write("1")
    pty.write("\x1b[F")  # end
    pty.write("9")
    scr.feed(read_all(pty, 0.6))
    return composer_text(scr)


def case_delete(pty, scr):
    pty.write("abcd")
    scr.feed(read_all(pty, 0.4))
    pty.write("\x1b[H")   # home
    pty.write("\x1b[3~")  # delete
    scr.feed(read_all(pty, 0.5))
    return composer_text(scr)


def case_paste(pty, scr):
    # bracketed paste with a multi-line payload: the newline must be preserved
    pty.write("\x1b[200~line one\nline two\x1b[201~")
    scr.feed(read_all(pty, 0.8))
    rows = [ln for ln in scr.text().split("\n")]
    hits = [ln for ln in rows if "line one" in ln or "line two" in ln]
    return " | ".join(h.replace("❯", "").strip("│ ").strip() for h in hits)


def case_paste_with_escape(pty, scr):
    # bracketed paste whose text contains a literal ESC sequence
    pty.write("\x1b[200~a\x1b[<0;5;5Mb\x1b[201~")
    scr.feed(read_all(pty, 0.8))
    return composer_text(scr)


def case_fkeys_ignored(pty, scr):
    pty.write("ab")
    scr.feed(read_all(pty, 0.4))
    # F1, F5, shift-tab, and a DECRPM report should all be swallowed
    pty.write("\x1bOP")
    pty.write("\x1b[15~")
    pty.write("\x1b[Z")
    pty.write("\x1b[?1;2$y")
    pty.write("c")
    scr.feed(read_all(pty, 0.8))
    return composer_text(scr)


CASES = [
    ("plain typing", case_typing, "hello"),
    ("arrow left x2 + X", case_arrows, "aXbc"),
    ("home/end insert", case_home_end, "1abc9"),
    ("home + delete", case_delete, "bcd"),
    ("bracketed paste multi-line", case_paste, "line one | line two"),
    ("paste containing ESC seq", case_paste_with_escape, "a<0;5;5Mb"),
    ("F-keys/shift-tab/DECRPM ignored", case_fkeys_ignored, "abc"),
]


def main():
    fails = 0
    for name, fn, expect in CASES:
        got = session(fn)
        ok = got == expect
        if not ok:
            fails += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        expect: {expect!r}")
        print(f"        got   : {got!r}")
    print(f"\n{fails} failure(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
