//! Raw-mode terminal layer: alternate screen, key input, ANSI drawing.
//! Windows: ReadConsoleInputW key events + VT output. POSIX: termios + escape seqs.

const std = @import("std");
const builtin = @import("builtin");
const win = builtin.os.tag == .windows;

pub const Key = union(enum) {
    char: u21,
    enter,
    backspace,
    delete,
    up,
    down,
    left,
    right,
    home,
    end,
    page_up,
    page_down,
    escape,
    tab,
    ctrl_c,
    ctrl_d,
    ctrl_l,
    resize,
    paste: []const u8, // bracketed-paste payload (caller owns until handled)
    /// 0-based terminal cell coords; btn: 0=left,1=middle,2=right
    mouse: struct { col: u16, row: u16, btn: u8, down: bool, move: bool = false },
};

/// ESC-sequence parser state for VT input (Windows delivers bytes, not keys).
pub const InputState = enum { ground, esc, csi, ss3, x10, paste, paste_esc, paste_csi };

pub const Term = struct {
    in: std.fs.File,
    out: std.fs.File,
    active: bool = false,
    w: u16 = 80,
    h: u16 = 24,
    // saved modes
    saved_in_mode: u32 = 0,
    saved_out_mode: u32 = 0,
    old_termios: ?SavedTermios = null,

    // VT input parsing. Windows hands us ESC sequences as ordinary key
    // events, so bytes are reassembled here instead of inserted as text.
    state: InputState = .ground,
    seq: [32]u8 = undefined,
    seq_len: usize = 0,
    x10: [3]u8 = undefined,
    x10_need: u8 = 0,
    pending_high: u16 = 0, // UTF-16 high surrogate awaiting its low half
    moji: [4]u8 = undefined, // 0x80..0xFF chars pending UTF-8 repair
    moji_len: u8 = 0,
    paste_buf: std.ArrayList(u8) = undefined,
    paste_alloc: std.mem.Allocator = undefined,

    // ---------- lifecycle ----------

    pub fn enter(alloc: std.mem.Allocator) !Term {
        var t = Term{ .in = std.io.getStdIn(), .out = std.io.getStdOut() };
        t.paste_alloc = alloc;
        t.paste_buf = std.ArrayList(u8).init(alloc);
        if (win) {
            const hin = std.os.windows.kernel32.GetStdHandle(std.os.windows.STD_INPUT_HANDLE).?;
            const hout = std.os.windows.kernel32.GetStdHandle(std.os.windows.STD_OUTPUT_HANDLE).?;
            var im: u32 = 0;
            var om: u32 = 0;
            _ = std.os.windows.kernel32.GetConsoleMode(hin, &im);
            _ = std.os.windows.kernel32.GetConsoleMode(hout, &om);
            t.saved_in_mode = im;
            t.saved_out_mode = om;
            const ENABLE_EXTENDED_FLAGS: u32 = 0x0080;
            const ENABLE_VIRTUAL_TERMINAL_INPUT: u32 = 0x0200;
            const ENABLE_WINDOW_INPUT: u32 = 0x0008;
            const ENABLE_MOUSE_INPUT: u32 = 0x0010;
            // no line input, no echo, VT in (bracketed paste), window + mouse events
            // (QUICK_EDIT intentionally off so mouse events reach the app)
            _ = std.os.windows.kernel32.SetConsoleMode(hin, ENABLE_EXTENDED_FLAGS | ENABLE_VIRTUAL_TERMINAL_INPUT | ENABLE_WINDOW_INPUT | ENABLE_MOUSE_INPUT);
            const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
            const DISABLE_NEWLINE_AUTO_RETURN: u32 = 0x0008;
            _ = std.os.windows.kernel32.SetConsoleMode(hout, om | ENABLE_VIRTUAL_TERMINAL_PROCESSING | DISABLE_NEWLINE_AUTO_RETURN);
        } else {
            const fd = t.in.handle;
            const old = try std.posix.tcgetattr(fd);
            t.old_termios = old;
            var raw = old;
            raw.lflag.ICANON = false;
            raw.lflag.ECHO = false;
            raw.lflag.ISIG = false;
            raw.iflag.IXON = false;
            raw.iflag.ICRNL = false;
            raw.cc[@intFromEnum(std.posix.V.MIN)] = 0;
            raw.cc[@intFromEnum(std.posix.V.TIME)] = 0;
            try std.posix.tcsetattr(fd, .FLUSH, raw);
        }
        t.out.writeAll("\x1b[?1049h\x1b[?2004h\x1b[?25l\x1b[?1003h\x1b[?1006h") catch {};
        t.active = true;
        try t.querySize();
        return t;
    }

    pub fn leave(self: *Term) void {
        if (!self.active) return;
        self.out.writeAll("\x1b[?1003l\x1b[?2004l\x1b[?25h\x1b[?1049l") catch {};
        if (win) {
            const hin = std.os.windows.kernel32.GetStdHandle(std.os.windows.STD_INPUT_HANDLE).?;
            const hout = std.os.windows.kernel32.GetStdHandle(std.os.windows.STD_OUTPUT_HANDLE).?;
            _ = std.os.windows.kernel32.SetConsoleMode(hin, self.saved_in_mode);
            _ = std.os.windows.kernel32.SetConsoleMode(hout, self.saved_out_mode);
        } else if (self.old_termios) |old| {
            std.posix.tcsetattr(self.in.handle, .FLUSH, old) catch {};
        }
        self.active = false;
    }

    pub fn querySize(self: *Term) !void {
        if (win) {
            const hout = std.os.windows.kernel32.GetStdHandle(std.os.windows.STD_OUTPUT_HANDLE).?;
            var info: CONSOLE_SCREEN_BUFFER_INFO = undefined;
            if (GetConsoleScreenBufferInfo(hout, &info) != 0) {
                self.w = @intCast(info.srWindow.Right - info.srWindow.Left + 1);
                self.h = @intCast(info.srWindow.Bottom - info.srWindow.Top + 1);
            }
        } else {
            var ws: std.posix.winsize = undefined;
            const rc = std.posix.system.ioctl(self.out.handle, std.posix.T.IOCGWINSZ, @intFromPtr(&ws));
            if (std.posix.errno(rc) == .SUCCESS and ws.col > 0) {
                self.w = ws.col;
                self.h = ws.row;
            }
        }
    }

    // ---------- input ----------

    /// Poll for key events, waiting up to `wait_ms`. Appends to `keys`.
    pub fn pollKeys(self: *Term, alloc: std.mem.Allocator, keys: *std.ArrayList(Key), wait_ms: u32) !void {
        if (win) try self.pollKeysWin(alloc, keys, wait_ms) else try self.pollKeysPosix(alloc, keys, wait_ms);
    }

    fn pollKeysWin(self: *Term, alloc: std.mem.Allocator, keys: *std.ArrayList(Key), wait_ms: u32) !void {
        _ = alloc;
        const hin = std.os.windows.kernel32.GetStdHandle(std.os.windows.STD_INPUT_HANDLE).?;
        std.os.windows.WaitForSingleObject(hin, wait_ms) catch return; // timeout => no input
        var recs: [32]INPUT_RECORD = undefined;
        var n: u32 = 0;
        if (ReadConsoleInputW(hin, &recs, recs.len, &n) == 0) return;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const rec = recs[i];
            switch (rec.EventType) {
                4 => { // WINDOW_BUFFER_SIZE_EVENT
                    try self.flushMoji(keys);
                    try keys.append(.resize);
                },
                2 => { // MOUSE_EVENT
                    try self.flushMoji(keys);
                    const m = rec.Event.MouseEvent;
                    const flags = m.dwEventFlags;
                    if (flags == 0) { // button press/release, not move/wheel
                        const col: u16 = @intCast(m.dwMousePosition & 0xFFFF);
                        const row: u16 = @intCast((m.dwMousePosition >> 16) & 0xFFFF);
                        const btn: u8 = if ((m.dwButtonState & 0x2) != 0) 2 else if ((m.dwButtonState & 0x4) != 0) 1 else 0;
                        if (m.dwButtonState != 0) {
                            try keys.append(.{ .mouse = .{ .col = col, .row = row, .btn = btn, .down = true } });
                        } else {
                            try keys.append(.{ .mouse = .{ .col = col, .row = row, .btn = 0, .down = false } });
                        }
                    } else if (flags == 1) { // MOUSE_MOVED
                        const col: u16 = @intCast(m.dwMousePosition & 0xFFFF);
                        const row: u16 = @intCast((m.dwMousePosition >> 16) & 0xFFFF);
                        const btn: u8 = if ((m.dwButtonState & 0x2) != 0) 2 else if ((m.dwButtonState & 0x4) != 0) 1 else 0;
                        try keys.append(.{ .mouse = .{ .col = col, .row = row, .btn = btn, .down = false, .move = true } });
                    } else if (flags == 4) { // MOUSE_WHEELED
                        // wheel up = positive delta in high word
                        const delta: i16 = @bitCast(@as(u16, @intCast((m.dwButtonState >> 16) & 0xFFFF)));
                        const col: u16 = @intCast(m.dwMousePosition & 0xFFFF);
                        const row: u16 = @intCast((m.dwMousePosition >> 16) & 0xFFFF);
                        _ = col;
                        _ = row;
                        try keys.append(if (delta > 0) .page_up else .page_down);
                    }
                },
                1 => { // KEY_EVENT
                    const k = rec.Event.KeyEvent;
                    if (k.bKeyDown == 0) continue;
                    const ctrl = (k.dwControlKeyState & 0x000C) != 0; // left|right ctrl
                    const uc: u32 = k.UnicodeChar;
                    // Anything that can't be a UTF-8 byte char ends a
                    // pending mojibake-repair sequence (below).
                    if (uc < 0x80 or uc > 0xFF) try self.flushMoji(keys);
                    // Printable characters. With ENABLE_VIRTUAL_TERMINAL_INPUT the
                    // console delivers escape sequences (mouse reports, focus events,
                    // CSI arrow keys) here too, so they are reassembled by feedByte()
                    // instead of being inserted into the composer as literal text.
                    if (uc >= 0x20 and uc != 0x7f) {
                        // UnicodeChar is a UTF-16 code unit — re-encode to
                        // UTF-8 before feeding the byte parser. A bare
                        // @intCast would panic on chars > U+00FF and
                        // mojibake U+0080..U+00FF as lone bytes.
                        var cp = uc;
                        if (cp >= 0xD800 and cp <= 0xDBFF) {
                            self.pending_high = @intCast(cp);
                            continue;
                        }
                        if (cp >= 0xDC00 and cp <= 0xDFFF) {
                            if (self.pending_high == 0) continue;
                            cp = 0x10000 + ((@as(u32, self.pending_high) - 0xD800) << 10) + (cp - 0xDC00);
                            self.pending_high = 0;
                        } else self.pending_high = 0;
                        if (cp <= 0xFF) {
                            // Some hosts decode pasted UTF-8 bytes into
                            // code-page chars first (uc = a raw byte).
                            // Buffer them: a completed valid sequence is
                            // re-encoded as real UTF-8, a broken one is
                            // treated as ordinary Latin-1 input.
                            self.moji[self.moji_len] = @intCast(cp);
                            self.moji_len += 1;
                            const want = std.unicode.utf8ByteSequenceLength(self.moji[0]) catch blk: {
                                try self.flushMoji(keys);
                                break :blk 0;
                            };
                            if (want > 0 and self.moji_len >= want) {
                                if (std.unicode.utf8Decode(self.moji[0..want])) |real| {
                                    self.moji_len = 0;
                                    try self.feedCp(keys, real);
                                } else |_| try self.flushMoji(keys);
                            }
                            continue;
                        }
                        try self.feedCp(keys, cp);
                        continue;
                    }
                    // Inside a bracketed paste every byte is literal text,
                    // including newlines and ESC — route it to the parser.
                    if (self.state == .paste or self.state == .paste_esc or self.state == .paste_csi) {
                        try self.feedByte(keys, @intCast(uc & 0xFF));
                        continue;
                    }
                    // control chars: real keys AND PTY-injected input both
                    // carry UnicodeChar (vk may be 0 for injected input)
                    switch (uc) {
                        0x0D => {
                            try keys.append(.enter);
                            continue;
                        },
                        0x0A => continue,
                        0x03 => {
                            try keys.append(.ctrl_c);
                            continue;
                        },
                        0x04 => {
                            try keys.append(.ctrl_d);
                            continue;
                        },
                        0x0C => {
                            try keys.append(.ctrl_l);
                            continue;
                        },
                        0x08, 0x7F => {
                            try keys.append(.backspace);
                            continue;
                        },
                        0x1B => {
                            try self.feedByte(keys, 0x1b);
                            continue;
                        },
                        0x09 => {
                            try keys.append(.tab);
                            continue;
                        },
                        else => {},
                    }
                    if (ctrl) {
                        switch (k.wVirtualKeyCode) {
                            'C' => try keys.append(.ctrl_c),
                            'D' => try keys.append(.ctrl_d),
                            'L' => try keys.append(.ctrl_l),
                            else => {},
                        }
                        continue;
                    }
                    // keys with no UnicodeChar: navigation
                    switch (k.wVirtualKeyCode) {
                        0x08 => try keys.append(.backspace),
                        0x2E => try keys.append(.delete),
                        0x26 => try keys.append(.up),
                        0x28 => try keys.append(.down),
                        0x25 => try keys.append(.left),
                        0x27 => try keys.append(.right),
                        0x24 => try keys.append(.home),
                        0x23 => try keys.append(.end),
                        0x21 => try keys.append(.page_up),
                        0x22 => try keys.append(.page_down),
                        else => {},
                    }
                },
                else => {},
            }
        }
        // A buffered char that didn't complete a UTF-8 sequence by the end
        // of the batch is ordinary Latin-1 input — don't leave it pending.
        try self.flushMoji(keys);
        // A sequence still open here was truncated by the console (or the
        // window closed). Never leak its bytes into the composer.
        if (self.state == .csi and self.seq_len > 0) {
            try self.emitCsiBody(keys, 0, false);
        }
        self.state = .ground;
        self.seq_len = 0;
    }

    /// Reassemble VT sequences byte by byte. With ENABLE_VIRTUAL_TERMINAL_INPUT
    /// the Windows console delivers every escape byte as a KEY_EVENT, so without
    /// this the bytes land in the composer as literal junk like "[<35;40;10M".
    fn feedByte(self: *Term, keys: *std.ArrayList(Key), b: u8) !void {
        switch (self.state) {
            .ground => {
                if (b == 0x1b) {
                    self.state = .esc;
                } else {
                    try keys.append(.{ .char = b });
                }
            },

            // ESC seen: decide what kind of sequence this is
            .esc => {
                if (b == '[') {
                    self.state = .csi;
                    self.seq_len = 0;
                } else if (b == 'O') {
                    self.state = .ss3;
                } else {
                    // bare ESC followed by an ordinary key
                    self.state = .ground;
                    try keys.append(.escape);
                    try self.feedByte(keys, b);
                }
            },

            .ss3 => {
                self.state = .ground;
                try self.emitFinal(keys, b);
            },

            .csi => {
                // ESC [ M is the classic X10 mouse: 3 raw bytes follow
                if (b == 'M' and self.seq_len == 0) {
                    self.state = .x10;
                    self.x10_need = 3;
                    return;
                }
                const final = b >= 0x40 and b <= 0x7e; // VT: final byte range
                if (!final) {
                    // parameter or intermediate byte
                    if (self.seq_len < self.seq.len) {
                        self.seq[self.seq_len] = b;
                        self.seq_len += 1;
                    }
                    return;
                }
                // complete CSI
                self.state = .ground;
                try self.emitCsiBody(keys, b, false);
            },

            .x10 => {
                const idx: usize = 3 - self.x10_need;
                if (idx < 3) self.x10[idx] = b;
                self.x10_need -= 1;
                if (self.x10_need == 0) {
                    self.state = .ground;
                    try self.emitX10(keys);
                }
            },

            // ---- bracketed paste: body is raw text until ESC[201~ ----
            .paste => {
                if (b == 0x1b) {
                    self.state = .paste_esc;
                } else {
                    try self.paste_buf.append(b);
                }
            },

            .paste_esc => {
                if (b == '[') {
                    self.state = .paste_csi;
                    self.seq_len = 0;
                } else {
                    try self.paste_buf.append(0x1b);
                    try self.paste_buf.append(b);
                    self.state = .paste;
                }
            },

            .paste_csi => {
                const final = b >= 0x40 and b <= 0x7e; // VT: final byte range
                if (!final) {
                    if (self.seq_len < self.seq.len) {
                        self.seq[self.seq_len] = b;
                        self.seq_len += 1;
                    }
                    return;
                }
                const pp = self.seq[0..self.seq_len];
                if (b == '~' and std.mem.eql(u8, pp, "201")) {
                    // end of paste — hand the text over as one key
                    self.state = .ground;
                    self.seq_len = 0;
                    if (self.paste_buf.items.len > 0) {
                        try keys.append(.{ .paste = try self.paste_buf.toOwnedSlice() });
                        self.paste_buf = std.ArrayList(u8).init(self.paste_alloc);
                    }
                } else {
                    // some other sequence inside the pasted text: keep it literal
                    try self.paste_buf.appendSlice("\x1b[");
                    try self.paste_buf.appendSlice(pp);
                    try self.paste_buf.append(b);
                    self.state = .paste;
                    self.seq_len = 0;
                }
            },
        }
    }

    /// Emit one codepoint to the key parser as UTF-8 bytes.
    fn feedCp(self: *Term, keys: *std.ArrayList(Key), cp: u32) !void {
        var tmp: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(@intCast(cp), &tmp) catch return;
        for (tmp[0..n]) |tb| try self.feedByte(keys, tb);
    }

    /// Pending 0x80..0xFF chars that didn't complete a UTF-8 sequence —
    /// emitted as individual codepoints (real Latin-1 input).
    fn flushMoji(self: *Term, keys: *std.ArrayList(Key)) !void {
        var j: usize = 0;
        while (j < self.moji_len) : (j += 1) try self.feedCp(keys, self.moji[j]);
        self.moji_len = 0;
    }

    fn emitX10(self: *Term, keys: *std.ArrayList(Key)) !void {
        const cb = self.x10[0];
        if ((cb & 64) != 0) {
            try keys.append(if ((cb & 1) == 0) .page_up else .page_down);
            return;
        }
        try keys.append(.{ .mouse = .{
            .col = @intCast(@max(1, @as(u32, self.x10[1])) - 32),
            .row = @intCast(@max(1, @as(u32, self.x10[2])) - 32),
            .btn = @intCast(cb & 3),
            .down = (cb & 3) != 3,
        } });
    }

    /// A complete CSI: self.seq holds the parameter bytes, `fin` is the final byte.
    fn emitCsiBody(self: *Term, keys: *std.ArrayList(Key), fin: u8, in_paste: bool) !void {
        const p = self.seq[0..self.seq_len];

        // bracketed paste delimiters
        if (fin == '~' and std.mem.eql(u8, p, "200")) {
            self.state = .paste;
            self.paste_buf.clearRetainingCapacity();
            return;
        }
        if (fin == '~' and std.mem.eql(u8, p, "201")) {
            if (self.paste_buf.items.len > 0) {
                try keys.append(.{ .paste = try self.paste_buf.toOwnedSlice() });
                self.paste_buf = std.ArrayList(u8).init(self.paste_buf.allocator);
            }
            self.state = .ground;
            return;
        }
        // from inside a paste, any other sequence is literal text
        if (in_paste) {
            try self.paste_buf.appendSlice("\x1b[");
            try self.paste_buf.appendSlice(p);
            try self.paste_buf.append(fin);
            self.state = .paste;
            return;
        }

        if (p.len > 0 and p[0] == '<') {
            // SGR mouse: ESC[<Cb;Cx;Cy(M|m)
            if (fin != 'M' and fin != 'm') return;
            var nums = [3]u32{ 0, 0, 0 };
            var ni: usize = 0;
            for (p[1..]) |d| {
                if (d >= '0' and d <= '9') {
                    if (ni < 3) nums[ni] = nums[ni] * 10 + (d - '0');
                } else if (d == ';') {
                    ni += 1;
                }
            }
            const btn_code = nums[0];
            if ((btn_code & 64) != 0) { // wheel
                const up = (btn_code & 1) == 0;
                keys.append(if (up) .page_up else .page_down) catch {};
                return;
            }
            keys.append(.{ .mouse = .{
                .col = @intCast(@max(1, nums[1]) - 1),
                .row = @intCast(@max(1, nums[2]) - 1),
                .btn = @intCast(btn_code & 3),
                .down = fin == 'M' and (btn_code & 32) == 0,
                .move = (btn_code & 32) != 0,
            } }) catch {};
            return;
        }

        // private / device reports (ESC[?.., ESC[>..): consume silently.
        // These are the replies a terminal sends for the query sequences the
        // app emits on startup, e.g. ESC[?1;2$y or ESC[>0;276;0c.
        if (p.len > 0 and (p[0] == '?' or p[0] == '>' or p[0] == '=')) return;

        if (fin == '~') {
            var num: u32 = 0;
            for (p) |d| {
                if (d >= '0' and d <= '9') num = num * 10 + (d - '0') else break;
            }
            switch (num) {
                1, 7 => keys.append(.home) catch {},
                3 => keys.append(.delete) catch {},
                4, 8 => keys.append(.end) catch {},
                5 => keys.append(.page_up) catch {},
                6 => keys.append(.page_down) catch {},
                else => {}, // F-keys, modifyOtherKeys, etc.
            }
            return;
        }
        try self.emitFinal(keys, fin);
    }

    /// Map a final byte to a key.
    fn emitFinal(self: *Term, keys: *std.ArrayList(Key), fin: u8) !void {
        _ = self;
        switch (fin) {
            'A' => try keys.append(.up),
            'B' => try keys.append(.down),
            'C' => try keys.append(.right),
            'D' => try keys.append(.left),
            'H' => try keys.append(.home),
            'F' => try keys.append(.end),
            // focus in/out (ESC[I / ESC[O) and everything else: not a key
            else => {},
        }
    }

    fn pollKeysPosix(self: *Term, alloc: std.mem.Allocator, keys: *std.ArrayList(Key), wait_ms: u32) !void {
        var pfd = [1]std.posix.pollfd{.{ .fd = self.in.handle, .events = std.posix.POLL.IN, .revents = 0 }};
        const n = std.posix.poll(&pfd, @intCast(wait_ms)) catch return;
        if (n <= 0) return;
        var buf: [4096]u8 = undefined;
        var got = std.posix.read(self.in.handle, &buf) catch return;
        var i: usize = 0;
        while (i < got) {
            const b = buf[i];
            i += 1;
            if (b == 0x1b and i < got) {
                // escape sequence
                if (buf[i] == '[' and i + 1 < got) {
                    i += 1;
                    const c = buf[i];
                    i += 1;
                    if (c == '<') {
                        // SGR mouse: ESC [ < Cb ; Cx ; Cy (M=press, m=release)
                        var nums = [3]u32{ 0, 0, 0 };
                        var ni: usize = 0;
                        var fin: u8 = 0;
                        while (i < got) {
                            const d = buf[i];
                            i += 1;
                            if (d >= '0' and d <= '9') {
                                if (ni < 3) nums[ni] = nums[ni] * 10 + (d - '0');
                            } else if (d == ';') {
                                ni += 1;
                            } else {
                                fin = d;
                                break;
                            }
                        }
                        if (fin == 'M' or fin == 'm') {
                            const btn_code = nums[0];
                            if ((btn_code & 64) == 0) { // not wheel
                                const btn: u8 = @intCast(btn_code & 3);
                                try keys.append(.{ .mouse = .{
                                    .col = @intCast(@max(1, nums[1]) - 1),
                                    .row = @intCast(@max(1, nums[2]) - 1),
                                    .btn = btn,
                                    .down = fin == 'M' and (btn_code & 32) == 0,
                                    .move = (btn_code & 32) != 0,
                                } });
                            } else {
                                try keys.append(if ((btn_code & 1) == 0) .page_up else .page_down);
                            }
                        }
                        continue;
                    }
                    switch (c) {
                        'A' => try keys.append(.up),
                        'B' => try keys.append(.down),
                        'C' => try keys.append(.right),
                        'D' => try keys.append(.left),
                        'H' => try keys.append(.home),
                        'F' => try keys.append(.end),
                        'Z' => continue,
                        '1'...'9' => {
                            var num: u32 = c - '0';
                            while (i < got and buf[i] >= '0' and buf[i] <= '9') : (i += 1)
                                num = num * 10 + (buf[i] - '0');
                            const fin = if (i < got) blk: {
                                const f = buf[i];
                                i += 1;
                                break :blk f;
                            } else '?';
                            if (fin == '~') switch (num) {
                                3 => try keys.append(.delete),
                                5 => try keys.append(.page_up),
                                6 => try keys.append(.page_down),
                                200 => {
                                    // bracketed paste: read until ESC[201~
                                    var payload = std.ArrayList(u8).init(alloc);
                                    var tail: usize = i;
                                    outer: while (true) {
                                        if (tail + 6 <= got and
                                            std.mem.eql(u8, buf[tail .. tail + 6], "\x1b[201~"))
                                        {
                                            i = tail + 6;
                                            break :outer;
                                        }
                                        if (tail >= got) {
                                            const n2 = std.posix.read(self.in.handle, &buf) catch 0;
                                            if (n2 <= 0) {
                                                i = got;
                                                break :outer;
                                            }
                                            // shift remaining into front
                                            const rem = buf[tail..got];
                                            std.mem.copyForwards(u8, buf[0..rem.len], rem);
                                            got = rem.len + n2;
                                            tail = 0;
                                            continue;
                                        }
                                        try payload.append(buf[tail]);
                                        tail += 1;
                                    }
                                    try keys.append(.{ .paste = payload.items });
                                },
                                else => {},
                            };
                        },
                        else => try keys.append(.escape),
                    }
                    continue;
                }
                try keys.append(.escape);
                continue;
            }
            switch (b) {
                '\r', '\n' => try keys.append(.enter),
                0x7f, 0x08 => try keys.append(.backspace),
                0x03 => try keys.append(.ctrl_c),
                0x04 => try keys.append(.ctrl_d),
                0x0c => try keys.append(.ctrl_l),
                '\t' => try keys.append(.tab),
                else => {
                    if (b < 0x80) {
                        try keys.append(.{ .char = b });
                    } else {
                        // utf8 decode
                        const len = std.unicode.utf8ByteSequenceLength(b) catch 1;
                        if (i - 1 + len <= got) {
                            const cp = std.unicode.utf8Decode(buf[i - 1 .. i - 1 + len]) catch b;
                            i += len - 1;
                            try keys.append(.{ .char = cp });
                        } else {
                            try keys.append(.{ .char = b });
                        }
                    }
                },
            }
        }
    }
};

const SavedTermios = if (win) void else std.posix.termios;

// ---------- Windows types not exposed by std ----------

const BOOL = std.os.windows.BOOL;
const HANDLE = std.os.windows.HANDLE;

const KEY_EVENT_RECORD = extern struct {
    bKeyDown: BOOL,
    wRepeatCount: u16,
    wVirtualKeyCode: u16,
    wVirtualScanCode: u16,
    UnicodeChar: u16,
    dwControlKeyState: u32,
};

const MOUSE_EVENT_RECORD = extern struct {
    dwMousePosition: u32,
    dwButtonState: u32,
    dwControlKeyState: u32,
    dwEventFlags: u32,
};

const INPUT_RECORD = extern struct {
    EventType: u16,
    _pad: u16 = 0,
    Event: extern union {
        KeyEvent: KEY_EVENT_RECORD,
        MouseEvent: MOUSE_EVENT_RECORD,
        WindowBufferSizeEvent: u32,
        MenuEvent: u32,
        FocusEvent: u32,
    },
};

const SMALL_RECT = extern struct {
    Left: i16,
    Top: i16,
    Right: i16,
    Bottom: i16,
};

const CONSOLE_SCREEN_BUFFER_INFO = extern struct {
    dwSize: i32,
    dwCursorPosition: i32,
    wAttributes: u16,
    srWindow: SMALL_RECT,
    dwMaximumWindowSize: i32,
};

extern "kernel32" fn ReadConsoleInputW(hConsoleInput: HANDLE, lpBuffer: [*]INPUT_RECORD, nLength: u32, lpNumberOfEventsRead: *u32) callconv(std.os.windows.WINAPI) BOOL;
extern "kernel32" fn GetConsoleScreenBufferInfo(hConsoleOutput: HANDLE, lpConsoleScreenBufferInfo: *CONSOLE_SCREEN_BUFFER_INFO) callconv(std.os.windows.WINAPI) BOOL;

// ---------- drawing helpers ----------

pub const Style = struct {
    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";
    pub const italic = "\x1b[3m";
    pub const underline = "\x1b[4m";
    pub const inverse = "\x1b[7m";
    pub const gray = "\x1b[90m";
    pub const red = "\x1b[31m";
    pub const green = "\x1b[32m";
    pub const yellow = "\x1b[33m";
    pub const blue = "\x1b[34m";
    pub const magenta = "\x1b[35m";
    pub const cyan = "\x1b[36m";
    pub const white = "\x1b[37m";
    pub const bright_blue = "\x1b[94m";
    pub const bright_green = "\x1b[92m";
    pub const bright_cyan = "\x1b[96m";
    /// xera accent — sky blue (24-bit)
    pub const accent = "\x1b[38;2;96;170;255m";
    /// deep shadow tone for 3D/depth effects (24-bit)
    pub const shadow = "\x1b[38;2;40;60;100m";
};

/// Display width of a string, ignoring ANSI CSI sequences.
pub fn displayWidth(s: []const u8) usize {
    var w: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c == 0x1b and i + 1 < s.len and s[i + 1] == '[') {
            i += 2;
            while (i < s.len and !((s[i] >= 'a' and s[i] <= 'z') or (s[i] >= 'A' and s[i] <= 'Z'))) i += 1;
            i += 1; // skip final byte
            continue;
        }
        if (c < 0x80) {
            w += 1;
            i += 1;
            continue;
        }
        const len = std.unicode.utf8ByteSequenceLength(c) catch 1;
        const cp = if (i + len <= s.len) std.unicode.utf8Decode(s[i .. i + len]) catch c else c;
        w += cpWidth(cp);
        i += len;
    }
    return w;
}

pub fn cpWidth(cp: u21) usize {
    if (cp == 0) return 0;
    if (cp < 32 or (cp >= 0x7f and cp < 0xa0)) return 0;
    if (cp >= 0x300 and cp <= 0x36f) return 0; // combining
    if (cp >= 0x1100 and (cp <= 0x115f or cp == 0x2329 or cp == 0x232a or
        (cp >= 0x2e80 and cp <= 0xa4cf and cp != 0x303f) or
        (cp >= 0xac00 and cp <= 0xd7a3) or
        (cp >= 0xf900 and cp <= 0xfaff) or
        (cp >= 0xfe30 and cp <= 0xfe6f) or
        (cp >= 0xff00 and cp <= 0xff60) or
        (cp >= 0xffe0 and cp <= 0xffe6) or
        (cp >= 0x20000 and cp <= 0x2fffd) or
        (cp >= 0x30000 and cp <= 0x3fffd) or cp >= 0x1f300)) return 2;
    return 1;
}

/// Wrap a styled line (may contain ANSI codes) to `width` display columns.
/// Never splits inside an escape sequence; hard-breaks long words.
pub fn wrapLine(alloc: std.mem.Allocator, line: []const u8, width: usize) !std.ArrayList([]const u8) {
    var out = std.ArrayList([]const u8).init(alloc);
    if (width == 0) {
        try out.append(line);
        return out;
    }
    var cur = std.ArrayList(u8).init(alloc);
    var cur_w: usize = 0;
    var i: usize = 0;
    var last_space: ?usize = null; // byte index into cur
    while (i < line.len) {
        const c = line[i];
        if (c == 0x1b and i + 1 < line.len and line[i + 1] == '[') {
            const start = i;
            i += 2;
            while (i < line.len and !((line[i] >= 'a' and line[i] <= 'z') or (line[i] >= 'A' and line[i] <= 'Z'))) i += 1;
            i += 1;
            try cur.appendSlice(line[start..i]);
            continue;
        }
        const len = if (c < 0x80) 1 else (std.unicode.utf8ByteSequenceLength(c) catch 1);
        const cp = if (c < 0x80) c else if (i + len <= line.len) (std.unicode.utf8Decode(line[i .. i + len]) catch c) else c;
        const cw = cpWidth(cp);
        if (c == ' ') last_space = cur.items.len;
        if (cur_w + cw > width and cur.items.len > 0) {
            if (last_space) |sp| {
                if (sp < cur.items.len) {
                    // break at the space
                    try out.append(try alloc.dupe(u8, cur.items[0..sp]));
                    const rest = cur.items[sp + 1 ..];
                    var ncur = std.ArrayList(u8).init(alloc);
                    try ncur.appendSlice(rest);
                    cur = ncur;
                    cur_w = displayWidth(rest);
                    last_space = std.mem.lastIndexOfScalar(u8, cur.items, ' ');
                } else {
                    // sp == cur.items.len: the space that triggered this wrap
                    // hasn't been appended yet — it is the break point.
                    try out.append(try alloc.dupe(u8, cur.items));
                    cur.clearRetainingCapacity();
                    cur_w = 0;
                    last_space = null;
                    i += len;
                    continue;
                }
            } else {
                try out.append(try alloc.dupe(u8, cur.items));
                cur.clearRetainingCapacity();
                cur_w = 0;
            }
        }
        try cur.appendSlice(line[i .. i + @min(len, line.len - i)]);
        cur_w += cw;
        i += len;
    }
    try out.append(cur.items);
    return out;
}

test "wrapLine handles a space at the width boundary" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const wrapped = try wrapLine(arena.allocator(), "hello world", 5);
    try std.testing.expectEqual(@as(usize, 2), wrapped.items.len);
    try std.testing.expectEqualStrings("hello", wrapped.items[0]);
    try std.testing.expectEqualStrings("world", wrapped.items[1]);
}

/// Wrap all lines of a block (split on \n) to width.
pub fn wrapText(alloc: std.mem.Allocator, text: []const u8, width: usize) !std.ArrayList([]const u8) {
    var out = std.ArrayList([]const u8).init(alloc);
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |ln| {
        if (ln.len == 0) {
            try out.append("");
            continue;
        }
        const wrapped = try wrapLine(alloc, ln, width);
        for (wrapped.items) |w| try out.append(w);
    }
    return out;
}
