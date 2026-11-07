const std = @import("std");
const builtin = @import("builtin");

pub const ansi = struct {
    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const dim = "\x1b[2m";
    pub const italic = "\x1b[3m";
    pub const cyan = "\x1b[36m";
    pub const green = "\x1b[32m";
    pub const yellow = "\x1b[33m";
    pub const red = "\x1b[31m";
    pub const magenta = "\x1b[35m";
    pub const blue = "\x1b[34m";
    pub const gray = "\x1b[90m";
};

var color_enabled: bool = true;

pub fn enableColor(on: bool) void {
    color_enabled = on;
}

pub fn colored(on: bool) bool {
    return color_enabled and on;
}

/// Set Windows console to UTF-8 and enable VT processing so ANSI works.
pub fn setupConsole() void {
    if (builtin.os.tag == .windows) {
        _ = std.os.windows.kernel32.SetConsoleOutputCP(65001);
        setConsoleCP(65001);
    }
}

extern "kernel32" fn SetConsoleCP(wCodePageID: std.os.windows.UINT) callconv(std.os.windows.WINAPI) std.os.windows.BOOL;

fn setConsoleCP(cp: std.os.windows.UINT) void {
    _ = SetConsoleCP(cp);
}

pub fn homeDir(alloc: std.mem.Allocator) ![]u8 {
    if (builtin.os.tag == .windows) {
        if (std.process.getEnvVarOwned(alloc, "USERPROFILE")) |h| return h else |_| {}
    }
    if (std.process.getEnvVarOwned(alloc, "HOME")) |h| return h else |_| {}
    return error.NoHome;
}

pub fn getenv(alloc: std.mem.Allocator, name: []const u8) ?[]u8 {
    return std.process.getEnvVarOwned(alloc, name) catch null;
}

pub fn getenvBool(alloc: std.mem.Allocator, name: []const u8) ?bool {
    const v = getenv(alloc, name) orelse return null;
    if (v.len == 0) return true;
    const truthy = [_][]const u8{ "1", "true", "yes", "on", "y" };
    for (truthy) |t| if (std.ascii.eqlIgnoreCase(v, t)) return true;
    return false;
}

/// Match a glob pattern against a path (slash-separated). Supports `*`, `**`, `?`.
/// `**` matches across separators; `*` and `?` do not.
pub fn globMatch(pattern: []const u8, text: []const u8) bool {
    return globMatchAt(pattern, 0, text, 0);
}

fn globMatchAt(p: []const u8, pi: usize, t: []const u8, ti: usize) bool {
    var i = pi;
    var j = ti;
    while (i < p.len) {
        const c = p[i];
        if (c == '*') {
            var star_len: usize = 1;
            while (i + star_len < p.len and p[i + star_len] == '*') star_len += 1;
            const double = star_len >= 2;
            i += star_len;
            if (i >= p.len) {
                // trailing ** matches everything; trailing * matches non-slash tail
                if (double) return true;
                return std.mem.indexOfScalar(u8, t[j..], '/') == null and
                    std.mem.indexOfScalar(u8, t[j..], '\\') == null;
            }
            // try to match rest at every position
            var k = j;
            while (true) {
                if (globMatchAt(p, i, t, k)) return true;
                if (k >= t.len) return false;
                const ch = t[k];
                if (!double and (ch == '/' or ch == '\\')) return false;
                k += 1;
            }
        } else if (c == '?') {
            if (j >= t.len or t[j] == '/' or t[j] == '\\') return false;
            i += 1;
            j += 1;
        } else {
            if (j >= t.len or !charEq(c, t[j])) return false;
            i += 1;
            j += 1;
        }
    }
    return j == t.len;
}

fn charEq(a: u8, b: u8) bool {
    if (a == '/' or a == '\\') return b == '/' or b == '\\';
    return a == b;
}

pub fn normalizePath(alloc: std.mem.Allocator, path: []const u8) []u8 {
    const out = alloc.dupe(u8, path) catch return @constCast(path[0..0]);
    for (out) |*c| {
        if (c.* == '\\') c.* = '/';
    }
    return out;
}

/// Resolve `path` against `cwd`, normalize separators. Absolute paths pass through.
pub fn resolvePath(alloc: std.mem.Allocator, cwd: []const u8, path: []const u8) ![]u8 {
    const p = std.mem.trim(u8, path, " \t\"'");
    if (p.len == 0) return error.EmptyPath;
    if (builtin.os.tag == .windows and p.len >= 2 and p[0] == '/' and p[1] == '/') {
        // UNC — leave as is
    }
    const resolved = if (std.fs.path.isAbsolute(p))
        try alloc.dupe(u8, p)
    else
        try std.fs.path.resolve(alloc, &.{ cwd, p });
    defer alloc.free(resolved);
    return normalizePathOwned(alloc, resolved);
}

fn normalizePathOwned(alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    const out = try alloc.dupe(u8, path);
    for (out) |*c| {
        if (c.* == '\\') c.* = '/';
    }
    return out;
}

/// True if `child` is inside `root` (case-insensitive on Windows).
pub fn pathWithin(root: []const u8, child: []const u8) bool {
    if (child.len < root.len) return false;
    var i: usize = 0;
    while (i < root.len) : (i += 1) {
        const a = normSep(root[i]);
        const b = normSep(child[i]);
        if (builtin.os.tag == .windows) {
            if (std.ascii.toLower(a) != std.ascii.toLower(b)) return false;
        } else if (a != b) return false;
    }
    if (child.len == root.len) return true;
    return child[root.len] == '/' or child[root.len] == '\\' or
        root[root.len - 1] == '/' or root[root.len - 1] == '\\';
}

fn normSep(c: u8) u8 {
    return if (c == '\\') '/' else c;
}

/// Is `file` a terminal (console)?
pub fn isTty(f: std.fs.File) bool {
    if (builtin.os.tag == .windows) {
        var m: u32 = 0;
        return std.os.windows.kernel32.GetConsoleMode(f.handle, &m) != 0;
    }
    return std.posix.isatty(f.handle);
}

/// Keep the last `max` display columns of a path with a "…" prefix.
pub fn shortenPath(alloc: std.mem.Allocator, path: []const u8, max: usize) ![]const u8 {
    if (path.len <= max) return path;
    const tail = path[path.len - (max - 1) ..];
    return std.fmt.allocPrint(alloc, "…{s}", .{tail});
}

pub fn truncateMiddle(alloc: std.mem.Allocator, s: []const u8, max: usize) ![]const u8 {
    if (s.len <= max) return s;
    const head = max * 2 / 3;
    const tail = max - head;
    return std.fmt.allocPrint(alloc, "{s}\n…[{d} bytes omitted]…\n{s}", .{
        s[0..head], s.len - head - tail, s[s.len - tail ..],
    });
}

pub fn firstLine(s: []const u8) []const u8 {
    const idx = std.mem.indexOfAny(u8, s, "\r\n") orelse return s;
    return s[0..idx];
}

// ---------- JSON helpers ----------

pub fn jget(v: std.json.Value, key: []const u8) ?std.json.Value {
    if (v != .object) return null;
    return v.object.get(key);
}

pub fn jstr(v: std.json.Value, key: []const u8) ?[]const u8 {
    const g = jget(v, key) orelse return null;
    if (g != .string) return null;
    return g.string;
}

pub fn jint(v: std.json.Value, key: []const u8) ?i64 {
    const g = jget(v, key) orelse return null;
    return switch (g) {
        .integer => |n| n,
        .float => |f| @intFromFloat(f),
        else => null,
    };
}

pub fn jbool(v: std.json.Value, key: []const u8) ?bool {
    const g = jget(v, key) orelse return null;
    if (g != .bool) return null;
    return g.bool;
}

pub fn jobjPut(obj: *std.json.ObjectMap, key: []const u8, v: std.json.Value) !void {
    try obj.put(key, v);
}

pub fn jstrValue(s: []const u8) std.json.Value {
    return .{ .string = s };
}

pub fn makeObj(alloc: std.mem.Allocator) std.json.ObjectMap {
    return std.json.ObjectMap.init(alloc);
}

pub fn makeArr(alloc: std.mem.Allocator) std.json.Array {
    return std.json.Array.init(alloc);
}

/// Stringify a std.json.Value into an allocated string.
pub fn jstringify(alloc: std.mem.Allocator, v: std.json.Value) ![]u8 {
    var buf = std.ArrayList(u8).init(alloc);
    try std.json.stringify(v, .{}, buf.writer());
    return buf.toOwnedSlice();
}

/// Parse a string slice as JSON value (leaks into `alloc` — use an arena).
pub fn jparse(alloc: std.mem.Allocator, s: []const u8) !std.json.Value {
    return std.json.parseFromSliceLeaky(std.json.Value, alloc, s, .{});
}

/// Escape a string for embedding inside a JSON string literal (no quotes added).
pub fn jescape(alloc: std.mem.Allocator, s: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(alloc);
    for (s) |c| {
        switch (c) {
            '"', '\\' => {
                try out.append('\\');
                try out.append(c);
            },
            '\n' => try out.appendSlice("\\n"),
            '\r' => try out.appendSlice("\\r"),
            '\t' => try out.appendSlice("\\t"),
            else => {
                if (c < 0x20) {
                    try out.writer().print("\\u{x:0>4}", .{c});
                } else {
                    try out.append(c);
                }
            },
        }
    }
    return out.toOwnedSlice();
}
