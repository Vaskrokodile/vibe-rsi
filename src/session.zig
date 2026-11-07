const std = @import("std");
const util = @import("util.zig");
const config = @import("config.zig");

/// Sessions are JSONL files under ~/.xera-rsi/sessions/<id>.jsonl.
/// First line is a {"_meta":{...}} record; subsequent lines are message values.
pub const Log = struct {
    alloc: std.mem.Allocator,
    path: []const u8,
    id: []const u8,

    pub fn append(self: *Log, msg: std.json.Value) !void {
        const f = try std.fs.cwd().createFile(self.path, .{ .truncate = false });
        defer f.close();
        try f.seekFromEnd(0);
        const line = try util.jstringify(self.alloc, msg);
        try f.writeAll(line);
        try f.writeAll("\n");
    }
};

pub fn sessionsDir(alloc: std.mem.Allocator, cfg: *const config.Config) ![]const u8 {
    if (cfg.sessions_dir) |d| return d;
    const home = try util.homeDir(alloc);
    return std.fs.path.join(alloc, &.{ home, ".xera-rsi", "sessions" });
}

/// Create a new session log; writes the _meta line.
pub fn create(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    cwd: []const u8,
) !Log {
    const dir = try sessionsDir(alloc, cfg);
    try std.fs.cwd().makePath(dir);
    const ts = std.time.milliTimestamp();
    const id = try std.fmt.allocPrint(alloc, "s-{d}", .{ts});
    const path = try std.fs.path.join(alloc, &.{ dir, id });
    const path_jsonl = try std.fmt.allocPrint(alloc, "{s}.jsonl", .{path});
    var log = Log{ .alloc = alloc, .path = path_jsonl, .id = id };
    var meta = util.makeObj(alloc);
    var inner = util.makeObj(alloc);
    try inner.put("id", .{ .string = id });
    try inner.put("cwd", .{ .string = cwd });
    try inner.put("model", .{ .string = cfg.model });
    try inner.put("created_ms", .{ .integer = ts });
    try meta.put("_meta", .{ .object = inner });
    try log.append(.{ .object = meta });
    return log;
}

pub const Info = struct {
    id: []const u8,
    cwd: []const u8,
    model: []const u8,
    created_ms: i64,
    turns: usize,
};

pub fn list(alloc: std.mem.Allocator, cfg: *const config.Config) ![]Info {
    var out = std.ArrayList(Info).init(alloc);
    const dir = try sessionsDir(alloc, cfg);
    var d = std.fs.cwd().openDir(dir, .{ .iterate = true }) catch return out.toOwnedSlice();
    defer d.close();
    var it = d.iterate();
    while (try it.next()) |e| {
        if (e.kind != .file or !std.mem.endsWith(u8, e.name, ".jsonl")) continue;
        const p = try std.fs.path.join(alloc, &.{ dir, e.name });
        const raw = std.fs.cwd().readFileAlloc(alloc, p, 8 << 20) catch continue;
        var info = Info{
            .id = try alloc.dupe(u8, e.name[0 .. e.name.len - 6]),
            .cwd = "",
            .model = "",
            .created_ms = 0,
            .turns = 0,
        };
        var lit = std.mem.splitScalar(u8, raw, '\n');
        var first = true;
        while (lit.next()) |line| {
            const t = std.mem.trim(u8, line, " \r\t");
            if (t.len == 0) continue;
            if (first) {
                first = false;
                const v = util.jparse(alloc, t) catch continue;
                if (util.jget(v, "_meta")) |m| {
                    if (util.jstr(m, "cwd")) |c| info.cwd = c;
                    if (util.jstr(m, "model")) |m2| info.model = m2;
                    if (util.jint(m, "created_ms")) |ts| info.created_ms = ts;
                    if (util.jstr(m, "id")) |sid| info.id = sid;
                    continue;
                }
            }
            const v = util.jparse(alloc, t) catch continue;
            const role = util.jstr(v, "role") orelse continue;
            if (std.mem.eql(u8, role, "user")) info.turns += 1;
        }
        try out.append(info);
    }
    // newest first
    std.mem.sort(Info, out.items, {}, struct {
        fn lt(_: void, a: Info, b: Info) bool {
            return a.created_ms > b.created_ms;
        }
    }.lt);
    return out.toOwnedSlice();
}

/// Load session messages (excluding _meta) into a list for resume.
pub fn loadMessages(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    id: []const u8,
) !std.ArrayList(std.json.Value) {
    var out = std.ArrayList(std.json.Value).init(alloc);
    const dir = try sessionsDir(alloc, cfg);
    const fname = try std.fmt.allocPrint(alloc, "{s}.jsonl", .{id});
    const p = try std.fs.path.join(alloc, &.{ dir, fname });
    const raw = std.fs.cwd().readFileAlloc(alloc, p, 32 << 20) catch |e| return e;
    var lit = std.mem.splitScalar(u8, raw, '\n');
    while (lit.next()) |line| {
        const t = std.mem.trim(u8, line, " \r\t");
        if (t.len == 0) continue;
        const v = util.jparse(alloc, t) catch continue;
        if (util.jget(v, "_meta") != null) continue;
        try out.append(v);
    }
    return out;
}
