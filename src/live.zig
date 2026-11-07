const std = @import("std");
const util = @import("util.zig");

/// Live observability log. One JSONL event stream per session under
/// ~/.xera-rsi/live/<session-id>.jsonl, plus a global
/// ~/.xera-rsi/live/improvements.jsonl for Harness² runs.
/// Everything is best-effort: logging failures are swallowed so the
/// agent is never broken by the observer.
pub const Live = struct {
    alloc: std.mem.Allocator,
    path: []const u8,
    id: []const u8,
    mutex: std.Thread.Mutex = .{},

    /// Create a live log for session `id` (same id as session.Log).
    pub fn create(alloc: std.mem.Allocator, id: []const u8) !*Live {
        const dir = try liveDir(alloc);
        try std.fs.cwd().makePath(dir);
        const path = try std.fs.path.join(alloc, &.{ dir, try std.fmt.allocPrint(alloc, "{s}.jsonl", .{id}) });
        const self = try alloc.create(Live);
        self.* = .{ .alloc = alloc, .path = path, .id = id };
        return self;
    }

    /// Core emit: fields is raw JSON fragment "key":value pairs (already escaped).
    fn emit(self: *Live, kind: []const u8, fields: []const u8, depth: u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        appendLine(self.alloc, self.path, kind, self.id, fields, depth);
    }

    fn fieldStr(alloc: std.mem.Allocator, key: []const u8, val: []const u8) ![]u8 {
        return std.fmt.allocPrint(alloc, "\"{s}\":\"{s}\"", .{ key, try util.jescape(alloc, val) });
    }

    pub fn sessionStart(self: *Live, cwd: []const u8, model: []const u8, harness_dir: []const u8, player: []const u8) void {
        const a = self.alloc;
        const f = std.fmt.allocPrint(a, "{s},{s},{s},{s}", .{
            fieldStr(a, "cwd", cwd) catch return,
            fieldStr(a, "model", model) catch return,
            fieldStr(a, "harness", harness_dir) catch return,
            fieldStr(a, "name", player) catch return,
        }) catch return;
        self.emit("session_start", f, 0);
    }

    pub fn user(self: *Live, text: []const u8, sent: []const u8, author: []const u8, depth: u8) void {
        const a = self.alloc;
        var extra: []const u8 = "";
        if (!std.mem.eql(u8, text, sent)) {
            extra = std.fmt.allocPrint(a, ",{s}", .{fieldStr(a, "sent", sent) catch return}) catch "";
        }
        const f = std.fmt.allocPrint(a, "{s},{s}{s}", .{
            fieldStr(a, "text", text) catch return,
            fieldStr(a, "author", author) catch return,
            extra,
        }) catch return;
        self.emit("user", f, depth);
    }

    pub fn reasoning(self: *Live, text: []const u8, depth: u8) void {
        const f = fieldStr(self.alloc, "text", text) catch return;
        self.emit("reasoning", f, depth);
    }

    pub fn assistant(self: *Live, text: []const u8, tool_names: []const u8, depth: u8) void {
        const a = self.alloc;
        // tool_names is a comma-separated list (may be "")
        var tc: []const u8 = "";
        if (tool_names.len > 0) {
            var arr = std.ArrayList(u8).init(a);
            arr.append('[') catch return;
            var it = std.mem.splitScalar(u8, tool_names, ',');
            var first = true;
            while (it.next()) |n| {
                if (n.len == 0) continue;
                if (!first) arr.append(',') catch return;
                first = false;
                arr.writer().print("\"{s}\"", .{n}) catch return;
            }
            arr.append(']') catch return;
            tc = std.fmt.allocPrint(a, ",\"tools\":{s}", .{arr.items}) catch "";
        }
        const f = std.fmt.allocPrint(a, "{s}{s}", .{
            fieldStr(a, "text", text) catch return,
            tc,
        }) catch return;
        self.emit("assistant", f, depth);
    }

    pub fn toolCall(self: *Live, name: []const u8, summary: []const u8, args: []const u8, call_id: []const u8, depth: u8) void {
        const a = self.alloc;
        // args is embedded as a JSON string (it's raw model JSON; escaping keeps it intact)
        const f = std.fmt.allocPrint(a, "{s},{s},{s},{s}", .{
            fieldStr(a, "name", name) catch return,
            fieldStr(a, "summary", summary) catch return,
            fieldStr(a, "args", args) catch return,
            fieldStr(a, "call_id", call_id) catch return,
        }) catch return;
        self.emit("tool_call", f, depth);
    }

    pub fn toolResult(self: *Live, name: []const u8, ok: bool, preview: []const u8, call_id: []const u8, depth: u8) void {
        const a = self.alloc;
        const f = std.fmt.allocPrint(a, "{s},\"ok\":{s},{s},{s}", .{
            fieldStr(a, "name", name) catch return,
            if (ok) "true" else "false",
            fieldStr(a, "preview", preview) catch return,
            fieldStr(a, "call_id", call_id) catch return,
        }) catch return;
        self.emit("tool_result", f, depth);
    }

    pub fn skillUsed(self: *Live, name: []const u8, depth: u8) void {
        const f = fieldStr(self.alloc, "name", name) catch return;
        self.emit("skill", f, depth);
    }

    pub fn subagent(self: *Live, agent_name: []const u8, title: []const u8, depth: u8) void {
        const a = self.alloc;
        const f = std.fmt.allocPrint(a, "{s},{s}", .{
            fieldStr(a, "agent", agent_name) catch return,
            fieldStr(a, "title", title) catch return,
        }) catch return;
        self.emit("subagent", f, depth);
    }

    /// Generic typed event: emit(type, {what,detail}) — for subsystems
    /// that don't warrant their own event kind.
    pub fn event(self: *Live, ev_type: []const u8, what: []const u8, detail: []const u8, depth: u8) void {
        const a = self.alloc;
        const f = std.fmt.allocPrint(a, "{s},{s}", .{
            fieldStr(a, "what", what) catch return,
            fieldStr(a, "detail", detail) catch return,
        }) catch return;
        self.emit(ev_type, f, depth);
    }

    /// Memory-layer event: remember / recall / forget / recall_auto / capture.
    pub fn memEvent(self: *Live, what: []const u8, detail: []const u8, depth: u8) void {
        const a = self.alloc;
        const f = std.fmt.allocPrint(a, "{s},{s}", .{
            fieldStr(a, "what", what) catch return,
            fieldStr(a, "detail", detail) catch return,
        }) catch return;
        self.emit("memory", f, depth);
    }

    /// Coordination-fabric event: claim / claim_conflict / blocked / send.
    pub fn coordEvent(self: *Live, what: []const u8, detail: []const u8, depth: u8) void {
        const a = self.alloc;
        const f = std.fmt.allocPrint(a, "{s},{s}", .{
            fieldStr(a, "what", what) catch return,
            fieldStr(a, "detail", detail) catch return,
        }) catch return;
        self.emit("coord", f, depth);
    }

    pub fn usage(self: *Live, prompt: u64, completion: u64, depth: u8) void {
        const f = std.fmt.allocPrint(self.alloc, "\"prompt\":{d},\"completion\":{d}", .{ prompt, completion }) catch return;
        self.emit("usage", f, depth);
    }

    pub fn turnEnd(self: *Live, stop_reason: []const u8, iters: usize, prompt: u64, completion: u64, depth: u8) void {
        const a = self.alloc;
        const f = std.fmt.allocPrint(a, "{s},\"iters\":{d},\"prompt\":{d},\"completion\":{d}", .{
            fieldStr(a, "stop", stop_reason) catch return,
            iters, prompt, completion,
        }) catch return;
        self.emit("turn_end", f, depth);
    }
};

pub fn liveDir(alloc: std.mem.Allocator) ![]u8 {
    const home = try util.homeDir(alloc);
    return std.fs.path.join(alloc, &.{ home, ".xera-rsi", "live" });
}

fn appendLine(alloc: std.mem.Allocator, path: []const u8, kind: []const u8, id: []const u8, fields: []const u8, depth: u8) void {
    const line = std.fmt.allocPrint(alloc,
        "{{\"type\":\"{s}\",\"session\":\"{s}\",\"ts\":{d},\"depth\":{d},{s}}}\n",
        .{ kind, id, std.time.milliTimestamp(), depth, fields }) catch return;
    defer alloc.free(line);
    const f = std.fs.cwd().createFile(path, .{ .truncate = false }) catch return;
    defer f.close();
    f.seekFromEnd(0) catch return;
    f.writeAll(line) catch {};
}

var imp_mutex: std.Thread.Mutex = .{};

/// Append a Harness² improvement event to the global improvements.jsonl.
/// fields is a raw JSON fragment like the emit helpers produce.
pub fn improvement(alloc: std.mem.Allocator, fields: []const u8) void {
    imp_mutex.lock();
    defer imp_mutex.unlock();
    const dir = liveDir(alloc) catch return;
    std.fs.cwd().makePath(dir) catch return;
    const path = std.fs.path.join(alloc, &.{ dir, "improvements.jsonl" }) catch return;
    defer alloc.free(path);
    const line = std.fmt.allocPrint(alloc,
        "{{\"type\":\"improve\",\"ts\":{d},{s}}}\n",
        .{ std.time.milliTimestamp(), fields }) catch return;
    defer alloc.free(line);
    const f = std.fs.cwd().createFile(path, .{ .truncate = false }) catch return;
    defer f.close();
    f.seekFromEnd(0) catch return;
    f.writeAll(line) catch {};
}
