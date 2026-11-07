const std = @import("std");
const util = @import("util.zig");

/// Lifecycle hooks (harness2 c8): `hooks.json` (or `.codex/hooks.json`) in the
/// harness dir, Claude-Code schema:
///
///   {"hooks": {"PreToolUse": [{"matcher": "run_shell",
///                "hooks": [{"type": "command", "command": "my_check.cmd"}]}]}}
///
/// Events: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse,
/// SubagentStart, SubagentStop, PreCompact, PostCompact, Stop.
/// The command gets a JSON event payload on stdin. If its stdout parses as
/// {"decision":"block","reason":"..."} the reason is returned to the model —
/// PreToolUse blocks the call entirely; PostToolUse feeds back extra context.
pub const event_names = [_][]const u8{
    "SessionStart",   "UserPromptSubmit", "PreToolUse", "PostToolUse",
    "SubagentStart",  "SubagentStop",     "PreCompact", "PostCompact", "Stop",
};

const HookCmd = struct {
    matcher: []const u8, // substring matched against tool name ("" = all)
    command: []const u8,
};

pub const Hooks = struct {
    alloc: std.mem.Allocator,
    entries: std.StringArrayHashMap(std.ArrayList(HookCmd)),

    /// Load hooks.json from the harness dir. null = no hooks configured.
    pub fn load(alloc: std.mem.Allocator, path_opt: ?[]const u8) ?Hooks {
        const path = path_opt orelse return null;
        const body = std.fs.cwd().readFileAlloc(alloc, path, 1 << 20) catch return null;
        const root = util.jparse(alloc, body) catch return null;
        const hooks_obj = util.jget(root, "hooks") orelse return null;
        if (hooks_obj != .object) return null;

        var h = Hooks{
            .alloc = alloc,
            .entries = std.StringArrayHashMap(std.ArrayList(HookCmd)).init(alloc),
        };
        var it = hooks_obj.object.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* != .array) continue;
            var list = std.ArrayList(HookCmd).init(alloc);
            for (kv.value_ptr.array.items) |entry| {
                const matcher = util.jstr(entry, "matcher") orelse "";
                const inner = util.jget(entry, "hooks") orelse continue;
                if (inner != .array) continue;
                for (inner.array.items) |hook| {
                    const t = util.jstr(hook, "type") orelse continue;
                    if (!std.mem.eql(u8, t, "command")) continue;
                    const cmd = util.jstr(hook, "command") orelse continue;
                    if (cmd.len == 0) continue;
                    list.append(.{ .matcher = matcher, .command = cmd }) catch {};
                }
            }
            if (list.items.len > 0) h.entries.put(kv.key_ptr.*, list) catch {};
        }
        return if (h.entries.count() > 0) h else null;
    }

    /// Fire an event. `payload` is a JSON object handed to the hook on stdin.
    /// Returns non-empty string = block reason / extra context from the hook.
    pub fn fire(
        self: *const Hooks,
        scratch: std.mem.Allocator,
        event: []const u8,
        payload: []const u8,
        tool_name: []const u8,
    ) []const u8 {
        const list = self.entries.get(event) orelse return "";
        var out = std.ArrayList(u8).init(scratch);
        for (list.items) |hc| {
            if (hc.matcher.len > 0 and tool_name.len > 0 and
                std.mem.indexOf(u8, tool_name, hc.matcher) == null) continue;
            if (hc.matcher.len > 0 and tool_name.len == 0) continue;
            const res = runCapture(scratch, hc.command, payload) catch continue;
            // a hook's stdout: parse for {"decision":"block","reason":"..."}
            if (util.jparse(scratch, res)) |doc| {
                if (util.jstr(doc, "decision")) |d| {
                    if (std.mem.eql(u8, d, "block")) {
                        const r = util.jstr(doc, "reason") orelse "blocked by hook";
                        if (out.items.len > 0) out.appendSlice("\n") catch {};
                        out.appendSlice(r) catch {};
                    } else if (std.mem.eql(u8, d, "context")) {
                        const r = util.jstr(doc, "reason") orelse "";
                        if (r.len > 0) {
                            if (out.items.len > 0) out.appendSlice("\n") catch {};
                            out.appendSlice(r) catch {};
                        }
                    }
                }
            } else |_| {
                // non-JSON stdout = informational context
                const t = std.mem.trim(u8, res, " \r\n\t");
                if (t.len > 0 and t.len < 4000) {
                    if (out.items.len > 0) out.appendSlice("\n") catch {};
                    out.appendSlice(t) catch {};
                }
            }
        }
        return out.items;
    }
};

const PipeRead = struct {
    file: std.fs.File,
    list: *std.ArrayList(u8),
    max: usize,

    fn run(p: PipeRead) void {
        p.file.reader().readAllArrayList(p.list, p.max) catch {};
    }
};

const KillWatch = struct {
    child: *std.process.Child,
    done: *std.atomic.Value(bool),
    timeout_ms: u64,

    fn run(self: KillWatch) void {
        const deadline = std.time.milliTimestamp() + @as(i64, @intCast(self.timeout_ms));
        while (std.time.milliTimestamp() < deadline) {
            if (self.done.load(.acquire)) return;
            std.time.sleep(20 * std.time.ns_per_ms);
        }
        if (!self.done.load(.acquire)) _ = self.child.kill() catch return;
    }
};

/// Spawn a hook command: `cmd /c` on Windows, `/bin/sh -c` on POSIX.
/// JSON payload goes to stdin; stdout captured (15s timeout).
fn runCapture(scratch: std.mem.Allocator, command: []const u8, stdin_data: []const u8) ![]const u8 {
    const argv: []const []const u8 = if (@import("builtin").os.tag == .windows)
        &.{ "cmd.exe", "/c", command }
    else
        &.{ "/bin/sh", "-c", command };

    var child = std.process.Child.init(argv, scratch);
    child.stdin_behavior = .Pipe;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Ignore;
    try child.spawn();

    if (child.stdin) |w| {
        w.writeAll(stdin_data) catch {};
        w.close();
        child.stdin = null;
    }

    var out = std.ArrayList(u8).init(scratch);
    var done = std.atomic.Value(bool).init(false);
    const watch = try std.Thread.spawn(.{}, KillWatch.run, .{KillWatch{
        .child = &child,
        .done = &done,
        .timeout_ms = 15_000,
    }});
    const reader = try std.Thread.spawn(.{}, PipeRead.run, .{PipeRead{
        .file = child.stdout.?,
        .list = &out,
        .max = 64 << 10,
    }});

    _ = child.wait() catch {};
    done.store(true, .release);
    watch.join();
    reader.join();
    return out.items;
}
