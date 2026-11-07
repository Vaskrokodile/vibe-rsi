const std = @import("std");
const util = @import("util.zig");
const config = @import("config.zig");
const http = @import("http.zig");
const llm = @import("llm.zig");
const tools = @import("tools.zig");
const harness = @import("harness.zig");
const agent = @import("agent.zig");
const session = @import("session.zig");
const live = @import("live.zig");
const memory_mod = @import("memory.zig");
const traces_mod = @import("traces.zig");
const improve = @import("improve.zig");

const a = util.ansi;

pub const Term = struct {
    file: std.fs.File,
    w: std.fs.File.Writer,
    color: bool,

    pub fn init() Term {
        const f = std.io.getStdOut();
        return .{
            .file = f,
            .w = f.writer(),
            .color = f.supportsAnsiEscapeCodes(),
        };
    }

    pub fn print(self: *Term, comptime fmt: []const u8, args: anytype) void {
        self.w.print(fmt, args) catch {};
    }

    pub fn write(self: *Term, s: []const u8) void {
        self.w.writeAll(s) catch {};
    }

    pub fn style(self: *Term, s: []const u8) void {
        if (self.color) self.write(s);
    }
};

var term: Term = undefined;

pub fn initTerm() void {
    term = Term.init();
    if (!term.color) util.enableColor(false);
}

fn cbText(ctx: *anyopaque, text: []const u8) void {
    _ = ctx;
    term.write(text);
}

fn cbReasoning(ctx: *anyopaque, text: []const u8) void {
    _ = ctx;
    if (term.color) {
        term.style(a.dim ++ a.italic);
        term.write(text);
        term.style(a.reset);
    } else {
        term.write(text);
    }
}

fn cbToolCall(ctx: *anyopaque, name: []const u8, summary: []const u8) void {
    _ = ctx;
    term.write("\n");
    term.style(a.cyan);
    term.print(" ▸ {s}", .{name});
    term.style(a.gray);
    const s = if (summary.len > 160) summary[0..160] else summary;
    term.print(" {s}", .{s});
    term.style(a.reset);
    term.write("\n");
}

fn cbToolResult(ctx: *anyopaque, name: []const u8, preview: []const u8, ok: bool) void {
    _ = ctx;
    _ = name;
    var lines: usize = 0;
    var end: usize = 0;
    while (lines < 8 and end < preview.len) {
        if (std.mem.indexOfScalarPos(u8, preview, end, '\n')) |idx| {
            end = idx + 1;
        } else {
            end = preview.len;
        }
        lines += 1;
    }
    const show = preview[0..end];
    term.style(a.gray);
    var it = std.mem.splitScalar(u8, show, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        term.print("   {s}\n", .{line});
    }
    term.style(if (ok) a.green else a.red);
    term.print("   {s}\n", .{if (ok) "✓" else "✗"});
    if (preview.len > end) {
        term.style(a.gray);
        term.print("   … ({d} more bytes)\n", .{preview.len - end});
    }
    term.style(a.reset);
}

fn cbStatus(ctx: *anyopaque, msg: []const u8) void {
    _ = ctx;
    term.style(a.magenta);
    term.print("◆ {s}\n", .{msg});
    term.style(a.reset);
}

fn ui() tools.Ui {
    return .{
        .ctx = undefined,
        .on_text = cbText,
        .on_reasoning = cbReasoning,
        .on_tool_call = cbToolCall,
        .on_tool_result = cbToolResult,
        .on_status = cbStatus,
    };
}

/// quiet UI for non-interactive runs: tool lines + status, streamed text
pub fn quietUi() tools.Ui {
    return .{
        .ctx = undefined,
        .on_text = cbText,
        .on_tool_call = cbToolCall,
        .on_status = cbStatus,
    };
}

// ---------- approval ----------

fn readLine(alloc: std.mem.Allocator) !?[]const u8 {
    const stdin = std.io.getStdIn().reader();
    const line = stdin.readUntilDelimiterOrEofAlloc(alloc, '\n', 1 << 20) catch return null;
    if (line == null) return null;
    return std.mem.trim(u8, line.?, " \r\t");
}

fn askApproval(ctx: *anyopaque, tool_name: []const u8, summary: []const u8, danger: tools.Danger) anyerror!tools.ApprovalDecision {
    _ = ctx;
    const marker: []const u8 = switch (danger) {
        .write => "write",
        .shell => "shell",
        .safe => "tool",
    };
    term.write("\n");
    term.style(a.yellow);
    const s = if (summary.len > 400) summary[0..400] else summary;
    term.print(" ? {s} [{s}] {s}\n", .{ tool_name, marker, s });
    term.style(a.reset);
    while (true) {
        term.write("   allow? [y]es/[n]o/[a]lways: ");
        var stack_alloc_buf: [4096]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&stack_alloc_buf);
        const line = readLine(fba.allocator()) catch return error.ApprovalReadFailed;
        const l = line orelse return error.ApprovalReadFailed;
        if (l.len == 0) continue;
        switch (l[0]) {
            'y', 'Y' => return .allow,
            'a', 'A' => return .always,
            'n', 'N' => return .deny,
            else => continue,
        }
    }
}

pub fn approver() tools.Approver {
    return .{ .ctx = undefined, .ask = askApproval };
}

// ---------- repl ----------

pub fn replLoop(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    h: *harness.Harness,
    cwd: []const u8,
    initial_history: ?std.ArrayList(std.json.Value),
) !void {
    var ag = try agent.Agent.init(alloc, cfg, hclient, h, cwd, cwd, 0, "");
    if (traces_mod.Store.create(alloc)) |ts| ag.traces = ts;
    if (initial_history) |hist| {
        // replace messages: keep new system prompt + replay prior non-system messages
        for (hist.items) |m| {
            const role = util.jstr(m, "role") orelse continue;
            if (std.mem.eql(u8, role, "system")) continue;
            try ag.messages.append(m);
        }
    }
    var log = try session.create(alloc, cfg, cwd);
    ag.slog = &log;
    ag.sid = log.id;
    if (live.Live.create(alloc, log.id) catch null) |lv| {
        ag.livelog = lv;
        lv.sessionStart(cwd, cfg.model, h.dir, "repl");
    }
    // long-term memory (supermemory) — recall + capture like the TUI
    if (memory_mod.Memory.create(alloc, hclient, cfg.memory_key, cfg.memory_base,
        if (cfg.memory_namespace.len > 0) cfg.memory_namespace
        else memory_mod.Memory.defaultNs(alloc, cwd))) |mem|
    {
        ag.memory = mem;
        ag.mem_auto = cfg.memory_auto;
        ag.mem_doc_id = try std.fmt.allocPrint(alloc, "xera-{s}-main", .{log.id});
    }
    var ui_obj = ui();
    ag.ui = &ui_obj;
    ag.approver = &approver_inst;
    ag.interactive = true;

    term.style(a.bold ++ a.cyan);
    term.print("xera-rsi", .{});
    term.style(a.reset);
    term.print("  model={s}  cwd={s}\n", .{ cfg.model, cwd });
    term.style(a.gray);
    term.print("harness: {s}   session: {s}\n", .{ h.dir, log.id });
    term.print("type /help for commands, /exit to quit\n", .{});
    term.style(a.reset);

    while (true) {
        term.write("\n");
        term.style(a.bold ++ a.green);
        term.write("xera> ");
        term.style(a.reset);

        var stack_buf: [64 * 1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&stack_buf);
        const line_opt = readLine(fba.allocator()) catch {
            term.print("(input error — line too long?)\n", .{});
            continue;
        };
        const line = line_opt orelse break; // EOF
        if (line.len == 0) continue;

        if (line[0] == '/') {
            const keep_going = try handleCommand(alloc, cfg, hclient, h, cwd, &ag, line);
            if (!keep_going) break;
            continue;
        }

        const owned_line = try alloc.dupe(u8, line);
        const outcome = ag.runTask(owned_line) catch |e| {
            term.style(a.red);
            term.print("\nerror: {s}\n", .{@errorName(e)});
            term.style(a.reset);
            continue;
        };
        _ = outcome;
        term.write("\n");
    }
    term.style(a.gray);
    term.print("\nsession saved to {s}\n", .{log.path});
    term.style(a.reset);
}

var approver_inst = tools.Approver{ .ctx = undefined, .ask = askApproval };

fn handleCommand(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    h: *harness.Harness,
    cwd: []const u8,
    ag: *agent.Agent,
    line: []const u8,
) !bool {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    const cmd = it.next() orelse return true;
    const rest = std.mem.trim(u8, line[cmd.len..], " ");

    if (eqAny(cmd, &.{ "/exit", "/quit", "/q" })) return false;
    if (eqAny(cmd, &.{ "/help", "/?" })) {
        term.print(
            \\commands:
            \\  /help             this list
            \\  /exit             quit (Ctrl+D also works)
            \\  /clear            drop conversation history (keeps harness)
            \\  /model [name]     show or switch model
            \\  /harness          show harness location + components
            \\  /sessions         list saved sessions
            \\  /resume <id>      load a session into context
            \\  /improve <task>   run Harness² propose-probe-compose on a task
            \\  /auto             toggle auto-approve (currently: {s})
            \\anything else is sent to the agent.
        , .{@tagName(cfg.approval)});
        term.write("\n");
        return true;
    }
    if (eqAny(cmd, &.{"/clear"})) {
        ag.messages.clearRetainingCapacity();
        const sys = try h.systemPrompt(cwd, "");
        try ag.messages.append(try llm.systemMessage(alloc, sys));
        term.print("(history cleared)\n", .{});
        return true;
    }
    if (eqAny(cmd, &.{"/model"})) {
        if (rest.len == 0) {
            term.print("model: {s}\n", .{cfg.model});
        } else {
            // cfg is const — patch via mutable copy? cfg points to caller cfg; mutate allowed via *? It's *const.
            term.print("(model switch takes effect next launch — set XERA_MODEL or config.json)\n", .{});
        }
        return true;
    }
    if (eqAny(cmd, &.{"/harness"})) {
        term.print("harness dir: {s}\n", .{h.dir});
        for (harness.component_files) |f| {
            const body = h.read(f);
            term.print("  {s: <18} {d} bytes\n", .{ f, body.len });
        }
        const skills = h.listSkills() catch &[_]harness.NamedItem{};
        if (skills.len > 0) {
            term.print("  skills: ", .{});
            for (skills) |s| term.print("{s} ", .{s.name});
            term.write("\n");
        }
        const agents = h.listAgents() catch &[_]harness.NamedItem{};
        if (agents.len > 0) {
            term.print("  sub_agents: ", .{});
            for (agents) |agt| term.print("{s} ", .{agt.name});
            term.write("\n");
        }
        return true;
    }
    if (eqAny(cmd, &.{"/sessions"})) {
        const infos = session.list(alloc, cfg) catch &[_]session.Info{};
        if (infos.len == 0) {
            term.print("(no sessions)\n", .{});
        }
        for (infos) |info| {
            term.print("  {s}  {d} turn(s)  {s}\n    {s}\n", .{ info.id, info.turns, info.model, info.cwd });
        }
        return true;
    }
    if (eqAny(cmd, &.{"/resume"})) {
        if (rest.len == 0) {
            term.print("usage: /resume <session-id>\n", .{});
            return true;
        }
        const msgs = session.loadMessages(alloc, cfg, rest) catch |e| {
            term.print("cannot load {s}: {s}\n", .{ rest, @errorName(e) });
            return true;
        };
        ag.messages.clearRetainingCapacity();
        const sys = try h.systemPrompt(cwd, "");
        try ag.messages.append(try llm.systemMessage(alloc, sys));
        var n: usize = 0;
        for (msgs.items) |m| {
            const role = util.jstr(m, "role") orelse continue;
            if (std.mem.eql(u8, role, "system")) continue;
            try ag.messages.append(m);
            n += 1;
        }
        term.print("resumed {d} message(s) from {s}\n", .{ n, rest });
        return true;
    }
    if (eqAny(cmd, &.{"/improve"})) {
        if (rest.len == 0) {
            term.print("usage: /improve <task>\n", .{});
            return true;
        }
        var ui_obj = ui();
        improve.run(alloc, cfg, hclient, h, cwd, rest, .{}, &ui_obj) catch |e| {
            term.style(a.red);
            term.print("improve failed: {s}\n", .{@errorName(e)});
            term.style(a.reset);
        };
        return true;
    }
    if (eqAny(cmd, &.{"/auto"})) {
        term.print("(approval mode is fixed at launch — pass -y/--full-auto)\n", .{});
        return true;
    }
    term.print("unknown command {s} — try /help\n", .{cmd});
    return true;
}

fn eqAny(s: []const u8, opts: []const []const u8) bool {
    for (opts) |o| if (std.mem.eql(u8, s, o)) return true;
    return false;
}
