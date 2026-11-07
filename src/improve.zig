const std = @import("std");
const util = @import("util.zig");
const config = @import("config.zig");
const http = @import("http.zig");
const llm = @import("llm.zig");
const tools = @import("tools.zig");
const harness = @import("harness.zig");
const agent = @import("agent.zig");
const live = @import("live.zig");
const memory_mod = @import("memory.zig");
const traces_mod = @import("traces.zig");

pub const Mode = enum { parallel, sequential };

/// Escape `s` for embedding in a JSON string fragment (no quotes added).
fn je(alloc: std.mem.Allocator, s: []const u8) []const u8 {
    return util.jescape(alloc, s) catch "";
}

pub const Options = struct {
    k: u8 = 2,
    mode: Mode = .parallel,
    dry_run: bool = false,
    /// extra instruction folded into the task for every execution
    context: []const u8 = "",
};

const Execution = struct {
    text: []const u8,
    trajectory: []const u8,
    ok: bool,
};

const Candidate = struct {
    id: []const u8,
    rationale: []const u8,
    files: []FileEdit,
};

const FileEdit = struct {
    path: []const u8,
    op: []const u8, // write | append | delete
    content: []const u8,
};

/// A Ui that mirrors agent activity into a string buffer (the trajectory) while
/// also forwarding to an optional real Ui.
const CollectUi = struct {
    ui: tools.Ui,
    buf: *std.ArrayList(u8),
    inner: ?*const tools.Ui,

    fn make(buf: *std.ArrayList(u8), inner: ?*const tools.Ui) *CollectUi {
        const alloc = buf.allocator;
        const cu = alloc.create(CollectUi) catch @panic("oom");
        cu.* = .{
            .buf = buf,
            .inner = inner,
            .ui = .{
                .ctx = cu,
                .on_text = onText,
                .on_tool_call = onToolCall,
                .on_tool_result = onToolResult,
                .on_status = onStatus,
            },
        };
        return cu;
    }

    fn self(ctx: *anyopaque) *CollectUi {
        return @ptrCast(@alignCast(ctx));
    }

    fn onText(ctx: *anyopaque, text: []const u8) void {
        const c = self(ctx);
        if (c.inner) |u| {
            if (u.on_text) |f| f(u.ctx, text);
        }
    }
    fn onToolCall(ctx: *anyopaque, name: []const u8, summary: []const u8) void {
        const c = self(ctx);
        c.buf.writer().print("\n> {s}({s})\n", .{ name, summary }) catch {};
        if (c.inner) |u| {
            if (u.on_tool_call) |f| f(u.ctx, name, summary);
        }
    }
    fn onToolResult(ctx: *anyopaque, name: []const u8, preview: []const u8, ok: bool) void {
        const c = self(ctx);
        const cap: usize = 800;
        const p = if (preview.len > cap) preview[0..cap] else preview;
        c.buf.writer().print("{s}\n", .{p}) catch {};
        if (c.inner) |u| {
            if (u.on_tool_result) |f| f(u.ctx, name, preview, ok);
        }
    }
    fn onStatus(ctx: *anyopaque, msg: []const u8) void {
        const c = self(ctx);
        if (c.inner) |u| {
            if (u.on_status) |f| f(u.ctx, msg);
        }
    }
};

fn say(ui: ?*const tools.Ui, msg: []const u8) void {
    if (ui) |u| {
        if (u.on_status) |f| f(u.ctx, msg);
    }
}

/// Run one fresh agent on `task` against harness dir `hdir`.
fn runExecution(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    hdir: []const u8,
    root: []const u8,
    cwd: []const u8,
    task: []const u8,
    ui: ?*const tools.Ui,
) !Execution {
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const a = arena.allocator();

    var hh = harness.Harness{ .alloc = a, .dir = hdir, .root = root, .global = false };
    var ag = try agent.Agent.init(a, cfg, hclient, &hh, cwd, cwd, 0, "");
    if (traces_mod.Store.create(a)) |ts| ag.traces = ts;
    ag.sid = "improve";
    // Harness² learns across runs: probes recall prior lessons and the
    // turn is captured into the workspace's memory namespace.
    if (memory_mod.Memory.create(a, hclient, cfg.memory_key, cfg.memory_base,
        if (cfg.memory_namespace.len > 0) cfg.memory_namespace
        else memory_mod.Memory.defaultNs(a, cwd))) |mem|
    {
        ag.memory = mem;
        ag.mem_auto = cfg.memory_auto;
        ag.mem_doc_id = "xera-improve";
    }
    var buf = std.ArrayList(u8).init(a);
    const cu = CollectUi.make(&buf, ui);
    ag.ui = &cu.ui;
    // probes run unattended; auto-approve like exec --full-auto
    const outcome = ag.runTask(task) catch |e| {
        return .{
            .text = try std.fmt.allocPrint(alloc, "(execution failed: {s})", .{@errorName(e)}),
            .trajectory = try alloc.dupe(u8, buf.items),
            .ok = false,
        };
    };
    return .{
        .text = try alloc.dupe(u8, outcome.text),
        .trajectory = try alloc.dupe(u8, buf.items),
        .ok = true,
    };
}

/// Materialize a candidate harness dir: snapshot of base + edits applied.
fn materializeCandidate(
    alloc: std.mem.Allocator,
    base: *harness.Harness,
    dest: []const u8,
    edits: []const FileEdit,
) !void {
    const snap = try base.snapshot();
    var it = snap.iterator();
    while (it.next()) |e| {
        const p = try std.fs.path.join(alloc, &.{ dest, e.key_ptr.* });
        if (std.fs.path.dirname(p)) |parent| try std.fs.cwd().makePath(parent);
        const f = try std.fs.cwd().createFile(p, .{});
        defer f.close();
        try f.writeAll(e.value_ptr.*);
    }
    var cand = harness.Harness{ .alloc = alloc, .dir = dest, .root = dest, .global = false };
    for (edits) |ed| {
        cand.applyEdit(ed.path, ed.op, ed.content) catch |e| {
            std.debug.print("warning: candidate edit on {s} failed: {s}\n", .{ ed.path, @errorName(e) });
        };
    }
}

fn stripFences(s: []const u8) []const u8 {
    var t = std.mem.trim(u8, s, " \n\r\t");
    if (std.mem.startsWith(u8, t, "```")) {
        if (std.mem.indexOfScalar(u8, t, '\n')) |nl| t = t[nl + 1 ..];
        if (std.mem.lastIndexOf(u8, t, "```")) |end| t = t[0..end];
        t = std.mem.trim(u8, t, " \n\r\t");
    }
    return t;
}

fn extractJson(s: []const u8) []const u8 {
    const t = stripFences(s);
    const start = std.mem.indexOfScalar(u8, t, '{') orelse return t;
    const end = std.mem.lastIndexOfScalar(u8, t, '}') orelse return t;
    if (end <= start) return t;
    return t[start .. end + 1];
}

fn harnessManifest(alloc: std.mem.Allocator, h: *harness.Harness) ![]u8 {
    var b = std.ArrayList(u8).init(alloc);
    const w = b.writer();
    const snap = try h.snapshot();
    var it = snap.iterator();
    while (it.next()) |e| {
        const trimmed = try util.truncateMiddle(alloc, e.value_ptr.*, 6000);
        try w.print("=== {s} ===\n{s}\n\n", .{ e.key_ptr.*, trimmed });
    }
    return b.toOwnedSlice();
}

const propose_prompt =
    \\You are a harness editor in the Harness² framework. A harness is the editable
    \\system around an agent model: system prompt, guardrails, tool guidance, memory,
    \\lessons, subagent policy, and skills as markdown files.
    \\
    \\You are given a TASK, the BASE harness files, and a BASE execution trajectory for
    \\the same task. Propose {d} candidate harness revisions. Make candidates
    \\meaningfully different: at least one should be a general-purpose improvement
    \\good for any task; the rest may be domain- or task-specific. Edit only files
    \\inside the harness (e.g. tool_guidance.md, subagents.md, guardrails.md,
    \\skills/<name>.md, system_prompt.md). Small surgical edits beat rewrites.
    \\
    \\Respond with ONLY JSON (no prose, no fences):
    \\{{"candidates":[{{"id":"c1","rationale":"short why",
    \\ "files":[{{"path":"tool_guidance.md","op":"write|append|delete","content":"..."}}]}}]}}
;

fn propose(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    h: *harness.Harness,
    task: []const u8,
    base_traj: []const u8,
    k: u8,
    extra: []const u8,
) ![]Candidate {
    const manifest = try harnessManifest(alloc, h);
    const traj_cap = try util.truncateMiddle(alloc, base_traj, 12_000);
    const sys = try std.fmt.allocPrint(alloc, propose_prompt ++
        "The base execution trajectory for the task follows, delimited:\n<base_trajectory>\n{s}\n</base_trajectory>\n" ++
        "Current harness files:\n{s}", .{ k, traj_cap, manifest });
    var user = std.ArrayList(u8).init(alloc);
    try user.writer().print("TASK:\n{s}\n{s}", .{ task, extra });

    const msgs = [_]std.json.Value{
        try llm.systemMessage(alloc, sys),
        try llm.userMessage(alloc, user.items),
    };
    const res = try llm.chat(hclient, alloc, cfg, &msgs, &[_]llm.ToolSchema{}, .{
        .max_tokens = 8192,
        .stream = false,
        .no_tools = true,
    });
    return parseCandidates(alloc, res.content);
}

fn parseCandidates(alloc: std.mem.Allocator, text: []const u8) ![]Candidate {
    const js = extractJson(text);
    const root = try util.jparse(alloc, js);
    var out = std.ArrayList(Candidate).init(alloc);
    const cands = util.jget(root, "candidates") orelse return error.NoCandidates;
    if (cands != .array) return error.NoCandidates;
    for (cands.array.items, 0..) |c, i| {
        var cand = Candidate{
            .id = util.jstr(c, "id") orelse try std.fmt.allocPrint(alloc, "c{d}", .{i + 1}),
            .rationale = util.jstr(c, "rationale") orelse "",
            .files = &.{},
        };
        var files = std.ArrayList(FileEdit).init(alloc);
        if (util.jget(c, "files")) |fs| {
            if (fs == .array) {
                for (fs.array.items) |f| {
                    const path = util.jstr(f, "path") orelse continue;
                    const op = util.jstr(f, "op") orelse "write";
                    if (!std.mem.eql(u8, op, "write") and !std.mem.eql(u8, op, "append") and
                        !std.mem.eql(u8, op, "delete")) continue;
                    try files.append(.{
                        .path = path,
                        .op = op,
                        .content = util.jstr(f, "content") orelse "",
                    });
                }
            }
        }
        cand.files = try files.toOwnedSlice();
        try out.append(cand);
    }
    return out.toOwnedSlice();
}

const compose_prompt =
    \\You are a harness composer in the Harness² framework. You are given a TASK, the
    \\BASE execution result, and several CANDIDATE harness revisions each with their
    \\probe execution result on the same task. Use the behavioral contrast between
    \\executions — not a ground truth — to decide what helped.
    \\
    \\Choose the single best candidate to adopt, or merge the strongest ideas across
    \\candidates. Then emit the FINAL set of harness file edits to apply, and a short
    \\"lesson" capturing the evidence-supported insight (one or two sentences) to be
    \\stored in the improvement harness for future tasks.
    \\
    \\Respond with ONLY JSON (no prose, no fences):
    \\{"adopt":"c2 | merge","rationale":"short why",
    \\ "final_files":[{"path":"tool_guidance.md","op":"write|append|delete","content":"..."}],
    \\ "lesson":"evidence-supported lesson learned"}
;

fn compose(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    task: []const u8,
    base: Execution,
    cands: []const Candidate,
    execs: []const Execution,
) !ComposeResult {
    var user = std.ArrayList(u8).init(alloc);
    const w = user.writer();
    try w.print("TASK:\n{s}\n\n", .{task});
    try w.print("BASE result:\n{s}\n\n", .{try util.truncateMiddle(alloc, base.text, 4000)});
    for (cands, 0..) |c, i| {
        try w.print("=== candidate {s} ===\nrationale: {s}\nedits:\n", .{ c.id, c.rationale });
        for (c.files) |f| {
            try w.print("- {s} {s} ({d} bytes)\n", .{ f.op, f.path, f.content.len });
        }
        const e = if (i < execs.len) execs[i] else Execution{ .text = "", .trajectory = "", .ok = false };
        try w.print("probe result:\n{s}\ntrajectory excerpt:\n{s}\n\n", .{
            try util.truncateMiddle(alloc, e.text, 3000),
            try util.truncateMiddle(alloc, e.trajectory, 3000),
        });
    }
    const msgs = [_]std.json.Value{
        try llm.systemMessage(alloc, compose_prompt),
        try llm.userMessage(alloc, user.items),
    };
    const res = try llm.chat(hclient, alloc, cfg, &msgs, &[_]llm.ToolSchema{}, .{
        .max_tokens = 8192,
        .stream = false,
        .no_tools = true,
    });
    const js = extractJson(res.content);
    const root = try util.jparse(alloc, js);
    var out = ComposeResult{
        .adopt = util.jstr(root, "adopt") orelse "merge",
        .rationale = util.jstr(root, "rationale") orelse "",
        .lesson = util.jstr(root, "lesson") orelse "",
        .final_files = &.{},
    };
    var files = std.ArrayList(FileEdit).init(alloc);
    if (util.jget(root, "final_files")) |fs| {
        if (fs == .array) {
            for (fs.array.items) |f| {
                const path = util.jstr(f, "path") orelse continue;
                const op = util.jstr(f, "op") orelse "write";
                try files.append(.{
                    .path = path,
                    .op = op,
                    .content = util.jstr(f, "content") orelse "",
                });
            }
        }
    }
    out.final_files = try files.toOwnedSlice();
    return out;
}

const ComposeResult = struct {
    adopt: []const u8,
    rationale: []const u8,
    final_files: []FileEdit,
    lesson: []const u8,
};

pub fn run(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    h: *harness.Harness,
    cwd: []const u8,
    task: []const u8,
    opts: Options,
    ui: ?*const tools.Ui,
) !void {
    try h.ensure();

    const runs_root = try std.fs.path.join(alloc, &.{ h.root, ".xera-rsi", "runs" });
    const ts = std.time.milliTimestamp();
    const run_dir = try std.fmt.allocPrint(alloc, "{s}/improve-{d}", .{ runs_root, ts });
    try std.fs.cwd().makePath(run_dir);

    const full_task = if (opts.context.len > 0)
        try std.fmt.allocPrint(alloc, "{s}\n\n{s}", .{ task, opts.context })
    else
        task;

    var round: u8 = 0;
    const rounds: u8 = if (opts.mode == .sequential) opts.k else 1;
    const per_round: u8 = if (opts.mode == .sequential) 1 else opts.k;
    var evidence = std.ArrayList(u8).init(alloc);
    const ew = evidence.writer();

    // harness2-style evidence tree under <run_dir>/context/
    const ctx_dir = try std.fs.path.join(alloc, &.{ run_dir, "context" });
    try ctxFile(alloc, ctx_dir, "task.md", full_task);
    {
        const snap = try h.snapshot();
        try copyTree(alloc, snap, try std.fs.path.join(alloc, &.{ ctx_dir, "harness" }));
    }

    live.improvement(alloc, try std.fmt.allocPrint(alloc,
        "\"phase\":\"start\",\"task\":\"{s}\",\"k\":{d},\"mode\":\"{s}\",\"run_dir\":\"{s}\",\"harness\":\"{s}\",\"cwd\":\"{s}\"",
        .{ je(alloc, full_task), opts.k, @tagName(opts.mode), je(alloc, run_dir), je(alloc, h.dir), je(alloc, cwd) }));

    while (round < rounds) : (round += 1) {
        if (rounds > 1) {
            const m = try std.fmt.allocPrint(alloc, "── improve round {d}/{d} ──", .{ round + 1, rounds });
            say(ui, m);
        }

        // 1) BASE execution with the current harness
        say(ui, "[1/4] base execution with current harness");
        const base = try runExecution(alloc, cfg, hclient, h.dir, h.root, cwd, full_task, ui);
        try writeFile(alloc, run_dir, try std.fmt.allocPrint(alloc, "base-{d}.txt", .{round}), base.trajectory);
        try ctxFile(alloc, ctx_dir, try std.fmt.allocPrint(alloc, "rollouts/base-r{d}.md", .{round}), base.trajectory);
        try ctxFile(alloc, ctx_dir, try std.fmt.allocPrint(alloc, "deliverables/base-r{d}.md", .{round}), base.text);
        try ew.print("## round {d} base\n{s}\n\n", .{ round, try util.truncateMiddle(alloc, base.text, 2000) });

        // 2) PROPOSE candidates
        const m2 = try std.fmt.allocPrint(alloc, "[2/4] propose {d} candidate harness edit(s)", .{per_round});
        say(ui, m2);
        const cands = propose(alloc, cfg, hclient, h, full_task, base.trajectory, per_round, "") catch |e| {
            say(ui, "proposer returned unusable output; aborting round");
            return e;
        };
        if (cands.len == 0) {
            say(ui, "proposer returned no candidates; aborting");
            return error.NoCandidates;
        }
        const cands_json = try std.fmt.allocPrint(alloc, "{d} candidate(s) proposed", .{cands.len});
        say(ui, cands_json);

        if (opts.dry_run) {
            var b = std.ArrayList(u8).init(alloc);
            for (cands) |c| {
                try b.writer().print("\n=== {s} ===\n{s}\n", .{ c.id, c.rationale });
                for (c.files) |f| try b.writer().print("  {s} {s} ({d} bytes)\n", .{ f.op, f.path, f.content.len });
            }
            try writeFile(alloc, run_dir, "dry-run.txt", b.items);
            say(ui, b.items);
            say(ui, "dry run complete; nothing applied");
            live.improvement(alloc, try std.fmt.allocPrint(alloc,
                "\"phase\":\"dry_run\",\"task\":\"{s}\",\"candidates\":{d},\"run_dir\":\"{s}\"",
                .{ je(alloc, full_task), cands.len, je(alloc, run_dir) }));
            return;
        }

        // 3) PROBE each candidate on the same task
        var execs = try alloc.alloc(Execution, cands.len);
        for (cands, 0..) |c, i| {
            const m3 = try std.fmt.allocPrint(alloc, "[3/4] probe candidate {s} ({d}/{d})", .{ c.id, i + 1, cands.len });
            say(ui, m3);
            const cdir = try std.fmt.allocPrint(alloc, "{s}/r{d}-{s}-harness", .{ run_dir, round, c.id });
            try materializeCandidate(alloc, h, cdir, c.files);
            // harness2 evidence layout: context/pool/<id>/{harness/,harness.diff,rollout.md,produced.md}
            const pool_dir = try std.fmt.allocPrint(alloc, "{s}/pool/{s}", .{ ctx_dir, c.id });
            {
                const base_snap = try h.snapshot();
                try copyTree(alloc, blk: {
                    var cand = harness.Harness{ .alloc = alloc, .dir = cdir, .root = cdir, .global = false };
                    break :blk try cand.snapshot();
                }, try std.fs.path.join(alloc, &.{ pool_dir, "harness" }));
                const diff = try harnessDiff(alloc, base_snap, cdir);
                try ctxFile(alloc, pool_dir, "harness.diff", diff);
            }
            // surface validation — an invalid candidate is evidence too, not a probe
            var cand_h = harness.Harness{ .alloc = alloc, .dir = cdir, .root = cdir, .global = false };
            const problems = cand_h.validate(alloc) catch &[_][]const u8{};
            if (problems.len > 0) {
                var pb = std.ArrayList(u8).init(alloc);
                for (problems) |p| try pb.writer().print("- {s}\n", .{p});
                execs[i] = .{
                    .text = "candidate harness rejected by surface validation",
                    .trajectory = try alloc.dupe(u8, pb.items),
                    .ok = false,
                };
                try ctxFile(alloc, pool_dir, "rollout.md", "(skipped — invalid surface)");
                try ctxFile(alloc, pool_dir, "produced.md", "(none)");
                try writeFile(alloc, run_dir, try std.fmt.allocPrint(alloc, "r{d}-{s}.txt", .{ round, c.id }), pb.items);
                continue;
            }
            execs[i] = try runExecution(alloc, cfg, hclient, cdir, h.root, cwd, full_task, ui);
            try writeFile(alloc, run_dir, try std.fmt.allocPrint(alloc, "r{d}-{s}.txt", .{ round, c.id }), execs[i].trajectory);
            try ctxFile(alloc, pool_dir, "rollout.md", execs[i].trajectory);
            try ctxFile(alloc, pool_dir, "produced.md", execs[i].text);
            try ew.print("## round {d} candidate {s}\nrationale: {s}\nresult:\n{s}\n\n", .{
                round, c.id, c.rationale, try util.truncateMiddle(alloc, execs[i].text, 2000),
            });
        }

        // 4) COMPOSE the refined harness
        say(ui, "[4/4] compose refined harness from behavioral contrasts");
        const comp = try compose(alloc, cfg, hclient, full_task, base, cands, execs);
        for (comp.final_files) |f| {
            h.applyEdit(f.path, f.op, f.content) catch |e| {
                std.debug.print("warning: failed to apply {s} {s}: {s}\n", .{ f.op, f.path, @errorName(e) });
            };
        }
        if (comp.lesson.len > 0) {
            const lesson = try std.fmt.allocPrint(alloc, "- ({d}) {s}\n", .{ ts, comp.lesson });
            h.applyEdit("lessons.md", "append", lesson) catch {};
        }
        const report = try std.fmt.allocPrint(alloc,
            \\adopt: {s}
            \\rationale: {s}
            \\lesson: {s}
            \\final edits: {d} file(s)
        , .{ comp.adopt, comp.rationale, comp.lesson, comp.final_files.len });
        try writeFile(alloc, run_dir, try std.fmt.allocPrint(alloc, "compose-{d}.txt", .{round}), report);
        try ctxFile(alloc, ctx_dir, try std.fmt.allocPrint(alloc, "decisions/compose-r{d}.md", .{round}), report);
        try ew.print("## round {d} compose\n{s}\n\n", .{ round, report });
        say(ui, report);
        live.improvement(alloc, try std.fmt.allocPrint(alloc,
            "\"phase\":\"compose\",\"round\":{d},\"adopt\":\"{s}\",\"rationale\":\"{s}\",\"lesson\":\"{s}\",\"files\":{d},\"run_dir\":\"{s}\"",
            .{ round, je(alloc, comp.adopt), je(alloc, comp.rationale), je(alloc, comp.lesson), comp.final_files.len, je(alloc, run_dir) }));
    }

    // persist the final adopted harness into the library (cross-task evidence)
    const lib_id = try std.fmt.allocPrint(alloc, "improve-{d}", .{ts});
    const lib_dir = libraryAdopt(alloc, h, lib_id, full_task,
        "adopted after improve run") catch null;
    if (lib_dir) |ld| {
        const m = try std.fmt.allocPrint(alloc, "harness snapshot adopted → library/{s}", .{ld});
        say(ui, m);
    }

    try writeFile(alloc, run_dir, "evidence.md", evidence.items);
    const done = try std.fmt.allocPrint(alloc, "improve complete → {s}", .{run_dir});
    say(ui, done);
}

fn writeFile(alloc: std.mem.Allocator, dir: []const u8, name: []const u8, content: []const u8) !void {
    const p = try std.fs.path.join(alloc, &.{ dir, name });
    if (std.fs.path.dirname(p)) |parent| try std.fs.cwd().makePath(parent);
    const f = try std.fs.cwd().createFile(p, .{});
    defer f.close();
    try f.writeAll(content);
}

/// Write a rel-path file under the context tree (context/<rel>).
fn ctxFile(alloc: std.mem.Allocator, ctx_dir: []const u8, rel: []const u8, content: []const u8) !void {
    try writeFile(alloc, ctx_dir, rel, content);
}

/// Copy a whole harness snapshot map into a directory on disk.
fn copyTree(alloc: std.mem.Allocator, snap: std.StringArrayHashMap([]const u8), dest: []const u8) !void {
    var it = snap.iterator();
    while (it.next()) |e| {
        try writeFile(alloc, dest, e.key_ptr.*, e.value_ptr.*);
    }
}

/// Per-file diff summary between the live harness and a materialized candidate.
fn harnessDiff(
    alloc: std.mem.Allocator,
    before: std.StringArrayHashMap([]const u8),
    cand_dir: []const u8,
) ![]u8 {
    var out = std.ArrayList(u8).init(alloc);
    const w = out.writer();
    var cand = harness.Harness{ .alloc = alloc, .dir = cand_dir, .root = cand_dir, .global = false };
    const after = try cand.snapshot();

    var it = before.iterator();
    while (it.next()) |e| {
        const old = e.value_ptr.*;
        if (after.get(e.key_ptr.*)) |new| {
            if (std.mem.eql(u8, old, new)) continue;
            try w.print("--- current/{s}\n+++ candidate/{s}\n", .{ e.key_ptr.*, e.key_ptr.* });
            // naive line diff: lines removed / lines added (set difference)
            var added = std.ArrayList([]const u8).init(alloc);
            var removed = std.ArrayList([]const u8).init(alloc);
            var old_lines = std.StringHashMap(void).init(alloc);
            var oit = std.mem.splitScalar(u8, old, '\n');
            while (oit.next()) |l| try old_lines.put(l, {});
            var nit = std.mem.splitScalar(u8, new, '\n');
            while (nit.next()) |l| {
                if (!old_lines.contains(l)) try added.append(l);
            }
            var new_lines = std.StringHashMap(void).init(alloc);
            nit = std.mem.splitScalar(u8, new, '\n');
            while (nit.next()) |l| try new_lines.put(l, {});
            oit = std.mem.splitScalar(u8, old, '\n');
            while (oit.next()) |l| {
                if (!new_lines.contains(l)) try removed.append(l);
            }
            for (removed.items[0..@min(removed.items.len, 60)]) |l|
                try w.print("- {s}\n", .{l});
            for (added.items[0..@min(added.items.len, 60)]) |l|
                try w.print("+ {s}\n", .{l});
            try w.writeAll("\n");
        } else {
            try w.print("### removed: {s}\n\n", .{e.key_ptr.*});
        }
    }
    var it2 = after.iterator();
    while (it2.next()) |e| {
        if (!before.contains(e.key_ptr.*)) {
            try w.print("### added: {s} ({d} bytes)\n\n", .{ e.key_ptr.*, e.value_ptr.len });
        }
    }
    return out.toOwnedSlice();
}

/// Persist an adopted harness into the library (~/.xera-rsi/library/<id>/).
fn libraryAdopt(
    alloc: std.mem.Allocator,
    h: *harness.Harness,
    id: []const u8,
    task: []const u8,
    note: []const u8,
) ![]const u8 {
    const home = try util.homeDir(alloc);
    const lib = try std.fs.path.join(alloc, &.{ home, ".xera-rsi", "library" });
    const dir = try std.fs.path.join(alloc, &.{ lib, id });
    try std.fs.cwd().makePath(dir);
    const snap = try h.snapshot();
    try copyTree(alloc, snap, dir);
    try writeFile(alloc, dir, "INDEX.md",
        try std.fmt.allocPrint(alloc, "# {s}\n\ntask: {s}\n\n{s}\n", .{ id, task, note }));
    const idx = try std.fs.path.join(alloc, &.{ lib, "INDEX.md" });
    const f = try std.fs.cwd().createFile(idx, .{ .truncate = false });
    defer f.close();
    const end = try f.getEndPos();
    try f.seekFromEnd(0);
    if (end == 0) try f.writeAll("# harness library — adopted snapshots\n\n");
    try f.writer().print("- {s}  adopt={s}\n", .{ id, util.firstLine(task) });
    return dir;
}
