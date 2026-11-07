//! RSI loop: recursive self-improvement driver.
//!
//! Two entry points:
//!   - `selfCheck` — after every interactive answer, a cheap verification
//!     pass judges whether the turn actually completed the request. A
//!     failure produces an RSI *proposal* (never an automatic retry) that
//!     the user accepts or rejects.
//!   - `runLoop` — on accept (or `/rsi-loop`), the loop runs:
//!       1. analyze  — read the trace store (tool calls, skills, token
//!          medians, past failures, the 3D space projection) and produce
//!          a diagnosis + plan
//!       2. harness  — apply evidence-backed edits through the refinement
//!          subsystem (system files, skills, subagents, tool guidance)
//!       3. train    — when analysis says capability (not context) is the
//!          gap, build/pull a dataset and launch ETM training

const std = @import("std");
const util = @import("util.zig");
const config = @import("config.zig");
const http = @import("http.zig");
const llm = @import("llm.zig");
const harness = @import("harness.zig");
const traces = @import("traces.zig");
const refine_mod = @import("refine.zig");
const train = @import("train.zig");

pub const Verdict = struct {
    ok: bool,
    gap: []const u8 = "",
};

/// Where RSI output lands — the TUI and the repl each supply one.
pub const Sink = struct {
    ctx: *anyopaque,
    /// short phase/status lines ("analyzing traces…")
    status: *const fn (ctx: *anyopaque, msg: []const u8) void,
    /// the final report body, rendered as assistant text
    report: *const fn (ctx: *anyopaque, text: []const u8) void,
    /// training indicator: on=true shows "model is improving itself"
    train: ?*const fn (ctx: *anyopaque, on: bool, label: []const u8) void = null,
};

fn tell(sink: *const Sink, msg: []const u8) void {
    sink.status(sink.ctx, msg);
}
fn tellF(sink: *const Sink, alloc: std.mem.Allocator, comptime fmt: []const u8, args: anytype) void {
    const m = std.fmt.allocPrint(alloc, fmt, args) catch return;
    sink.status(sink.ctx, m);
}
fn trainState(sink: *const Sink, on: bool, label: []const u8) void {
    if (sink.train) |f| f(sink.ctx, on, label);
}

const VERIFY_SYS =
    \\You are a strict completion verifier inside an agentic coding tool.
    \\Given the user's request and the assistant's final response, judge
    \\whether the request was FULLY accomplished — not whether it was
    \\plausible-sounding. Partial work, errors, or unanswered parts = fail.
    \\Reply with ONLY a JSON object:
    \\{"ok": true|false, "gap": "<one line: what's missing or wrong>"}
    \\Use "gap": "" when ok is true.
;

/// Verification pass after an answer. Cheap: ~two tiny messages, no tools,
/// non-streaming. A failure is NOT retried — it becomes an RSI proposal.
pub fn selfCheck(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    prompt: []const u8,
    answer: []const u8,
) !Verdict {
    const user = try std.fmt.allocPrint(alloc,
        "REQUEST:\n{s}\n\nFINAL RESPONSE:\n{s}",
        .{ util.truncateMiddle(alloc, prompt, 6000) catch prompt,
           util.truncateMiddle(alloc, answer, 6000) catch answer });
    const msgs = [_]std.json.Value{
        try llm.systemMessage(alloc, VERIFY_SYS),
        try llm.userMessage(alloc, user),
    };
    const res = llm.chat(hclient, alloc, cfg, &msgs, &.{}, .{
        .max_tokens = 300,
        .temperature = 0.0,
        .stream = false,
        .no_tools = true,
    }) catch return error.VerifyFailed;
    const body = res.content;
    const start = std.mem.indexOfScalar(u8, body, '{') orelse return error.VerifyBadReply;
    const end = std.mem.lastIndexOfScalar(u8, body, '}') orelse return error.VerifyBadReply;
    const v = util.jparse(alloc, body[start .. end + 1]) catch return error.VerifyBadReply;
    const ok = util.jbool(v, "ok") orelse return error.VerifyBadReply;
    return .{ .ok = ok, .gap = util.jstr(v, "gap") orelse "" };
}

/// The proposal shown to the user: what failed + what the loop will do.
pub const Proposal = struct {
    focus: []const u8, // failed prompt, or the /rsi-loop argument
    gap: []const u8, // verifier's one-line diagnosis ("" when manual)
    stats_line: []const u8, // one-line trace digest for the panel
    steps: [][]const u8,
};

pub fn buildProposal(
    alloc: std.mem.Allocator,
    focus: []const u8,
    gap: []const u8,
) !*Proposal {
    const p = try alloc.create(Proposal);
    var stats_line: []const u8 = "no trace history yet";
    if (traces.stats(alloc)) |st| {
        stats_line = try std.fmt.allocPrint(alloc,
            "{d} turns · {d} failed · {d} tool calls · median {d} tok/answer",
            .{ st.turns, st.turns_failed, st.tool_calls, st.median_out });
    } else |_| {}
    var steps = std.ArrayList([]const u8).init(alloc);
    try steps.append("analyze trace history + failure patterns (3d space)");
    try steps.append("apply evidence-backed harness edits (prompt/skills/goals/tools)");
    try steps.append("decide if capability gap warrants training; if so build/pull a dataset and launch ETM");
    p.* = .{
        .focus = try alloc.dupe(u8, focus),
        .gap = try alloc.dupe(u8, gap),
        .stats_line = stats_line,
        .steps = steps.items,
    };
    return p;
}

// ---------- the loop ----------

const ANALYZE_SYS =
    \\You are the RSI (recursive self-improvement) analyzer inside an
    \\agentic coding harness. You receive:
    \\  - aggregate telemetry of past turns (tool calls, skills, token
    \\    medians, verified failures)
    \\  - a 3D projection of every turn (x=tool calls, y=tokens out,
    \\    z=skills used, ok=verdict) with failure clusters
    \\  - the failed task + verifier gap (or a manual focus)
    \\
    \\Decide what will actually fix the failure pattern:
    \\  a) harness edits (system guidance, skills, subagents, tool doctrine)
    \\     — for behavioral/process failures
    \\  b) training (ETM fine-tune on a dataset built from good traces and
    \\     optionally a HuggingFace dataset) — for capability failures that
    \\     repeat despite context
    \\
    \\Reply with ONLY a JSON object:
    \\{"diagnosis":"<2-4 sentences, cite trace evidence>",
    \\ "harness_fix":"<refinement instructions, or empty>",
    \\ "training": true|false,
    \\ "dataset_focus":"<what the training data should teach>",
    \\ "hf_queries":["<huggingface dataset search terms, 0-2>"]}
;

const Analysis = struct {
    diagnosis: []const u8,
    harness_fix: []const u8,
    training: bool,
    dataset_focus: []const u8,
    hf_queries: [][]const u8,
};

fn analyze(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    prop: *const Proposal,
) !Analysis {
    var user = std.ArrayList(u8).init(alloc);
    try user.appendSlice("== telemetry ==\n");
    if (traces.summary(alloc)) |s| {
        try user.appendSlice(s);
    } else |_| {
        try user.appendSlice("(no traces yet)");
    }
    try user.appendSlice("\n\n== 3d space ==\n");
    if (traces.spaceJson(alloc)) |s| {
        try user.appendSlice(try util.truncateMiddle(alloc, s, 8000));
    } else |_| {
        try user.appendSlice("(no space data)");
    }
    try user.writer().print("\n\n== focus ==\n{s}\n", .{prop.focus});
    if (prop.gap.len > 0) try user.writer().print("verifier gap: {s}\n", .{prop.gap});

    const msgs = [_]std.json.Value{
        try llm.systemMessage(alloc, ANALYZE_SYS),
        try llm.userMessage(alloc, user.items),
    };
    const res = try llm.chat(hclient, alloc, cfg, &msgs, &.{}, .{
        .max_tokens = 1200,
        .temperature = 0.2,
        .stream = false,
        .no_tools = true,
    });
    const body = res.content;
    const start = std.mem.indexOfScalar(u8, body, '{') orelse return error.AnalyzeBadReply;
    const end = std.mem.lastIndexOfScalar(u8, body, '}') orelse return error.AnalyzeBadReply;
    const v = try util.jparse(alloc, body[start .. end + 1]);

    var queries = std.ArrayList([]const u8).init(alloc);
    if (util.jget(v, "hf_queries")) |q| {
        if (q == .array) {
            for (q.array.items) |item| {
                if (item == .string) try queries.append(item.string);
            }
        }
    }
    return .{
        .diagnosis = util.jstr(v, "diagnosis") orelse "",
        .harness_fix = util.jstr(v, "harness_fix") orelse "",
        .training = util.jbool(v, "training") orelse false,
        .dataset_focus = util.jstr(v, "dataset_focus") orelse "",
        .hf_queries = queries.items,
    };
}

/// Run the full RSI loop synchronously (call on a worker — it blocks).
/// Every phase reports through `sink`; never throws on a bad model reply —
/// failures become readable status lines and the loop continues/ends.
pub fn runLoop(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    h: *harness.Harness,
    cwd: []const u8,
    prop: *const Proposal,
    sink: *const Sink,
) void {
    _ = cwd;
    var report = std.ArrayList(u8).init(alloc);
    const rw = report.writer();

    // ---- phase 1: analyze ----
    tell(sink, "◈ rsi: analyzing trace history + failure space…");
    const a = analyze(alloc, cfg, hclient, prop) catch |e| {
        tellF(sink, alloc, "◈ rsi: analysis failed ({s}) — stopping", .{@errorName(e)});
        return;
    };
    rw.print("**diagnosis** — {s}\n\n", .{a.diagnosis}) catch {};
    tell(sink, "◈ rsi: diagnosis ready");

    // ---- phase 2: harness ----
    if (a.harness_fix.len > 0) {
        tell(sink, "◈ rsi: refining harness…");
        var ctx_msgs = std.ArrayList(std.json.Value).init(alloc);
        ctx_msgs.append(llm.userMessage(alloc, prop.focus) catch return) catch return;
        ctx_msgs.append(llm.systemMessage(alloc,
            std.fmt.allocPrint(alloc,
                "verifier gap: {s}\nrsi diagnosis: {s}", .{ prop.gap, a.diagnosis })
                catch "") catch return) catch return;
        const instr = std.fmt.allocPrint(alloc,
            "RSI loop. Diagnosis: {s}\nRequired fix: {s}", .{ a.diagnosis, a.harness_fix })
            catch a.harness_fix;
        const sum = refine_mod.run(alloc, cfg, hclient, h, ctx_msgs.items, "", .{
            .instructions = instr,
        }) catch |e| std.fmt.allocPrint(alloc, "refine failed: {s}", .{@errorName(e)})
            catch "refine failed";
        rw.print("**harness** — {s}\n\n", .{sum}) catch {};
        tellF(sink, alloc, "◈ rsi: harness updated", .{});
    } else {
        rw.print("**harness** — analysis judged current harness adequate, no edits\n\n", .{}) catch {};
    }

    // ---- phase 3: training ----
    if (a.training) {
        trainState(sink, true, "preparing datasets");
        tell(sink, "◈ rsi: capability gap — building training data…");
        rw.print("**training** — capability gap warrants ETM fine-tune\n", .{}) catch {};

        const ds = train.buildDataset(alloc, "rsi") catch |e| blk: {
            tellF(sink, alloc, "◈ rsi: dataset build failed ({s})", .{@errorName(e)});
            break :blk train.Dataset{ .path = "", .samples = 0 };
        };
        if (ds.samples > 0)
            rw.print("- trace dataset: `{s}` ({d} samples)\n", .{ ds.path, ds.samples }) catch {}
        else
            rw.print("- trace dataset: none (no usable turns yet)\n", .{}) catch {};

        var hf_files = std.ArrayList([]const u8).init(alloc);
        for (a.hf_queries[0..@min(2, a.hf_queries.len)]) |q| {
            tellF(sink, alloc, "◈ rsi: searching huggingface for \"{s}\"…", .{q});
            const hits = train.hfSearch(alloc, hclient, q, 3) catch continue;
            if (hits.len == 0) continue;
            const id = hits[0].id;
            rw.print("- hf search `{s}` → {s} ({d} likes)\n", .{ q, id, hits[0].likes }) catch {};
            if (train.hfDownload(alloc, hclient, id)) |p| {
                rw.print("  downloaded → `{s}`\n", .{p}) catch {};
                hf_files.append(p) catch {};
            } else |_| {
                rw.print("  download failed (no text-format file?)\n", .{}) catch {};
            }
        }

        if (ds.samples > 0 or hf_files.items.len > 0) {
            tell(sink, "◈ rsi: launching ETM training (evict-train-merge)…");
            const focus_args = std.fmt.allocPrint(alloc,
                "--focus \"{s}\"", .{a.dataset_focus}) catch "";
            const data_path = if (ds.samples > 0) ds.path else hf_files.items[0];
            if (train.launchEtm(alloc, data_path, focus_args)) |tp| {
                rw.print("- etm: `{s}` → log `{s}`\n", .{ train.ETM_ENTRY, tp.log_path }) catch {};
                trainState(sink, true, "etm running");
                tell(sink, "◈ model is improving itself — etm running in background");
            } else |e| {
                rw.print("- launch failed: {s} (needs python + cuda env)\n", .{@errorName(e)}) catch {};
                trainState(sink, false, "");
                tellF(sink, alloc, "◈ rsi: training launch failed ({s})", .{@errorName(e)});
            }
        } else {
            rw.print("- skipped: no usable dataset (0 trace samples, no hf file)\n", .{}) catch {};
            trainState(sink, false, "");
        }
    } else {
        trainState(sink, false, "");
    }

    rw.print("\n*rsi loop complete*", .{}) catch {};
    sink.report(sink.ctx, report.items);
}
