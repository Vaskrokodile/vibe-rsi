//! Continual-harness refinement — surgical, evidence-backed edits to
//! the editable harness instead of wholesale rewrites.
//!
//! Ported concept from prime-agent's /refine: a separate model pass
//! reviews the trajectory and emits *precise* edits —
//!   {action: append|replace|delete, target, title, content, reason}
//! — which we validate, apply file-by-file, and record in
//! `harness/refinement_history.jsonl`. A `snapshots/<ts>/` copy of the
//! pre-edit harness is written first, so every refinement is undoable.
//!
//! Scope: `local` (workspace `.xera-rsi/harness`) by default —
//! lessons only this project needs. `global` targets
//! `~/.xera-rsi/harness` for cross-project lessons.
//!
//! Triggers: the `refine` tool (runs at turn end), and
//! `xera-rsi refine <instructions>` (refines the latest session).

const std = @import("std");
const util = @import("util.zig");
const config = @import("config.zig");
const http = @import("http.zig");
const llm = @import("llm.zig");
const harness = @import("harness.zig");

pub const Options = struct {
    instructions: []const u8 = "",
    scope_global: bool = false,
};

const SYSTEM =
    \\You are xera-rsi's refinement subsystem. Improve the editable harness
    \\files from the trajectory below — like compaction, but instead of
    \\summarizing you emit precise, evidence-backed edits to reusable state.
    \\
    \\Emit ONLY a JSON object:
    \\{"summary":"<one line>","edits":[
    \\ {"action":"append|replace","target":"<file>","title":"<short>","content":"<markdown>","reason":"<evidence from the trajectory>"}
    \\]}
    \\
    \\Rules:
    \\- targets (pick the narrowest that fits):
    \\    "memory"           → memory.md: durable facts, decisions, preferences, outcomes
    \\    "lessons"          → lessons.md: reusable tactics that worked/failed
    \\    "guardrails"       → guardrails.md: hard constraints the agent violated or should never violate
    \\    "tool_guidance"    → tool_guidance.md: better tool-usage doctrine
    \\    "skill:<name>"     → skills/<name>.md: a reusable workflow (create or replace the whole skill file)
    \\    "subagent:<name>"  → sub_agents/<name>/AGENT.md: a reusable delegation persona
    \\- append adds a `## <title>` section to the file; replace overwrites it.
    \\- Prefer few, high-quality edits (≤5). Skip transient noise, one-off
    \\  failures, and anything already covered by the current harness.
    \\- Every edit needs a reason citing concrete trajectory evidence.
    \\- Never edit system_prompt.md — the base prompt is immutable.
    \\- {"summary":"...","edits":[]} is a valid outcome when nothing is durable.
;

/// Run one refinement pass. Returns a compact summary for the caller
/// (injected into history / printed by the CLI). Never throws on a bad
/// model reply — returns an "error: ..." string instead.
pub fn run(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    h_local: *harness.Harness,
    messages: []const std.json.Value,
    trajectory: []const u8,
    opts: Options,
) ![]const u8 {
    _ = trajectory;
    var hg = h_local;
    var h_global_storage: harness.Harness = undefined;
    if (opts.scope_global) {
        const home = try util.homeDir(alloc);
        const dir = try std.fs.path.join(alloc, &.{ home, ".xera-rsi", "harness" });
        h_global_storage = .{ .alloc = alloc, .dir = dir, .root = h_local.root, .global = true };
        hg = &h_global_storage;
    }
    try hg.ensure();

    // ---- trajectory digest: tail of the conversation, bounded ----
    var digest = std.ArrayList(u8).init(alloc);
    var shown: usize = 0;
    var i: usize = messages.len;
    while (i > 0 and shown < 24 and digest.items.len < 24_000) {
        i -= 1;
        const m = messages[i];
        const role = util.jstr(m, "role") orelse continue;
        if (std.mem.eql(u8, role, "system")) continue;
        const content = util.jstr(m, "content") orelse continue;
        const slice = if (content.len > 800) content[0..800] else content;
        var line = std.ArrayList(u8).init(alloc);
        try line.writer().print("{s}: {s}\n", .{ role, slice });
        try digest.insertSlice(0, line.items);
        shown += 1;
    }

    var files = std.ArrayList(u8).init(alloc);
    for ([_][]const u8{ "memory.md", "lessons.md", "guardrails.md", "tool_guidance.md" }) |f| {
        const body = hg.readAny(&.{f});
        try files.writer().print("{s}: {d}B\n", .{ f, body.len });
    }

    const user = try std.fmt.allocPrint(alloc,
        "## instructions\n{s}\n\n## harness files (size)\n{s}\n## trajectory\n{s}",
        .{ if (opts.instructions.len > 0) opts.instructions else "(none — refine what the trajectory shows)",
           files.items, digest.items });

    const msgs = [_]std.json.Value{
        try llm.systemMessage(alloc, SYSTEM),
        try llm.userMessage(alloc, user),
    };
    const res = try llm.chat(hclient, alloc, cfg, &msgs, &[_]llm.ToolSchema{}, .{
        .max_tokens = 8192,
        .stream = false,
        .no_tools = true,
    });

    const root = util.jparse(alloc, extractJson(res.content)) catch
        return std.fmt.allocPrint(alloc, "error: refiner returned no usable JSON", .{});

    const summary = util.jstr(root, "summary") orelse "(no summary)";

    const edits_v = util.jget(root, "edits") orelse
        return std.fmt.allocPrint(alloc, "refined: {s} (0 edits)", .{summary});
    if (edits_v != .array or edits_v.array.items.len == 0)
        return std.fmt.allocPrint(alloc, "refined: {s} (0 edits)", .{summary});

    // ---- rollback point ----
    const snap = try hg.snapshot();
    const ts = std.time.milliTimestamp();
    try writeSnapshot(alloc, hg.dir, ts, snap);

    // ---- validate + apply ----
    var applied = std.ArrayList(u8).init(alloc);
    var n: usize = 0;
    for (edits_v.array.items) |e| {
        if (n >= 12) break;
        if (e != .object) continue;
        const action = util.jstr(e, "action") orelse "append";
        const target = util.jstr(e, "target") orelse continue;
        const content = util.jstr(e, "content") orelse continue;
        const title = util.jstr(e, "title") orelse "untitled";
        const reason = util.jstr(e, "reason") orelse "";
        if (content.len == 0 or content.len > 16_000) continue;

        const rel = targetPath(alloc, target) orelse continue;
        const body = if (std.mem.eql(u8, action, "append"))
            try std.fmt.allocPrint(alloc, "## {s}\n\n{s}\n\n_(reason: {s})_", .{ title, content, reason })
        else
            content;
        const op: []const u8 = if (std.mem.eql(u8, action, "append")) "append" else "write";
        hg.applyEdit(rel, op, body) catch continue;
        try applied.writer().print("{s}:{s}:{s};", .{ action, rel, title });
        n += 1;
    }

    // ---- history ----
    try appendHistory(alloc, hg.dir, .{
        .ts = ts,
        .scope = if (opts.scope_global) "global" else "local",
        .instructions = opts.instructions,
        .summary = summary,
        .edits = applied.items,
    });

    return std.fmt.allocPrint(alloc, "refined: {s} — {d} edit(s): {s}", .{ summary, n, applied.items });
}

/// Map a refiner target name to a harness-relative file path, or null
/// when the target is invalid/unsafe.
fn targetPath(alloc: std.mem.Allocator, target: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, target, "memory")) return alloc.dupe(u8, "memory.md") catch null;
    if (std.mem.eql(u8, target, "lessons")) return alloc.dupe(u8, "lessons.md") catch null;
    if (std.mem.eql(u8, target, "guardrails")) return alloc.dupe(u8, "guardrails.md") catch null;
    if (std.mem.eql(u8, target, "tool_guidance")) return alloc.dupe(u8, "tool_guidance.md") catch null;
    if (std.mem.startsWith(u8, target, "skill:")) {
        const name = sanitize(target[6..]) orelse return null;
        return std.fmt.allocPrint(alloc, "skills/{s}.md", .{name}) catch null;
    }
    if (std.mem.startsWith(u8, target, "subagent:")) {
        const name = sanitize(target[9..]) orelse return null;
        return std.fmt.allocPrint(alloc, "sub_agents/{s}/AGENT.md", .{name}) catch null;
    }
    return null;
}

fn sanitize(name: []const u8) ?[]const u8 {
    if (name.len == 0 or name.len > 40) return null;
    for (name) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '-' and c != '_') return null;
    }
    return name;
}

/// Pull the first balanced {...} object out of a model reply (direct,
/// fenced, or embedded in prose).
fn extractJson(text: []const u8) []const u8 {
    const t = std.mem.trim(u8, text, " \n\t");
    if (t.len > 0 and t[0] == '{' and t[t.len - 1] == '}') return t;
    if (std.mem.indexOf(u8, t, "```")) |fs| {
        var rest = t[fs + 3 ..];
        if (std.mem.startsWith(u8, rest, "json")) rest = rest[4..];
        rest = std.mem.trimLeft(u8, rest, " \n");
        if (std.mem.indexOf(u8, rest, "```")) |fe| {
            const cand = std.mem.trim(u8, rest[0..fe], " \n\t");
            if (cand.len > 0 and cand[0] == '{') return cand;
        }
    }
    const s = std.mem.indexOfScalar(u8, t, '{') orelse return t;
    const e = std.mem.lastIndexOfScalar(u8, t, '}') orelse return t;
    if (e > s) return t[s .. e + 1];
    return t;
}

/// Write a pre-edit harness snapshot to <dir>/snapshots/<ts>/ for rollback.
/// Also used by the `harness` tool before direct edits.
pub fn writeSnapshot(
    alloc: std.mem.Allocator,
    dir: []const u8,
    ts: i64,
    snap: std.StringArrayHashMap([]const u8),
) !void {
    const snap_dir = try std.fmt.allocPrint(alloc, "{s}/snapshots/{d}", .{ dir, ts });
    var it = snap.iterator();
    while (it.next()) |kv| {
        const p = try std.fs.path.join(alloc, &.{ snap_dir, kv.key_ptr.* });
        if (std.fs.path.dirname(p)) |parent| try std.fs.cwd().makePath(parent);
        const f = try std.fs.cwd().createFile(p, .{});
        defer f.close();
        try f.writeAll(kv.value_ptr.*);
    }
}

pub const HistEntry = struct {
    ts: i64,
    scope: []const u8,
    instructions: []const u8,
    summary: []const u8,
    edits: []const u8,
};

pub fn appendHistory(alloc: std.mem.Allocator, dir: []const u8, e: HistEntry) !void {
    const p = try std.fs.path.join(alloc, &.{ dir, "refinement_history.jsonl" });
    const f = try std.fs.cwd().createFile(p, .{ .truncate = false });
    defer f.close();
    try f.seekFromEnd(0);
    try f.writer().print(
        "{{\"ts\":{d},\"scope\":\"{s}\",\"instructions\":\"{s}\",\"summary\":\"{s}\",\"edits\":\"{s}\"}}\n",
        .{ e.ts, e.scope,
           try util.jescape(alloc, e.instructions),
           try util.jescape(alloc, e.summary),
           try util.jescape(alloc, e.edits) });
}

test "extractJson handles prose and fences" {
    try std.testing.expectEqualStrings("{\"a\":1}", extractJson("  {\"a\":1}  "));
    try std.testing.expectEqualStrings("{\"a\":1}", extractJson("sure:\n```json\n{\"a\":1}\n```\n"));
    try std.testing.expectEqualStrings("{\"a\":1}", extractJson("noise {\"a\":1} tail"));
}

test "targetPath validation" {
    const alloc = std.heap.page_allocator;
    try std.testing.expectEqualStrings("memory.md", targetPath(alloc, "memory").?);
    try std.testing.expectEqualStrings("skills/deploy.md", targetPath(alloc, "skill:deploy").?);
    try std.testing.expectEqualStrings("sub_agents/rev/AGENT.md", targetPath(alloc, "subagent:rev").?);
    try std.testing.expect(targetPath(alloc, "system_prompt") == null);
    try std.testing.expect(targetPath(alloc, "skill:../etc") == null);
}
