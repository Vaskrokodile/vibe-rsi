//! Persistent telemetry store: every turn the agent completes is appended
//! as JSONL to ~/.xera-rsi/traces/all.jsonl — prompt, answer, tool calls,
//! skills loaded, token usage, duration and the self-check verdict.
//!
//! This is the evidence base the RSI loop works on: `stats` aggregates it
//! for the model, `spaceJson` projects it into a 3D point space (tools x
//! tokens x skills) the model can navigate to find failure patterns, and
//! `exportDataset` turns the good traces into a fine-tuning dataset.

const std = @import("std");
const util = @import("util.zig");

const max_field: usize = 24_000; // stored prompt/answer cap

pub const Store = struct {
    alloc: std.mem.Allocator,
    path: []const u8,
    mu: std.Thread.Mutex = .{},

    /// Open (creating) the global trace store. Returns null when the home
    /// dir can't be resolved — tracing must never break a turn.
    pub fn create(alloc: std.mem.Allocator) ?*Store {
        const dir = dirPath(alloc) catch return null;
        std.fs.cwd().makePath(dir) catch return null;
        const path = std.fs.path.join(alloc, &.{ dir, "all.jsonl" }) catch return null;
        const s = alloc.create(Store) catch return null;
        s.* = .{ .alloc = alloc, .path = path };
        return s;
    }

    fn appendLine(self: *Store, line: []const u8) void {
        self.mu.lock();
        defer self.mu.unlock();
        const f = std.fs.cwd().createFile(self.path, .{ .truncate = false }) catch return;
        defer f.close();
        f.seekFromEnd(0) catch return;
        f.writeAll(line) catch return;
        f.writeAll("\n") catch return;
    }

    pub fn turn(
        self: *Store,
        alloc: std.mem.Allocator,
        sid: []const u8,
        n: usize,
        prompt: []const u8,
        answer: []const u8,
        tool_names: []const []const u8,
        skill_names: []const []const u8,
        tok_in: u64,
        tok_out: u64,
        iters: usize,
        stop_reason: []const u8,
        ms: i64,
    ) void {
        var buf = std.ArrayList(u8).init(alloc);
        buf.writer().print(
            "{{\"k\":\"turn\",\"sid\":\"{s}\",\"n\":{d},\"prompt\":\"{s}\",\"answer\":\"{s}\",\"tools\":[",
            .{
                sid,
                n,
                util.jescape(alloc, truncate(prompt)) catch "",
                util.jescape(alloc, truncate(answer)) catch "",
            },
        ) catch return;
        for (tool_names, 0..) |t, i| {
            if (i > 0) buf.append(',') catch return;
            buf.writer().print("\"{s}\"", .{util.jescape(alloc, t) catch ""}) catch return;
        }
        buf.appendSlice("],\"skills\":[") catch return;
        for (skill_names, 0..) |t, i| {
            if (i > 0) buf.append(',') catch return;
            buf.writer().print("\"{s}\"", .{util.jescape(alloc, t) catch ""}) catch return;
        }
        buf.writer().print(
            "],\"tin\":{d},\"tout\":{d},\"iters\":{d},\"stop\":\"{s}\",\"ms\":{d},\"ts\":{d}}}",
            .{ tok_in, tok_out, iters, stop_reason, ms, std.time.timestamp() },
        ) catch return;
        self.appendLine(buf.items);
    }

    pub fn tool(self: *Store, alloc: std.mem.Allocator, sid: []const u8, n: usize, name: []const u8, ok: bool, sum: []const u8) void {
        const line = std.fmt.allocPrint(alloc,
            "{{\"k\":\"tool\",\"sid\":\"{s}\",\"n\":{d},\"name\":\"{s}\",\"ok\":{s},\"sum\":\"{s}\",\"ts\":{d}}}",
            .{
                sid,
                n,
                util.jescape(alloc, name) catch "",
                if (ok) "true" else "false",
                util.jescape(alloc, util.firstLine(sum)) catch "",
                std.time.timestamp(),
            }) catch return;
        self.appendLine(line);
    }

    pub fn skill(self: *Store, alloc: std.mem.Allocator, sid: []const u8, n: usize, name: []const u8) void {
        const line = std.fmt.allocPrint(alloc,
            "{{\"k\":\"skill\",\"sid\":\"{s}\",\"n\":{d},\"name\":\"{s}\",\"ts\":{d}}}",
            .{ sid, n, util.jescape(alloc, name) catch "", std.time.timestamp() }) catch return;
        self.appendLine(line);
    }

    pub fn verdict(self: *Store, alloc: std.mem.Allocator, sid: []const u8, n: usize, ok: bool, gap: []const u8) void {
        const line = std.fmt.allocPrint(alloc,
            "{{\"k\":\"verdict\",\"sid\":\"{s}\",\"n\":{d},\"ok\":{s},\"gap\":\"{s}\",\"ts\":{d}}}",
            .{
                sid,
                n,
                if (ok) "true" else "false",
                util.jescape(alloc, util.firstLine(gap)) catch "",
                std.time.timestamp(),
            }) catch return;
        self.appendLine(line);
    }

    pub fn rsiEvent(self: *Store, alloc: std.mem.Allocator, sid: []const u8, phase: []const u8, detail: []const u8) void {
        const line = std.fmt.allocPrint(alloc,
            "{{\"k\":\"rsi\",\"sid\":\"{s}\",\"phase\":\"{s}\",\"detail\":\"{s}\",\"ts\":{d}}}",
            .{
                sid,
                util.jescape(alloc, phase) catch "",
                util.jescape(alloc, util.firstLine(detail)) catch "",
                std.time.timestamp(),
            }) catch return;
        self.appendLine(line);
    }
};

fn truncate(s: []const u8) []const u8 {
    return if (s.len > max_field) s[0..max_field] else s;
}

pub fn dirPath(alloc: std.mem.Allocator) ![]u8 {
    const home = try util.homeDir(alloc);
    return std.fs.path.join(alloc, &.{ home, ".xera-rsi", "traces" });
}

/// Read the whole trace log (bounded to the last `max_bytes`).
fn readAll(alloc: std.mem.Allocator, max_bytes: usize) ![]u8 {
    const dir = try dirPath(alloc);
    const path = try std.fs.path.join(alloc, &.{ dir, "all.jsonl" });
    const f = std.fs.cwd().openFile(path, .{}) catch return error.NoTraces;
    defer f.close();
    const size = (try f.stat()).size;
    if (size > max_bytes) {
        try f.seekTo(size - max_bytes);
        const tail = try f.readToEndAlloc(alloc, max_bytes);
        // skip to the next newline so we don't parse half a record
        if (std.mem.indexOfScalar(u8, tail, '\n')) |i| return tail[i + 1 ..];
        return tail;
    }
    return f.readToEndAlloc(alloc, max_bytes);
}

pub const Hist = struct { name: []const u8, count: u32 };
pub const Failure = struct { prompt: []const u8, gap: []const u8, ts: i64 };

pub const Stats = struct {
    turns: usize = 0,
    turns_ok: usize = 0,
    turns_failed: usize = 0,
    tool_calls: usize = 0,
    tool_errors: usize = 0,
    top_tools: []Hist = &.{},
    top_skills: []Hist = &.{},
    median_out: u64 = 0,
    median_in: u64 = 0,
    median_iters: u64 = 0,
    failures: []Failure = &.{},
};

fn topN(alloc: std.mem.Allocator, map: *std.StringHashMap(u32), n: usize) ![]Hist {
    var list = std.ArrayList(Hist).init(alloc);
    var it = map.iterator();
    while (it.next()) |kv| try list.append(.{ .name = kv.key_ptr.*, .count = kv.value_ptr.* });
    std.mem.sort(Hist, list.items, {}, struct {
        fn lt(_: void, x: Hist, y: Hist) bool {
            return x.count > y.count;
        }
    }.lt);
    return list.items[0..@min(n, list.items.len)];
}

fn medianOf(v: []u64) u64 {
    if (v.len == 0) return 0;
    std.mem.sort(u64, v, {}, std.sort.asc(u64));
    return v[v.len / 2];
}

/// Aggregate the whole trace log into Stats. Tolerates malformed lines.
pub fn stats(alloc: std.mem.Allocator) !Stats {
    const body = try readAll(alloc, 32 << 20);
    var st = Stats{};
    var tools = std.StringHashMap(u32).init(alloc);
    var skills = std.StringHashMap(u32).init(alloc);
    var outs = std.ArrayList(u64).init(alloc);
    var ins = std.ArrayList(u64).init(alloc);
    var iters_l = std.ArrayList(u64).init(alloc);
    var fails = std.ArrayList(Failure).init(alloc);
    var last_prompt = std.StringHashMap([]const u8).init(alloc); // sid|n -> prompt

    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line| {
        if (line.len < 2) continue;
        const v = util.jparse(alloc, line) catch continue;
        const k = util.jstr(v, "k") orelse continue;
        const sid = util.jstr(v, "sid") orelse "";
        const n = util.jint(v, "n") orelse 0;
        if (std.mem.eql(u8, k, "turn")) {
            st.turns += 1;
            const key = try std.fmt.allocPrint(alloc, "{s}|{d}", .{ sid, n });
            try last_prompt.put(key, util.jstr(v, "prompt") orelse "");
            const tout = util.jint(v, "tout") orelse 0;
            const tin = util.jint(v, "tin") orelse 0;
            if (tout > 0) try outs.append(@intCast(tout));
            if (tin > 0) try ins.append(@intCast(tin));
            const its = util.jint(v, "iters") orelse 0;
            if (its > 0) try iters_l.append(@intCast(its));
        } else if (std.mem.eql(u8, k, "tool")) {
            st.tool_calls += 1;
            const name = util.jstr(v, "name") orelse "?";
            const gop = try tools.getOrPut(name);
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += 1;
            if (util.jbool(v, "ok") == false) st.tool_errors += 1;
        } else if (std.mem.eql(u8, k, "skill")) {
            const name = util.jstr(v, "name") orelse "?";
            const gop = try skills.getOrPut(name);
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += 1;
        } else if (std.mem.eql(u8, k, "verdict")) {
            const ok = util.jbool(v, "ok") orelse true;
            if (ok) {
                st.turns_ok += 1;
            } else {
                st.turns_failed += 1;
                const key = try std.fmt.allocPrint(alloc, "{s}|{d}", .{ sid, n });
                try fails.append(.{
                    .prompt = last_prompt.get(key) orelse "",
                    .gap = util.jstr(v, "gap") orelse "",
                    .ts = util.jint(v, "ts") orelse 0,
                });
            }
        }
    }
    st.top_tools = try topN(alloc, &tools, 12);
    st.top_skills = try topN(alloc, &skills, 12);
    st.median_out = medianOf(outs.items);
    st.median_in = medianOf(ins.items);
    st.median_iters = medianOf(iters_l.items);
    st.failures = fails.items[if (fails.items.len > 8) fails.items.len - 8 else 0..];
    return st;
}

/// Compact stats block for model consumption (RSI analysis + traces tool).
pub fn summary(alloc: std.mem.Allocator) ![]const u8 {
    const st = try stats(alloc);
    var b = std.ArrayList(u8).init(alloc);
    try b.writer().print(
        "traces: {d} turns ({d} verified ok, {d} failed) · {d} tool calls ({d} errors)\n" ++
            "tokens/answer: median in={d} out={d} · median iterations={d}\n",
        .{ st.turns, st.turns_ok, st.turns_failed, st.tool_calls, st.tool_errors,
           st.median_in, st.median_out, st.median_iters });
    try b.appendSlice("top tools:");
    for (st.top_tools) |t| try b.writer().print(" {s}x{d}", .{ t.name, t.count });
    try b.appendSlice("\ntop skills:");
    if (st.top_skills.len == 0) try b.appendSlice(" (none)");
    for (st.top_skills) |t| try b.writer().print(" {s}x{d}", .{ t.name, t.count });
    try b.appendSlice("\nrecent failures:");
    if (st.failures.len == 0) try b.appendSlice(" (none)");
    for (st.failures) |f| {
        const p = if (f.prompt.len > 120) f.prompt[0..120] else f.prompt;
        try b.writer().print("\n  - \"{s}\" → {s}", .{ p, f.gap });
    }
    return b.items;
}

/// Project every turn into a 3D point space the model can navigate:
/// x = tool calls, y = tokens out, z = skills used; colour = verdict.
/// Emitted as JSON — both the `traces` tool and the livesite read it.
pub fn spaceJson(alloc: std.mem.Allocator) ![]u8 {
    const body = try readAll(alloc, 32 << 20);
    var points = std.ArrayList(u8).init(alloc);
    var verdicts = std.StringHashMap(bool).init(alloc);
    var fails = std.StringHashMap(void).init(alloc);
    var tool_per_turn = std.StringHashMap(u32).init(alloc);

    // first pass: verdicts + per-turn tool counts
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line| {
        if (line.len < 2) continue;
        const v = util.jparse(alloc, line) catch continue;
        const k = util.jstr(v, "k") orelse continue;
        const sid = util.jstr(v, "sid") orelse "";
        const n = util.jint(v, "n") orelse 0;
        const key = try std.fmt.allocPrint(alloc, "{s}|{d}", .{ sid, n });
        if (std.mem.eql(u8, k, "verdict")) {
            const ok = util.jbool(v, "ok") orelse true;
            try verdicts.put(key, ok);
            if (!ok) try fails.put(key, {});
        } else if (std.mem.eql(u8, k, "tool")) {
            const gop = try tool_per_turn.getOrPut(key);
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += 1;
        }
    }

    var npts: usize = 0;
    var clusters = std.StringHashMap(struct { n: u32, fails: u32 }).init(alloc);
    try points.append('[');
    var it2 = std.mem.splitScalar(u8, body, '\n');
    while (it2.next()) |line| {
        if (line.len < 2) continue;
        const v = util.jparse(alloc, line) catch continue;
        if (!std.mem.eql(u8, util.jstr(v, "k") orelse "", "turn")) continue;
        const sid = util.jstr(v, "sid") orelse "";
        const n = util.jint(v, "n") orelse 0;
        const key = try std.fmt.allocPrint(alloc, "{s}|{d}", .{ sid, n });
        const x: u32 = tool_per_turn.get(key) orelse 0;
        const y: i64 = util.jint(v, "tout") orelse 0;
        var z: u32 = 0;
        if (util.jget(v, "skills")) |sk| {
            if (sk == .array) z = @intCast(sk.array.items.len);
        }
        const ok = verdicts.get(key) orelse true;
        const title = util.jstr(v, "prompt") orelse "";
        const short = if (title.len > 80) title[0..80] else title;
        if (npts > 0) try points.append(',');
        try points.writer().print(
            "{{\"x\":{d},\"y\":{d},\"z\":{d},\"ok\":{d},\"sid\":\"{s}\",\"n\":{d},\"t\":\"{s}\"}}",
            .{ x, y, z, @intFromBool(ok), sid, n, util.jescape(alloc, short) catch "" });
        npts += 1;
        // cluster bucket: coarse tool/skill cell
        const ckey = try std.fmt.allocPrint(alloc, "t{d}-s{d}", .{ @min(x, 10), @min(z, 5) });
        const gop = try clusters.getOrPut(ckey);
        if (!gop.found_existing) gop.value_ptr.* = .{ .n = 0, .fails = 0 };
        gop.value_ptr.*.n += 1;
        if (!ok) gop.value_ptr.*.fails += 1;
    }
    try points.appendSlice("],\"clusters\":{");
    var first = true;
    var cit = clusters.iterator();
    while (cit.next()) |kv| {
        if (!first) try points.append(',');
        first = false;
        try points.writer().print("\"{s}\":{{\"n\":{d},\"fails\":{d}}}",
            .{ kv.key_ptr.*, kv.value_ptr.n, kv.value_ptr.fails });
    }
    var out = std.ArrayList(u8).init(alloc);
    try out.writer().print("{{\"points\":{s}}}", .{points.items});
    return out.items;
}

/// Export traced turns as an SFT dataset (one JSON object per line):
/// {"prompt","completion","tools","skills","sid","n"} for every turn that
/// didn't fail verification. Returns the sample count.
pub fn exportDataset(alloc: std.mem.Allocator, out_path: []const u8, only_ok: bool) !usize {
    const body = try readAll(alloc, 32 << 20);
    var fails = std.StringHashMap(void).init(alloc);
    if (only_ok) {
        var it = std.mem.splitScalar(u8, body, '\n');
        while (it.next()) |line| {
            if (line.len < 2) continue;
            const v = util.jparse(alloc, line) catch continue;
            if (!std.mem.eql(u8, util.jstr(v, "k") orelse "", "verdict")) continue;
            if (util.jbool(v, "ok") orelse true) continue;
            const key = try std.fmt.allocPrint(alloc, "{s}|{d}",
                .{ util.jstr(v, "sid") orelse "", util.jint(v, "n") orelse 0 });
            try fails.put(key, {});
        }
    }
    if (std.fs.path.dirname(out_path)) |d| try std.fs.cwd().makePath(d);
    const f = try std.fs.cwd().createFile(out_path, .{});
    defer f.close();
    var n_written: usize = 0;
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line| {
        if (line.len < 2) continue;
        const v = util.jparse(alloc, line) catch continue;
        if (!std.mem.eql(u8, util.jstr(v, "k") orelse "", "turn")) continue;
        const sid = util.jstr(v, "sid") orelse "";
        const n = util.jint(v, "n") orelse 0;
        const key = try std.fmt.allocPrint(alloc, "{s}|{d}", .{ sid, n });
        if (fails.contains(key)) continue;
        const prompt = util.jstr(v, "prompt") orelse continue;
        const answer = util.jstr(v, "answer") orelse continue;
        if (prompt.len == 0 or answer.len == 0) continue;
        var rec = std.ArrayList(u8).init(alloc);
        try rec.writer().print(
            "{{\"prompt\":\"{s}\",\"completion\":\"{s}\",\"sid\":\"{s}\",\"n\":{d}}}",
            .{
                util.jescape(alloc, prompt) catch continue,
                util.jescape(alloc, answer) catch continue,
                sid,
                n,
            });
        try f.writeAll(rec.items);
        try f.writeAll("\n");
        n_written += 1;
    }
    return n_written;
}
