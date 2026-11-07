//! Inter-agent coordination fabric for the agent session.
//!
//! One Coordinator lives per session and is shared by every agent the app
//! runs: the main agent and `task` subagents. It is deliberately *not* a chat system — agents exchange
//! compact, structured signals so overlap is prevented without burning
//! tokens on prose:
//!
//!   - claims: workspace-relative glob patterns an agent intends to write.
//!     Enforcement is hard: the write tools refuse paths claimed by
//!     another agent, so overlap is prevented even if a model never
//!     calls the coord tool.
//!   - feed: bounded ring of pipe-delimited events "seq|kind|from|data".
//!     Successful writes emit `w` events automatically — awareness is free.
//!   - inbox: directed agent→agent lines; "*" broadcasts to the feed.
//!   - presence: every registered agent has a short status note.
//!
//! Delivery is push-style: the agent loop drains new feed lines + inbox
//! before each model call and injects them as a <sync> system message,
//! so nothing depends on the model remembering to poll.
//!
//! Claims persist until explicitly released (coord rel) or CLAIM_TTL_MS
//! expiry — they must outlive a single turn since channel agents take
//! interleaved turns on the shared workspace. A subagent shares its
//! parent's lineage ("main/s" inherits "main"), so delegated claims
//! never block the delegate. Everything runs under one mutex.

const std = @import("std");
const util = @import("util.zig");

pub const MAX_FEED: usize = 512;
pub const MAX_INBOX: usize = 64;
pub const MAX_DATA: usize = 160;
pub const MAX_DRAIN_LINES: usize = 48;
pub const CLAIM_TTL_MS: i64 = 10 * 60 * 1000;

const Claim = struct {
    aid: []const u8,
    pat: []const u8, // workspace-relative glob, '/'-normalized
    ts: i64,
};

const Member = struct {
    note: []const u8 = "",
    seen: i64 = 0,
    cursor: u64 = 0,
};

/// A persistent objective an agent pursues across turns until it calls
/// `goal complete` — prime-agent's goal mode, mirrored per-agent so
/// `status` shows what everyone is ultimately driving at.
const Goal = struct {
    objective: []const u8 = "",
    set_ms: i64 = 0,
};

/// Strip protocol-hostile bytes from a payload and cap its length.
fn cleanData(alloc: std.mem.Allocator, s: []const u8) []const u8 {
    const n = @min(s.len, MAX_DATA);
    const d = alloc.dupe(u8, s[0..n]) catch return "";
    for (d) |*c| {
        if (c.* == '|' or c.* == '\n' or c.* == '\r') c.* = ' ';
    }
    return d;
}

fn hasGlob(p: []const u8) bool {
    return std.mem.indexOfAny(u8, p, "*?") != null;
}

/// Are two agent ids on one delegation chain? "main" and "main/s" are
/// the same worker — claims along a lineage must never self-block.
fn sameLineage(a: []const u8, b: []const u8) bool {
    if (std.mem.eql(u8, a, b)) return true;
    if (a.len < b.len and std.mem.startsWith(u8, b, a) and b[a.len] == '/') return true;
    if (b.len < a.len and std.mem.startsWith(u8, a, b) and a[b.len] == '/') return true;
    return false;
}

/// Does claim pattern `pat` cover concrete workspace-relative `rel`?
/// A literal claim also covers its subtree ("src" claims "src/a.zig").
pub fn patCovers(pat: []const u8, rel: []const u8) bool {
    if (util.globMatch(pat, rel)) return true;
    if (!hasGlob(pat) and rel.len > pat.len and
        std.mem.startsWith(u8, rel, pat) and rel[pat.len] == '/') return true;
    return false;
}

/// Static overlap check between two claim patterns. Conservative: when we
/// can't prove overlap (glob-vs-glob) we allow the claim — the concrete
/// write path is still hard-gated by conflict().
fn overlapPats(a: []const u8, b: []const u8) bool {
    if (std.mem.eql(u8, a, b)) return true;
    if (!hasGlob(a) and patCovers(b, a)) return true;
    if (!hasGlob(b) and patCovers(a, b)) return true;
    return false;
}

/// Feed/inbox line format is "seq|kind|from|data"; field extractor for it.
fn lineField(line: []const u8, comptime idx: usize) ?[]const u8 {
    var it = std.mem.splitScalar(u8, line, '|');
    var i: usize = 0;
    while (it.next()) |f| : (i += 1) {
        if (i == idx) return f;
    }
    return null;
}

fn lineSeq(line: []const u8) ?u64 {
    const f = lineField(line, 0) orelse return null;
    return std.fmt.parseInt(u64, f, 10) catch null;
}

pub const Coordinator = struct {
    alloc: std.mem.Allocator,
    mu: std.Thread.Mutex = .{},
    members: std.StringHashMap(*Member),
    claims: std.ArrayList(Claim),
    feed: std.ArrayList([]const u8),
    inbox: std.StringHashMap(*std.ArrayList([]const u8)),
    seq: u64 = 0,
    /// active goals keyed by aid (absent key = no goal)
    goals: std.StringHashMap(Goal),

    pub fn init(alloc: std.mem.Allocator) Coordinator {
        return .{
            .alloc = alloc,
            .members = std.StringHashMap(*Member).init(alloc),
            .claims = std.ArrayList(Claim).init(alloc),
            .feed = std.ArrayList([]const u8).init(alloc),
            .inbox = std.StringHashMap(*std.ArrayList([]const u8)).init(alloc),
            .goals = std.StringHashMap(Goal).init(alloc),
        };
    }

    pub fn count(self: *Coordinator) usize {
        self.mu.lock();
        defer self.mu.unlock();
        return self.members.count();
    }

    fn memberLocked(self: *Coordinator, aid: []const u8) !*Member {
        if (self.members.get(aid)) |m| return m;
        const m = try self.alloc.create(Member);
        // new members start at the head of the feed — no history flood
        m.* = .{ .cursor = self.seq };
        try self.members.put(try self.alloc.dupe(u8, aid), m);
        return m;
    }

    /// Register an agent; emits a "+" join event the first time it's seen.
    pub fn register(self: *Coordinator, aid: []const u8) void {
        self.mu.lock();
        defer self.mu.unlock();
        const is_new = !self.members.contains(aid);
        const m = self.memberLocked(aid) catch return;
        m.seen = std.time.milliTimestamp();
        if (is_new) self.emitLocked("+", aid, "");
    }

    /// Set or clear (objective == null) an agent's persistent goal.
    pub fn setGoal(self: *Coordinator, aid: []const u8, objective: ?[]const u8) void {
        self.mu.lock();
        defer self.mu.unlock();
        if (objective) |obj| {
            self.goals.put(self.alloc.dupe(u8, aid) catch return, .{
                .objective = cleanData(self.alloc, obj),
                .set_ms = std.time.milliTimestamp(),
            }) catch return;
            self.emitLocked("g", aid, obj);
        } else {
            if (self.goals.fetchRemove(aid) != null) self.emitLocked("G", aid, "");
        }
    }

    /// Short "what I'm doing" line, surfaced by `status`.
    pub fn setNote(self: *Coordinator, aid: []const u8, note: []const u8) void {
        self.mu.lock();
        defer self.mu.unlock();
        const m = self.memberLocked(aid) catch return;
        m.seen = std.time.milliTimestamp();
        m.note = cleanData(self.alloc, note);
        self.emitLocked("n", aid, m.note);
    }

    /// Drop claims older than CLAIM_TTL_MS (crash/interrupt backstop).
    fn reapLocked(self: *Coordinator, now: i64) void {
        var i: usize = 0;
        while (i < self.claims.items.len) {
            if (now - self.claims.items[i].ts > CLAIM_TTL_MS) {
                _ = self.claims.orderedRemove(i);
            } else i += 1;
        }
    }

    /// Claim a normalized workspace-relative pattern for `aid`. Returns
    /// the conflicting holder's aid when overlap is statically provable —
    /// otherwise null. Unprovable overlap is still enforced concretely by
    /// conflict() at write time, so this can never under-guard.
    pub fn claim(self: *Coordinator, aid: []const u8, pat: []const u8) ?[]const u8 {
        self.mu.lock();
        defer self.mu.unlock();
        const now = std.time.milliTimestamp();
        self.reapLocked(now);
        const m = self.memberLocked(aid) catch return null;
        m.seen = now;

        for (self.claims.items) |*c| {
            if (std.mem.eql(u8, c.aid, aid) and std.mem.eql(u8, c.pat, pat)) {
                c.ts = now;
                return null; // already mine — refresh
            }
        }
        for (self.claims.items) |c| {
            if (sameLineage(c.aid, aid)) continue;
            if (overlapPats(c.pat, pat)) {
                self.emitLocked("!", aid, pat); // contention observed
                return c.aid;
            }
        }
        self.claims.append(.{
            .aid = self.alloc.dupe(u8, aid) catch return null,
            .pat = self.alloc.dupe(u8, pat) catch return null,
            .ts = now,
        }) catch return null;
        self.emitLocked("c", aid, pat);
        return null;
    }

    /// Release one claim (exact pattern) or all of mine (pat == null).
    /// Only my own aid's claims — a subagent can't release its parent's.
    pub fn release(self: *Coordinator, aid: []const u8, pat: ?[]const u8) usize {
        self.mu.lock();
        defer self.mu.unlock();
        var n: usize = 0;
        var i: usize = 0;
        while (i < self.claims.items.len) {
            const c = self.claims.items[i];
            if (std.mem.eql(u8, c.aid, aid) and
                (pat == null or std.mem.eql(u8, c.pat, pat.?)))
            {
                _ = self.claims.orderedRemove(i);
                n += 1;
            } else i += 1;
        }
        if (n > 0) self.emitLocked("r", aid, pat orelse "*");
        return n;
    }

    /// The enforcement core: is concrete `rel` (workspace-relative,
    /// '/'-separated) claimed by an agent other than `aid`?
    pub fn conflict(self: *Coordinator, aid: []const u8, rel: []const u8) ?[]const u8 {
        self.mu.lock();
        defer self.mu.unlock();
        self.reapLocked(std.time.milliTimestamp());
        for (self.claims.items) |c| {
            if (sameLineage(c.aid, aid)) continue;
            if (patCovers(c.pat, rel)) return c.aid;
        }
        return null;
    }

    /// Direct message to an agent's inbox; "*" broadcasts on the feed.
    pub fn send(self: *Coordinator, from: []const u8, to: []const u8, data: []const u8) !void {
        self.mu.lock();
        defer self.mu.unlock();
        const m = self.memberLocked(from) catch return error.OutOfMemory;
        m.seen = std.time.milliTimestamp();
        const d = cleanData(self.alloc, data);
        if (std.mem.eql(u8, to, "*")) {
            self.emitLocked("m", from, d);
            return;
        }
        if (!self.members.contains(to)) return error.UnknownAgent;
        const q = self.inbox.get(to) orelse blk: {
            const nq = try self.alloc.create(std.ArrayList([]const u8));
            nq.* = std.ArrayList([]const u8).init(self.alloc);
            try self.inbox.put(try self.alloc.dupe(u8, to), nq);
            break :blk nq;
        };
        if (q.items.len >= MAX_INBOX) _ = q.orderedRemove(0);
        self.seq += 1;
        try q.append(try std.fmt.allocPrint(self.alloc, "{d}|m|{s}|{s}", .{ self.seq, from, d }));
    }

    /// Append a feed event "seq|kind|from|data".
    pub fn emit(self: *Coordinator, kind: []const u8, from: []const u8, data: []const u8) void {
        self.mu.lock();
        defer self.mu.unlock();
        self.emitLocked(kind, from, data);
    }

    fn emitLocked(self: *Coordinator, kind: []const u8, from: []const u8, data: []const u8) void {
        self.seq += 1;
        const line = std.fmt.allocPrint(self.alloc, "{d}|{s}|{s}|{s}", .{
            self.seq, kind, from, cleanData(self.alloc, data),
        }) catch return;
        if (self.feed.items.len >= MAX_FEED) _ = self.feed.orderedRemove(0);
        self.feed.append(line) catch {};
    }

    /// Everything `aid` hasn't seen: inbox lines + feed lines with
    /// seq > cursor (own events skipped). Returns "" when there's nothing.
    pub fn drain(self: *Coordinator, alloc: std.mem.Allocator, aid: []const u8) ![]const u8 {
        self.mu.lock();
        defer self.mu.unlock();
        const m = try self.memberLocked(aid);
        m.seen = std.time.milliTimestamp();
        var out = std.ArrayList(u8).init(alloc);
        var n: usize = 0;
        var skipped: usize = 0;
        if (self.inbox.get(aid)) |q| {
            for (q.items) |line| {
                if (n >= MAX_DRAIN_LINES) {
                    skipped += 1;
                    continue;
                }
                try out.appendSlice(line);
                try out.append('\n');
                n += 1;
            }
            q.clearRetainingCapacity();
        }
        for (self.feed.items) |line| {
            const s = lineSeq(line) orelse continue;
            if (s <= m.cursor) continue;
            if (lineField(line, 2)) |from| {
                if (std.mem.eql(u8, from, aid)) continue;
            }
            if (n >= MAX_DRAIN_LINES) {
                skipped += 1;
                continue;
            }
            try out.appendSlice(line);
            try out.append('\n');
            n += 1;
        }
        m.cursor = self.seq;
        if (skipped > 0) try out.writer().print("…+{d} more (coord poll/status)", .{skipped});
        return out.items;
    }

    /// One-line compact roster: `aid="note" [claims] · aid2 …`
    /// (`*` marks the caller). Returns "" when fewer than 2 agents exist.
    pub fn status(self: *Coordinator, alloc: std.mem.Allocator, me: []const u8) ![]const u8 {
        self.mu.lock();
        defer self.mu.unlock();
        self.reapLocked(std.time.milliTimestamp());
        if (self.members.count() < 2) return "";
        var out = std.ArrayList(u8).init(alloc);
        try out.appendSlice("agents: ");
        var first = true;
        var it = self.members.iterator();
        while (it.next()) |kv| {
            const aid = kv.key_ptr.*;
            const m = kv.value_ptr.*;
            if (!first) try out.appendSlice(" · ");
            first = false;
            try out.appendSlice(aid);
            if (std.mem.eql(u8, aid, me)) try out.append('*');
            if (m.note.len > 0) try out.writer().print("=\"{s}\"", .{m.note});
            if (self.goals.get(aid)) |g| {
                if (g.objective.len > 0) try out.writer().print(" ⭕{s}", .{g.objective});
            }
            var ci: usize = 0;
            for (self.claims.items) |c| {
                if (!std.mem.eql(u8, c.aid, aid)) continue;
                try out.appendSlice(if (ci == 0) " [" else ",");
                try out.appendSlice(c.pat);
                ci += 1;
            }
            if (ci > 0) try out.append(']');
        }
        return out.items;
    }
};

test "claims, conflicts, feed and inbox" {
    // session-lifetime ownership: the coordinator intentionally never
    // frees (lives on App until process exit) — page_allocator skips
    // the leak report that would flag that design.
    const alloc = std.heap.page_allocator;
    var co = Coordinator.init(alloc);
    co.register("main");
    co.register("t-1");
    try std.testing.expectEqual(@as(usize, 2), co.count());

    // claim + static conflict
    try std.testing.expect(co.claim("main", "src/*.zig") == null);
    try std.testing.expectEqualStrings("main", co.claim("t-1", "src/*.zig").?);
    // concrete enforcement: literal claim covers subtree, glob covers match
    try std.testing.expectEqualStrings("main", co.conflict("t-1", "src/a.zig").?);
    try std.testing.expect(co.conflict("main", "src/a.zig") == null); // own claim ok
    try std.testing.expect(co.conflict("t-1", "web/a.js") == null);

    // literal dir claim covers the dir itself and its subtree
    try std.testing.expect(co.claim("t-1", "docs") == null);
    try std.testing.expectEqualStrings("t-1", co.conflict("main", "docs/x.md").?);
    try std.testing.expectEqualStrings("t-1", co.conflict("main", "docs").?);

    // messaging
    try co.send("main", "t-1", "taking src/parser.zig");
    try co.send("main", "*", "broadcast hi");
    const d = try co.drain(alloc, "t-1");
    try std.testing.expect(std.mem.indexOf(u8, d, "taking src/parser.zig") != null);
    try std.testing.expect(std.mem.indexOf(u8, d, "broadcast hi") != null);
    // second drain is empty
    try std.testing.expectEqual(@as(usize, 0), (try co.drain(alloc, "t-1")).len);

    // release
    try std.testing.expectEqual(@as(usize, 1), co.release("main", "src/*.zig"));
    try std.testing.expect(co.conflict("t-1", "src/a.zig") == null);
}

test "subagent lineage never self-blocks" {
    const alloc = std.heap.page_allocator;
    var co = Coordinator.init(alloc);
    co.register("main");
    co.register("main/s");
    co.register("t-1");

    // parent claims; delegated subagent still writes there
    try std.testing.expect(co.claim("main", "src/x.zig") == null);
    try std.testing.expect(co.conflict("main/s", "src/x.zig") == null);
    try std.testing.expect(co.claim("main/s", "src/x.zig") == null); // no static conflict either
    // ...but an unrelated agent IS blocked both ways
    try std.testing.expectEqualStrings("main", co.conflict("t-1", "src/x.zig").?);
    // sub's own claim also doesn't block the parent (same worker chain)
    try std.testing.expect(co.claim("main/s", "web/app.js") == null);
    try std.testing.expect(co.conflict("main", "web/app.js") == null);
    try std.testing.expectEqualStrings("main/s", co.conflict("t-1", "web/app.js").?);
    // deeper nesting stays in the chain
    co.register("main/s/s");
    try std.testing.expect(co.conflict("main/s/s", "src/x.zig") == null);
}
