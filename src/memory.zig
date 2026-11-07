//! Supermemory v5 client — the session's long-term memory layer.
//!
//! Hosted (api.supermemory.ai) or self-hosted (`memory_base` in config).
//! Five routes cover the whole loop:
//!
//!   POST   /ns/{ns}/document           ingest raw content (conversations,
//!                                      notes, urls) — supermemory extracts
//!                                      and links the memories itself
//!   POST   /ns/{ns}/search             hybrid recall (memories + chunks)
//!   POST   /ns/{ns}/profile            static facts + dynamic context
//!   DELETE /ns/{ns}/memories           forget by id
//!   DELETE /ns/{ns}/memories/semantic  forget by meaning (dryRun first)
//!
//! The namespace partitions memory per scope — we default to
//! `xera-<workspace-name>` so a project's agents share one brain while
//! staying isolated from other projects. Agent ids ride along as
//! document metadata, not separate namespaces.
//!
//! Every call is best-effort at the agent layer: failures surface as
//! tool-result text, never as a crashed turn.

const std = @import("std");
const util = @import("util.zig");
const http = @import("http.zig");

pub const Memory = struct {
    alloc: std.mem.Allocator,
    hclient: *http.HttpClient,
    key: []const u8,
    base: []const u8, // no trailing slash
    ns: []const u8, // sanitized namespace

    /// Build a client. `ns_raw` is sanitized to supermemory's charset
    /// ([A-Za-z0-9_:-], ≤100 chars). Returns null without an api key.
    pub fn create(
        alloc: std.mem.Allocator,
        hclient: *http.HttpClient,
        key: ?[]const u8,
        base: []const u8,
        ns_raw: []const u8,
    ) ?*Memory {
        const k = key orelse return null;
        if (k.len == 0) return null;
        const self = alloc.create(Memory) catch return null;
        var ns = std.ArrayList(u8).init(alloc);
        for (ns_raw) |c| {
            const ok = std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == ':';
            ns.append(if (ok) c else '-') catch return null;
        }
        const n = @min(ns.items.len, 100);
        self.* = .{ .alloc = alloc, .hclient = hclient, .key = k, .base = base, .ns = ns.items[0..n] };
        return self;
    }

    /// Namespace derived from the workspace dir name: "xera-<basename>".
    pub fn defaultNs(alloc: std.mem.Allocator, workspace: []const u8) []const u8 {
        const base = std.fs.path.basename(std.mem.trimRight(u8, workspace, "/\\"));
        return std.fmt.allocPrint(alloc, "xera-{s}", .{base}) catch "xera";
    }

    fn auth(self: *Memory, alloc: std.mem.Allocator) ![]http.Header {
        const v = try std.fmt.allocPrint(alloc, "Bearer {s}", .{self.key});
        const h = try alloc.alloc(http.Header, 1);
        h[0] = .{ .name = "authorization", .value = v };
        return h;
    }

    fn url(self: *Memory, alloc: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(alloc, "{s}/ns/{s}/" ++ fmt, .{ self.base, self.ns } ++ args);
    }

    /// Raw call — returns the response body on 2xx, or an error carrying
    /// the status + a snippet of the error body for diagnostics.
    fn call(self: *Memory, alloc: std.mem.Allocator, method: std.http.Method, path: []const u8, body: ?[]const u8) ![]const u8 {
        const res = self.hclient.fetch(alloc, method, path, try self.auth(alloc), body) catch |e| {
            return std.fmt.allocPrint(alloc, "error: memory request failed: {s}", .{@errorName(e)});
        };
        const code = @intFromEnum(res.status);
        if (code >= 200 and code < 300) return res.body;
        const snippet = util.truncateMiddle(alloc, res.body, 300) catch res.body;
        return std.fmt.allocPrint(alloc, "error: memory api {d}: {s}", .{ code, snippet });
    }

    fn isErr(body: []const u8) bool {
        return std.mem.startsWith(u8, body, "error:");
    }

    /// Ingest raw content. `id` makes the doc append/update instead of
    /// duplicating (same id = same document). `dreaming` "instant" makes
    /// memories searchable right away at one extra billed op.
    pub fn add(
        self: *Memory,
        alloc: std.mem.Allocator,
        content: []const u8,
        id: ?[]const u8,
        meta_json: ?[]const u8,
        dreaming: ?[]const u8,
    ) ![]const u8 {
        var body = std.ArrayList(u8).init(alloc);
        try body.writer().print("{{\"content\":\"{s}\"", .{try util.jescape(alloc, content)});
        if (id) |i| try body.writer().print(",\"id\":\"{s}\"", .{try util.jescape(alloc, i)});
        if (meta_json) |m| try body.writer().print(",\"metadata\":{s}", .{m});
        if (dreaming) |d| try body.writer().print(",\"dreaming\":\"{s}\"", .{d});
        try body.append('}');
        const path = try self.url(alloc, "document", .{});
        const raw = try self.call(alloc, .POST, path, body.items);
        if (isErr(raw)) return raw;
        const v = try util.jparse(alloc, raw);
        const rid = util.jstr(v, "id") orelse "?";
        const st = util.jstr(v, "status") orelse "?";
        return std.fmt.allocPrint(alloc, "document {s} → {s}", .{ rid, st });
    }

    /// Hybrid recall. Returns one result per line:
    ///   `<id> [sim] <memory-or-chunk-text>`
    /// or "" when nothing is relevant. `err`-style strings pass through.
    pub fn search(self: *Memory, alloc: std.mem.Allocator, query: []const u8, limit: usize) ![]const u8 {
        const path = try self.url(alloc, "search", .{});
        const body = try std.fmt.allocPrint(alloc,
            "{{\"query\":\"{s}\",\"searchMode\":\"hybrid\",\"limit\":{d}}}",
            .{ try util.jescape(alloc, query), limit });
        const raw = try self.call(alloc, .POST, path, body);
        if (isErr(raw)) return raw;
        return formatResults(alloc, raw);
    }

    /// Static facts + dynamic context for this namespace, compact:
    ///   `facts: a · b · c\nrecent: x · y · z`
    pub fn profile(self: *Memory, alloc: std.mem.Allocator) ![]const u8 {
        const path = try self.url(alloc, "profile", .{});
        const raw = try self.call(alloc, .POST, path, "{}");
        if (isErr(raw)) return raw;
        const v = try util.jparse(alloc, raw);
        const prof = util.jget(v, "profile") orelse return "";
        var out = std.ArrayList(u8).init(alloc);
        try out.appendSlice("facts: ");
        try appendMemList(alloc, &out, prof, "static");
        try out.appendSlice("\nrecent: ");
        try appendMemList(alloc, &out, prof, "dynamic");
        return out.items;
    }

    /// Forget specific memory ids (comma-separated in `ids`).
    pub fn forget(self: *Memory, alloc: std.mem.Allocator, ids_csv: []const u8) ![]const u8 {
        var ids = std.ArrayList(u8).init(alloc);
        try ids.appendSlice("{\"ids\":[");
        var it = std.mem.tokenizeScalar(u8, ids_csv, ',');
        var first = true;
        while (it.next()) |id0| {
            const id = std.mem.trim(u8, id0, " \t\"'");
            if (id.len == 0) continue;
            if (!first) try ids.append(',');
            first = false;
            try ids.writer().print("\"{s}\"", .{try util.jescape(alloc, id)});
        }
        try ids.appendSlice("]}");
        const path = try self.url(alloc, "memories", .{});
        const raw = try self.call(alloc, .DELETE, path, ids.items);
        if (isErr(raw)) return raw;
        return formatForget(alloc, raw);
    }

    /// Forget by meaning. `dry_run` previews the matched memories; call
    /// again with apply=true to actually forget them.
    pub fn forgetQuery(self: *Memory, alloc: std.mem.Allocator, query: []const u8, apply: bool) ![]const u8 {
        const path = try self.url(alloc, "memories/semantic", .{});
        const body = try std.fmt.allocPrint(alloc,
            "{{\"query\":\"{s}\",\"dryRun\":{s}}}",
            .{ try util.jescape(alloc, query), if (apply) "false" else "true" });
        const raw = try self.call(alloc, .DELETE, path, body);
        if (isErr(raw)) return raw;
        const formatted = try formatForget(alloc, raw);
        if (!apply) {
            return std.fmt.allocPrint(alloc,
                "dry-run — would forget:\n{s}\n(re-run with apply=true to confirm)", .{formatted});
        }
        return std.fmt.allocPrint(alloc, "forgot:\n{s}", .{formatted});
    }
};

fn appendMemList(alloc: std.mem.Allocator, out: *std.ArrayList(u8), prof: std.json.Value, key: []const u8) !void {
    _ = alloc;
    const arr = util.jget(prof, key) orelse {
        try out.appendSlice("-");
        return;
    };
    if (arr != .array or arr.array.items.len == 0) {
        try out.appendSlice("-");
        return;
    }
    var first = true;
    for (arr.array.items) |item| {
        const m = util.jstr(item, "memory") orelse continue;
        if (!first) try out.appendSlice(" · ");
        first = false;
        try out.appendSlice(m);
    }
}

fn formatResults(alloc: std.mem.Allocator, raw: []const u8) ![]const u8 {
    const v = try util.jparse(alloc, raw);
    const arr = util.jget(v, "results") orelse return "";
    if (arr != .array) return "";
    var out = std.ArrayList(u8).init(alloc);
    for (arr.array.items) |r| {
        const text = util.jstr(r, "memory") orelse util.jstr(r, "chunk") orelse continue;
        const id = util.jstr(r, "id") orelse "";
        var sim: f64 = 0;
        if (util.jget(r, "similarity")) |s| {
            if (s == .float) sim = s.float else if (s == .integer) sim = @floatFromInt(s.integer);
        }
        const trimmed = try util.truncateMiddle(alloc, std.mem.trim(u8, text, " \n\t"), 400);
        try out.writer().print("{s} [{d:.2}] {s}\n", .{ id, sim, trimmed });
    }
    return std.mem.trimRight(u8, out.items, "\n");
}

fn formatForget(alloc: std.mem.Allocator, raw: []const u8) ![]const u8 {
    const v = try util.jparse(alloc, raw);
    var out = std.ArrayList(u8).init(alloc);
    if (util.jget(v, "matches")) |m| {
        if (m == .array) {
            for (m.array.items) |item| {
                const id = util.jstr(item, "id") orelse "?";
                const mem = util.jstr(item, "memory") orelse "";
                try out.writer().print("- {s} {s}\n", .{ id, mem });
            }
        }
    }
    if (out.items.len == 0) try out.appendSlice("(nothing matched)");
    if (util.jget(v, "errors")) |errs| {
        if (errs == .array and errs.array.items.len > 0) {
            try out.appendSlice("errors: ");
            for (errs.array.items, 0..) |e, i| {
                if (i > 0) try out.appendSlice(", ");
                try out.appendSlice(util.jstr(e, "id") orelse "?");
            }
        }
    }
    return std.mem.trimRight(u8, out.items, "\n");
}

test "hybrid result formatting" {
    const alloc = std.heap.page_allocator;
    const raw =
        \\{"results":[
        \\  {"id":"mem_1","memory":"User prefers dark mode","similarity":0.91},
        \\  {"id":"chunk_7","chunk":"dark mode is default since v2","similarity":0.66},
        \\  {"id":"mem_2","similarity":0.5}
        \\],"searchTime":42}
    ;
    const out = try formatResults(alloc, raw);
    try std.testing.expect(std.mem.indexOf(u8, out, "mem_1 [0.91] User prefers dark mode") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "chunk_7 [0.66] dark mode is default") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "mem_2") == null); // no text field
}

test "forget result formatting" {
    const alloc = std.heap.page_allocator;
    const raw =
        \\{"count":2,"matches":[{"id":"m1","memory":"a"},{"id":"m2","memory":"b"}],"errors":[{"id":"m9"}]}
    ;
    const out = try formatForget(alloc, raw);
    try std.testing.expect(std.mem.indexOf(u8, out, "- m1 a") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "errors: m9") != null);
    const empty = try formatForget(alloc, "{\"count\":0,\"matches\":[],\"errors\":[]}");
    try std.testing.expect(std.mem.indexOf(u8, empty, "nothing matched") != null);
}

test "namespace sanitize + default" {
    const alloc = std.heap.page_allocator;
    try std.testing.expectEqualStrings("xera-xera-rsi",
        Memory.defaultNs(alloc, "C:\\Users\\me\\xera-rsi"));
    const m = Memory.create(alloc, undefined, "k", "https://x", "weird ns!") orelse
        return error.TestFailed;
    try std.testing.expectEqualStrings("weird-ns-", m.ns);
    try std.testing.expect(Memory.create(alloc, undefined, null, "https://x", "a") == null);
}
