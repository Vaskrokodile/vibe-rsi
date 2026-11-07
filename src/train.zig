//! Training support for the RSI loop.
//!
//! Three jobs:
//!   1. dataset — build an SFT dataset from the trace store and/or pull a
//!      HuggingFace dataset the analyzer picked
//!   2. repo — keep a local checkout of the training algorithm
//!      (ETM: evict-train-merge, LoRA trained against a cached gradient so
//!      the base model is evicted from memory during the train phase)
//!   3. launch — spawn the trainer as a supervised background process with
//!      a log file; status() tails it so the UI can show progress
//!
//! Training is heavyweight and best-effort: missing python/CUDA/deps all
//! surface as a readable status line, never a crash.

const std = @import("std");
const util = @import("util.zig");
const config = @import("config.zig");
const http = @import("http.zig");
const traces = @import("traces.zig");

pub const ETM_URL = "https://github.com/Vaskrokodile/etm-evict-train-merge";
pub const ETM_ZIP = "https://codeload.github.com/Vaskrokodile/etm-evict-train-merge/zip/refs/heads/main";
/// best variant per the repo README (v5b: gentle MSE + breathing)
pub const ETM_ENTRY = "src/etm_v5b.py";

/// Where training artifacts live: ~/.xera-rsi/train/{etm,data,logs}
pub fn rootDir(alloc: std.mem.Allocator) ![]u8 {
    const home = try util.homeDir(alloc);
    return std.fs.path.join(alloc, &.{ home, ".xera-rsi", "train" });
}

pub fn repoDir(alloc: std.mem.Allocator) ![]u8 {
    const root = try rootDir(alloc);
    return std.fs.path.join(alloc, &.{ root, "etm" });
}

fn haveCommand(alloc: std.mem.Allocator, name: []const u8) bool {
    var child = std.process.Child.init(&.{ name, "--version" }, alloc);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Ignore;
    child.stderr_behavior = .Ignore;
    child.spawn() catch return false;
    _ = child.wait() catch return false;
    return true;
}

/// Local checkout of the ETM repo; clones on first use (git), falls back
/// to a codeload zip extracted with std.zip when git is missing.
pub fn ensureRepo(alloc: std.mem.Allocator) ![]const u8 {
    const dir = try repoDir(alloc);
    const marker = try std.fs.path.join(alloc, &.{ dir, ETM_ENTRY });
    if (std.fs.cwd().access(marker, .{})) |_| {
        return dir;
    } else |_| {}

    if (haveCommand(alloc, "git")) {
        var child = std.process.Child.init(
            &.{ "git", "clone", "--depth", "1", ETM_URL, dir }, alloc);
        child.stdin_behavior = .Ignore;
        child.stdout_behavior = .Ignore;
        child.stderr_behavior = .Ignore;
        child.spawn() catch return error.GitSpawnFailed;
        const term = child.wait() catch return error.GitFailed;
        if (term == .Exited and term.Exited == 0) return dir;
        // clone failed — clean the partial dir before the zip fallback
        std.fs.cwd().deleteTree(dir) catch {};
    }
    return downloadZip(alloc, dir);
}

fn downloadZip(alloc: std.mem.Allocator, dir: []const u8) ![]const u8 {
    var client = http.HttpClient.init(alloc);
    defer client.deinit();
    const res = try client.fetch(alloc, .GET, ETM_ZIP, &.{}, null);
    if (res.status != .ok) return error.ZipDownloadFailed;
    const zip_path = try std.fmt.allocPrint(alloc, "{s}.zip", .{dir});
    if (std.fs.path.dirname(zip_path)) |d| try std.fs.cwd().makePath(d);
    try std.fs.cwd().writeFile(.{ .sub_path = zip_path, .data = res.body });
    defer std.fs.cwd().deleteFile(zip_path) catch {};

    // extract to a staging dir, then move the single repo root into place
    const stage = try std.fmt.allocPrint(alloc, "{s}-stage", .{dir});
    std.fs.cwd().deleteTree(stage) catch {};
    var stage_dir = try std.fs.cwd().makeOpenPath(stage, .{ .iterate = true });
    defer stage_dir.close();
    const zf = try std.fs.cwd().openFile(zip_path, .{});
    defer zf.close();
    var diag: std.zip.Diagnostics = .{ .allocator = alloc };
    defer diag.deinit();
    try std.zip.extract(stage_dir, zf.seekableStream(), .{ .diagnostics = &diag });

    var inner: []const u8 = "";
    if (diag.root_dir.len > 0) {
        inner = diag.root_dir;
    } else {
        var it = stage_dir.iterate();
        while (try it.next()) |e| {
            if (e.kind == .directory) {
                inner = try alloc.dupe(u8, e.name);
                break;
            }
        }
    }
    if (inner.len == 0) return error.RepoExtractFailed;
    const src = try std.fs.path.join(alloc, &.{ stage, inner });
    std.fs.cwd().deleteTree(dir) catch {};
    try std.fs.cwd().rename(src, dir);
    std.fs.cwd().deleteTree(stage) catch {};

    if (std.fs.cwd().access(try std.fs.path.join(alloc, &.{ dir, ETM_ENTRY }), .{})) |_| {} else |_|
        return error.RepoExtractFailed;
    return dir;
}

/// Build the SFT dataset from the trace store. Returns {path, samples}.
pub const Dataset = struct { path: []const u8, samples: usize };

pub fn buildDataset(alloc: std.mem.Allocator, tag: []const u8) !Dataset {
    const root = try rootDir(alloc);
    const data_dir = try std.fs.path.join(alloc, &.{ root, "data" });
    try std.fs.cwd().makePath(data_dir);
    const path = try std.fs.path.join(alloc, &.{
        data_dir,
        try std.fmt.allocPrint(alloc, "xera-{s}-{d}.jsonl",
            .{ tag, std.time.timestamp() }),
    });
    const n = try traces.exportDataset(alloc, path, true);
    return .{ .path = path, .samples = n };
}

pub const HfHit = struct { id: []const u8, likes: i64, downloads: i64 };

/// Search HuggingFace datasets, sorted by likes. Returns up to `limit`.
pub fn hfSearch(
    alloc: std.mem.Allocator,
    hclient: *http.HttpClient,
    query: []const u8,
    limit: usize,
) ![]HfHit {
    var enc = std.ArrayList(u8).init(alloc);
    for (query) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '-' or c == '_') try enc.append(c)
        else try enc.writer().print("%{X:0>2}", .{c});
    }
    const url = try std.fmt.allocPrint(alloc,
        "https://huggingface.co/api/datasets?search={s}&limit={d}&sort=likes&direction=-1",
        .{ enc.items, limit });
    const res = try hclient.fetch(alloc, .GET, url, &.{}, null);
    if (res.status != .ok) return error.HfSearchFailed;
    const root = try util.jparse(alloc, res.body);
    if (root != .array) return error.HfBadResponse;
    var out = std.ArrayList(HfHit).init(alloc);
    for (root.array.items) |d| {
        const id = util.jstr(d, "id") orelse continue;
        try out.append(.{
            .id = id,
            .likes = util.jint(d, "likes") orelse 0,
            .downloads = util.jint(d, "downloads") orelse 0,
        });
    }
    return out.items;
}

/// Download the first train-ish text file of a HF dataset
/// (.jsonl/.json/.csv/.txt — parquet is skipped, we can't convert it).
/// Returns the saved path.
pub fn hfDownload(
    alloc: std.mem.Allocator,
    hclient: *http.HttpClient,
    repo_id: []const u8,
) ![]const u8 {
    const api = try std.fmt.allocPrint(alloc,
        "https://huggingface.co/api/datasets/{s}", .{repo_id});
    const res = try hclient.fetch(alloc, .GET, api, &.{}, null);
    if (res.status != .ok) return error.HfDatasetFailed;
    const root = try util.jparse(alloc, res.body);
    const sibs = util.jget(root, "siblings") orelse return error.HfBadResponse;
    if (sibs != .array) return error.HfBadResponse;

    var pick: ?[]const u8 = null;
    var fallback: ?[]const u8 = null;
    for (sibs.array.items) |s| {
        const name = util.jstr(s, "rfilename") orelse continue;
        const ok_ext = std.mem.endsWith(u8, name, ".jsonl") or
            std.mem.endsWith(u8, name, ".json") or
            std.mem.endsWith(u8, name, ".csv") or
            std.mem.endsWith(u8, name, ".txt");
        if (!ok_ext) continue;
        if (std.mem.indexOf(u8, name, "train") != null) {
            pick = name;
            break;
        }
        if (fallback == null) fallback = name;
    }
    const file = pick orelse fallback orelse return error.HfNoTextFile;

    const dl = try std.fmt.allocPrint(alloc,
        "https://huggingface.co/datasets/{s}/resolve/main/{s}", .{ repo_id, file });
    const dres = try hclient.fetch(alloc, .GET, dl, &.{}, null);
    if (dres.status != .ok) return error.HfDownloadFailed;
    if (dres.body.len > (64 << 20)) return error.HfTooLarge;

    const root_dir = try rootDir(alloc);
    const data_dir = try std.fs.path.join(alloc, &.{ root_dir, "data", "hf" });
    try std.fs.cwd().makePath(data_dir);
    var safe = std.ArrayList(u8).init(alloc);
    for (repo_id) |c| try safe.append(if (c == '/') '_' else c);
    const out_path = try std.fs.path.join(alloc, &.{
        data_dir, try std.fmt.allocPrint(alloc, "{s}--{s}", .{ safe.items, file }) });
    try std.fs.cwd().writeFile(.{ .sub_path = out_path, .data = dres.body });
    return out_path;
}

// ---------- launch ----------

pub const TrainProc = struct {
    child: *std.process.Child,
    log_path: []const u8,
    started_ms: i64,
    dataset: []const u8,
    done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    code: i64 = -1,
};

var current: ?*TrainProc = null;
var current_mu: std.Thread.Mutex = .{};

/// Launch ETM training on a dataset. The process is detached, output is
/// teed into ~/.xera-rsi/train/logs/. Returns the handle (also registered
/// globally for `status`).
pub fn launchEtm(
    alloc: std.mem.Allocator,
    dataset_path: []const u8,
    extra_args: []const u8,
) !*TrainProc {
    if (!haveCommand(alloc, "python")) return error.NoPython;
    const repo = try ensureRepo(alloc);
    const root = try rootDir(alloc);
    const logs = try std.fs.path.join(alloc, &.{ root, "logs" });
    try std.fs.cwd().makePath(logs);
    const log_path = try std.fs.path.join(alloc, &.{
        logs, try std.fmt.allocPrint(alloc, "train-{d}.log", .{std.time.timestamp()}) });

    const cmdline = try std.fmt.allocPrint(alloc,
        "cd /d \"{s}\" && python -u {s} --data \"{s}\" {s} > \"{s}\" 2>&1",
        .{ repo, ETM_ENTRY, dataset_path, extra_args, log_path });
    const argv = if (@import("builtin").os.tag == .windows)
        &.{ "cmd", "/c", cmdline }
    else
        &.{ "sh", "-c", cmdline };

    const child = try alloc.create(std.process.Child);
    child.* = std.process.Child.init(argv, alloc);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Ignore;
    child.stderr_behavior = .Ignore;
    try child.spawn();

    const tp = try alloc.create(TrainProc);
    tp.* = .{
        .child = child,
        .log_path = log_path,
        .started_ms = std.time.milliTimestamp(),
        .dataset = dataset_path,
    };
    const th = try std.Thread.spawn(.{}, watchProc, .{tp});
    th.detach();
    current_mu.lock();
    current = tp;
    current_mu.unlock();
    return tp;
}

fn watchProc(tp: *TrainProc) void {
    const term = tp.child.wait() catch {
        tp.done.store(true, .release);
        return;
    };
    tp.code = switch (term) {
        .Exited => |c| @intCast(c),
        else => -1,
    };
    tp.done.store(true, .release);
}

pub const TrainStatus = struct {
    running: bool,
    log_tail: []const u8,
    elapsed_s: i64,
    log_path: []const u8,
    code: i64,
};

/// Snapshot of the current (or most recent) training process.
pub fn status(alloc: std.mem.Allocator) TrainStatus {
    current_mu.lock();
    const tp = current;
    current_mu.unlock();
    const t = tp orelse return .{
        .running = false, .log_tail = "", .elapsed_s = 0, .log_path = "", .code = -1 };
    var tail: []const u8 = "";
    if (std.fs.cwd().openFile(t.log_path, .{})) |f| {
        defer f.close();
        const size = (f.stat() catch return .{
            .running = !t.done.load(.acquire), .log_tail = "",
            .elapsed_s = elapsed(t), .log_path = t.log_path, .code = t.code }).size;
        const want: u64 = @min(size, 4000);
        f.seekTo(size - want) catch {};
        tail = f.readToEndAlloc(alloc, @intCast(want)) catch "";
        if (size > want) {
            if (std.mem.indexOfScalar(u8, tail, '\n')) |i| tail = tail[i + 1 ..];
        }
    } else |_| {}
    return .{
        .running = !t.done.load(.acquire),
        .log_tail = tail,
        .elapsed_s = elapsed(t),
        .log_path = t.log_path,
        .code = t.code,
    };
}

fn elapsed(t: *TrainProc) i64 {
    return @divTrunc(std.time.milliTimestamp() - t.started_ms, 1000);
}

// ---------- launch presets ----------

/// Named ETM configurations — the `train` tool exposes them so a launch
/// is one call with a preset name instead of a hand-built arg string.
pub const Preset = struct { name: []const u8, desc: []const u8, args: []const u8 };
pub const presets = [_]Preset{
    .{ .name = "quick", .desc = "smoke test: 1 epoch, rank-8 lora — validates the pipeline end to end", .args = "--epochs 1 --lora-rank 8 --lr 2e-4" },
    .{ .name = "standard", .desc = "default post-rsi run: 3 epochs, rank 16", .args = "--epochs 3 --lora-rank 16 --lr 1e-4" },
    .{ .name = "deep", .desc = "large capability gap: 5 epochs, rank 32, lower lr", .args = "--epochs 5 --lora-rank 32 --lr 5e-5" },
};

pub fn presetArgs(name: []const u8) ?[]const u8 {
    for (&presets) |p| {
        if (std.mem.eql(u8, p.name, name)) return p.args;
    }
    return null;
}

// ---------- dataset quality ----------

/// Normalize a text field for dedup: lowercase + whitespace-collapsed.
fn normText(alloc: std.mem.Allocator, s: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(alloc);
    var last_ws = true;
    for (s) |c| {
        if (std.ascii.isWhitespace(c)) {
            if (!last_ws) try out.append(' ');
            last_ws = true;
        } else {
            try out.append(std.ascii.toLower(c));
            last_ws = false;
        }
    }
    if (out.items.len > 0 and out.items[out.items.len - 1] == ' ') _ = out.pop();
    return out.toOwnedSlice();
}

/// A row is repetitive when a single 24-char fragment covers >60% of it.
fn isRepetitive(s: []const u8) bool {
    if (s.len < 200) return false;
    const frag = s[0..24];
    var count: usize = 1;
    var i: usize = 24;
    while (std.mem.indexOfPos(u8, s, i, frag)) |j| {
        count += 1;
        i = j + 24;
    }
    return count * 24 * 10 > s.len * 6;
}

const RowClass = struct {
    bad_json: u32 = 0,
    empty: u32 = 0,
    exact_dup: u32 = 0,
    prompt_dup: u32 = 0,
    echo: u32 = 0,
    too_short: u32 = 0,
    too_long: u32 = 0,
    repetitive: u32 = 0,
    contaminated: u32 = 0,
    ok: u32 = 0,
};

fn classify(
    alloc: std.mem.Allocator,
    line: []const u8,
    seen_rows: *std.StringHashMap(void),
    seen_prompts: *std.StringHashMap(void),
    ref_prompts: ?*std.StringHashMap(void),
) ![]const u8 {
    const v = util.jparse(alloc, line) catch return "bad_json";
    const prompt = util.jstr(v, "prompt") orelse util.jstr(v, "instruction") orelse "";
    const completion = util.jstr(v, "completion") orelse util.jstr(v, "response") orelse
        util.jstr(v, "output") orelse "";
    if (prompt.len == 0 or completion.len == 0) return "empty";
    if (completion.len < 16) return "too_short";
    if (prompt.len + completion.len > 128_000) return "too_long";
    const np = try normText(alloc, prompt);
    const nc = try normText(alloc, completion);
    // echo: the whole prompt appears inside the completion and the answer
    // is short enough to be mostly parroting (not prompt + real answer)
    if (np.len >= 8 and std.mem.indexOf(u8, nc, np) != null and
        np.len * 4 >= nc.len)
        return "echo";
    const rowkey = try std.fmt.allocPrint(alloc, "{s}\x00{s}", .{ np, nc });
    if (seen_rows.contains(rowkey)) return "exact_dup";
    try seen_rows.put(rowkey, {});
    if (seen_prompts.contains(np)) return "prompt_dup";
    try seen_prompts.put(np, {});
    if (ref_prompts) |rp| {
        if (rp.contains(np)) return "contaminated";
    }
    if (isRepetitive(completion)) return "repetitive";
    return "ok";
}

/// Load a reference set of normalized prompts (the "do not train on me"
/// list — typically your eval/verify prompts).
fn loadRefSet(alloc: std.mem.Allocator, ref_path: []const u8) !std.StringHashMap(void) {
    var set = std.StringHashMap(void).init(alloc);
    const body = try std.fs.cwd().readFileAlloc(alloc, ref_path, 64 << 20);
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line| {
        if (line.len < 2) continue;
        const v = util.jparse(alloc, line) catch {
            try set.put(try normText(alloc, line), {});
            continue;
        };
        const p = util.jstr(v, "prompt") orelse util.jstr(v, "instruction") orelse continue;
        try set.put(try normText(alloc, p), {});
    }
    return set;
}

/// Audit a JSONL dataset: duplicates, echo/leak rows, degenerate rows,
/// contamination vs an optional reference set. Returns a readable report.
pub fn checkDataset(
    alloc: std.mem.Allocator,
    path: []const u8,
    ref_path: ?[]const u8,
) ![]const u8 {
    const body = std.fs.cwd().readFileAlloc(alloc, path, 64 << 20) catch
        return error.DatasetUnreadable;
    var ref_set: ?std.StringHashMap(void) = null;
    if (ref_path) |rp| ref_set = loadRefSet(alloc, rp) catch null;

    var cls = RowClass{};
    var seen_rows = std.StringHashMap(void).init(alloc);
    var seen_prompts = std.StringHashMap(void).init(alloc);
    var plens = std.ArrayList(u64).init(alloc);
    var clens = std.ArrayList(u64).init(alloc);
    var examples = std.ArrayList([]const u8).init(alloc);
    var total: usize = 0;

    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line0| {
        const line = std.mem.trimRight(u8, line0, "\r");
        if (line.len < 2) continue;
        total += 1;
        const kind = classify(alloc, line, &seen_rows, &seen_prompts,
            if (ref_set) |*rp| rp else null) catch "bad_json";
        if (!std.mem.eql(u8, kind, "ok") and examples.items.len < 6) {
            const snip = if (line.len > 90) line[0..90] else line;
            try examples.append(try std.fmt.allocPrint(alloc,
                "  - [{s}] {s}…", .{ kind, snip }));
        }
        if (util.jparse(alloc, line)) |v| {
            const p = util.jstr(v, "prompt") orelse "";
            const c = util.jstr(v, "completion") orelse "";
            if (p.len > 0) try plens.append(p.len);
            if (c.len > 0) try clens.append(c.len);
        } else |_| {}
        bump(&cls, kind);
    }

    var out = std.ArrayList(u8).init(alloc);
    const w = out.writer();
    try w.print("dataset: {s}\nrows: {d} total\n", .{ path, total });
    try w.print("ok {d} · bad-json {d} · empty {d} · dup {d}+{d}prompt · echo {d} · short {d} · long {d} · repetitive {d} · contaminated {d}\n", .{
        cls.ok, cls.bad_json, cls.empty, cls.exact_dup, cls.prompt_dup,
        cls.echo, cls.too_short, cls.too_long, cls.repetitive, cls.contaminated,
    });
    try w.print("lengths: prompt median {d} · completion median {d}\n", .{
        medianOf(plens.items), medianOf(clens.items) });
    if (examples.items.len > 0) {
        try w.writeAll("examples:\n");
        for (examples.items) |e| try w.print("{s}\n", .{e});
    }
    const bad = total - cls.ok;
    if (bad == 0) {
        try w.writeAll("verdict: clean — good to train on\n");
    } else {
        try w.print("verdict: {d} row(s) need cleaning — run train{{op:\"clean\",path:\"{s}\"}} or clean it by hand\n", .{ bad, path });
    }
    return out.items;
}

fn bump(cls: *RowClass, kind: []const u8) void {
    inline for (@typeInfo(RowClass).@"struct".fields) |f| {
        if (std.mem.eql(u8, f.name, kind)) {
            @field(cls, f.name) += 1;
            return;
        }
    }
}

fn medianOf(v: []u64) u64 {
    if (v.len == 0) return 0;
    std.mem.sort(u64, v, {}, std.sort.asc(u64));
    return v[v.len / 2];
}

/// Write a cleaned copy of a dataset: drops every row `checkDataset`
/// would flag. Returns a short report.
pub fn cleanDataset(
    alloc: std.mem.Allocator,
    in_path: []const u8,
    out_path: []const u8,
    ref_path: ?[]const u8,
) ![]const u8 {
    const body = std.fs.cwd().readFileAlloc(alloc, in_path, 64 << 20) catch
        return error.DatasetUnreadable;
    var ref_set: ?std.StringHashMap(void) = null;
    if (ref_path) |rp| ref_set = loadRefSet(alloc, rp) catch null;

    var cls = RowClass{};
    var seen_rows = std.StringHashMap(void).init(alloc);
    var seen_prompts = std.StringHashMap(void).init(alloc);
    var kept = std.ArrayList(u8).init(alloc);
    var total: usize = 0;

    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line0| {
        const line = std.mem.trimRight(u8, line0, "\r");
        if (line.len < 2) continue;
        total += 1;
        const kind = classify(alloc, line, &seen_rows, &seen_prompts,
            if (ref_set) |*rp| rp else null) catch "bad_json";
        bump(&cls, kind);
        if (std.mem.eql(u8, kind, "ok")) {
            try kept.appendSlice(line);
            try kept.append('\n');
        }
    }
    if (std.fs.path.dirname(out_path)) |d| try std.fs.cwd().makePath(d);
    try std.fs.cwd().writeFile(.{ .sub_path = out_path, .data = kept.items });
    return std.fmt.allocPrint(alloc,
        "cleaned {s} → {s}: kept {d}/{d} rows (dropped dup {d}+{d}prompt, echo {d}, bad {d}, short {d}, long {d}, repetitive {d}, contaminated {d})",
        .{ in_path, out_path, cls.ok, total, cls.exact_dup, cls.prompt_dup,
           cls.echo, cls.bad_json + cls.empty, cls.too_short, cls.too_long,
           cls.repetitive, cls.contaminated });
}

test "checkDataset + cleanDataset classify rows" {
    // arena over the gpa — the codebase parses json leaky, by design
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(alloc,
        "{s}/ds.jsonl", .{try tmp.dir.realpathAlloc(alloc, ".")});
    const fixture =
        \\{"prompt":"how do i reverse a list in zig","completion":"Use std.mem.reverse(T, slice) — it reverses in place."}
        \\{"prompt":"how do i reverse a list in zig","completion":"Use std.mem.reverse(T, slice) — it reverses in place."}
        \\{"prompt":"same prompt different answer","completion":"first answer that is long enough to pass the length check."}
        \\{"prompt":"same prompt different answer","completion":"second answer, also long enough to be a distinct row."}
        \\{"prompt":"tiny","completion":"short"}
        \\{"prompt":"echo me please","completion":"echo me please echo me please echo me please"}
        \\not json at all
        \\{"prompt":"keep me","completion":"a perfectly reasonable answer with enough substance to keep."}
        \\
    ;
    try tmp.dir.writeFile(.{ .sub_path = "ds.jsonl", .data = fixture });
    const report = try checkDataset(alloc, path, null);
    // ok rows: first "reverse" row, first "same prompt" row, "keep me"
    try std.testing.expect(std.mem.indexOf(u8, report, "ok 3") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "bad-json 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "dup 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "1prompt") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "echo 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "short 1") != null);

    const out_path = try std.fmt.allocPrint(alloc, "{s}.clean.jsonl", .{path});
    const clean_report = try cleanDataset(alloc, path, out_path, null);
    try std.testing.expect(std.mem.indexOf(u8, clean_report, "kept 3") != null);
    const kept = try std.fs.cwd().readFileAlloc(alloc, out_path, 1 << 20);
    try std.testing.expect(std.mem.indexOf(u8, kept, "keep me") != null);
    try std.testing.expect(std.mem.indexOf(u8, kept, "reverse a list") != null);
    try std.testing.expect(std.mem.indexOf(u8, kept, "second answer") == null);
    try std.testing.expect(std.mem.indexOf(u8, kept, "echo me") == null);
}
