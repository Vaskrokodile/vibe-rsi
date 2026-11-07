const std = @import("std");
const util = @import("util.zig");
const llm = @import("llm.zig");
const harness = @import("harness.zig");
const config = @import("config.zig");
const live = @import("live.zig");
const coord_mod = @import("coord.zig");
const memory_mod = @import("memory.zig");
const traces_mod = @import("traces.zig");
const train_mod = @import("train.zig");
const refine_mod = @import("refine.zig");
const http = @import("http.zig");

pub const Ui = struct {
    ctx: *anyopaque,
    on_text: ?*const fn (ctx: *anyopaque, text: []const u8) void = null,
    on_reasoning: ?*const fn (ctx: *anyopaque, text: []const u8) void = null,
    on_tool_call: ?*const fn (ctx: *anyopaque, name: []const u8, summary: []const u8) void = null,
    on_tool_result: ?*const fn (ctx: *anyopaque, name: []const u8, preview: []const u8, ok: bool) void = null,
    on_status: ?*const fn (ctx: *anyopaque, msg: []const u8) void = null,
};

pub const Danger = enum { safe, write, shell };

pub const ApprovalDecision = enum { allow, deny, always };

pub const Approver = struct {
    ctx: *anyopaque,
    ask: *const fn (ctx: *anyopaque, tool_name: []const u8, summary: []const u8, danger: Danger) anyerror!ApprovalDecision,
};

pub const ToolCtx = struct {
    /// arena that outlives the call — results live in history
    alloc: std.mem.Allocator,
    /// scratch for the call itself
    scratch: std.mem.Allocator,
    cwd: []const u8,
    /// absolute workspace root for write confinement
    workspace: []const u8,
    h: *harness.Harness,
    cfg: *const config.Config,
    approver: ?*const Approver,
    depth: u8 = 0,
    /// forwarded to subagents so their activity is visible
    ui: ?*const Ui = null,
    /// live observability stream, shared with subagents
    livelog: ?*live.Live = null,
    /// shared "always allow" set across subagents
    always_allow: ?*std.StringHashMap(void) = null,
    /// set by the agent so the `task` tool can spawn a nested run
    subagent_run: ?*const fn (ctx: *ToolCtx, prompt: []const u8, agent_name: ?[]const u8) anyerror![]const u8 = null,
    /// shared-session coordination fabric (claims, feed, inbox); null when
    /// the agent isn't attached to a coordinator
    coord: ?*coord_mod.Coordinator = null,
    /// this agent's id inside the coordinator ("main", "t-1", "main/s")
    aid: []const u8 = "",
    /// supermemory client; null when no api key is configured
    memory: ?*memory_mod.Memory = null,
    /// background shells owned by this agent (run_shell background:true)
    shells: ?*Shells = null,
    /// async subagents spawned by `task op=spawn`
    subs: ?*SubRegistry = null,
    /// notices delivered into the agent's next <sync> block
    notices: ?*NoticeQueue = null,
    /// persistent per-agent goal (goal tool)
    goal_state: ?*GoalState = null,
    /// pending refinement request (refine tool, runs at turn end)
    refine_req: ?*RefineReq = null,
    /// explicit aid for a spawned subagent ("parent/sub-3")
    sub_aid: ?[]const u8 = null,
    /// telemetry store (traces tool); null when tracing is off
    traces: ?*traces_mod.Store = null,
    /// session id for trace records (subagents derive "<sid>/<aid>")
    sid: []const u8 = "",
    /// http client for tools that fetch (train hf/pull); null = offline
    hclient: ?*http.HttpClient = null,
};

/// Events pushed to the agent between iterations — rendered into the
/// same <sync> block as coord feed/inbox lines.
pub const NoticeQueue = struct {
    mu: std.Thread.Mutex = .{},
    items: std.ArrayList([]const u8),

    pub fn init(alloc: std.mem.Allocator) NoticeQueue {
        return .{ .items = std.ArrayList([]const u8).init(alloc) };
    }
    pub fn push(self: *NoticeQueue, alloc: std.mem.Allocator, msg: []const u8) void {
        const copy = alloc.dupe(u8, msg) catch return;
        self.mu.lock();
        defer self.mu.unlock();
        self.items.append(copy) catch {};
    }
    /// Take all pending notices (caller keeps the slice).
    pub fn drain(self: *NoticeQueue) [][]const u8 {
        self.mu.lock();
        defer self.mu.unlock();
        const out = self.items.toOwnedSlice() catch return &.{};
        self.items.clearRetainingCapacity();
        return out;
    }
};

/// One background shell: the process runs detached, output streams into
/// out/err under the mutex, and a "[shell-done]" notice fires on exit.
pub const BgShell = struct {
    id: []const u8,
    command: []const u8,
    mu: std.Thread.Mutex = .{},
    out: std.ArrayList(u8),
    err: std.ArrayList(u8),
    done: bool = false,
    code: i64 = -1,
    child: ?*std.process.Child = null,
    started_ms: i64 = 0,
};

pub const Shells = struct {
    alloc: std.mem.Allocator,
    mu: std.Thread.Mutex = .{},
    map: std.StringHashMap(*BgShell),
    next: u32 = 1,
    notices: ?*NoticeQueue = null,

    pub fn init(alloc: std.mem.Allocator, notices: ?*NoticeQueue) Shells {
        return .{
            .alloc = alloc,
            .map = std.StringHashMap(*BgShell).init(alloc),
            .notices = notices,
        };
    }
};

/// One spawned (async) subagent tracked by its parent.
pub const BgSub = struct {
    aid: []const u8, // "parent/sub-N"
    title: []const u8,
    mu: std.Thread.Mutex = .{},
    done: bool = false,
    result: []const u8 = "",
    is_err: bool = false,
    started_ms: i64 = 0,
};

pub const SubRegistry = struct {
    alloc: std.mem.Allocator,
    mu: std.Thread.Mutex = .{},
    map: std.StringHashMap(*BgSub),
    next: u32 = 1,

    pub fn init(alloc: std.mem.Allocator) SubRegistry {
        return .{ .alloc = alloc, .map = std.StringHashMap(*BgSub).init(alloc) };
    }
};

/// Persistent per-agent objective, injected into every sync block while
/// active. The coord map mirrors it for `status`; this is the source an
/// agent injects from (works with or without a coordinator).
pub const GoalState = struct {
    mu: std.Thread.Mutex = .{},
    objective: []const u8 = "",
    set_ms: i64 = 0,
    turns: u32 = 0,
};

/// A refine request scheduled by the `refine` tool — the host runs it
/// when the turn ends (never mid-turn), like prime-agent's refine.run().
pub const RefineReq = struct {
    mu: std.Thread.Mutex = .{},
    pending: ?[]const u8 = null,
    scope_global: bool = false,
};

pub const Tool = struct {
    name: []const u8,
    description: []const u8,
    schema_src: []const u8,
    danger: Danger,
    run: *const fn (ctx: *ToolCtx, args: std.json.Value) anyerror![]const u8,
    summary: *const fn (ctx: *ToolCtx, args: std.json.Value) []const u8,
};

pub const tools = [_]Tool{
    .{
        .name = "read_file",
        .description = "Read a file's contents. Returns the text (truncated past ~100KB). Optionally offset/limit by lines.",
        .schema_src = \\{"type":"object","properties":{"path":{"type":"string","description":"file path, relative to the workspace"},"offset":{"type":"integer","description":"1-based start line"},"limit":{"type":"integer","description":"max lines to return"}},"required":["path"]}
        ,
        .danger = .safe,
        .run = runReadFile,
        .summary = sumPath,
    },
    .{
        .name = "write_file",
        .description = "Create or fully overwrite a file with the given content. Parent directories are created.",
        .schema_src = \\{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}
        ,
        .danger = .write,
        .run = runWriteFile,
        .summary = sumPath,
    },
    .{
        .name = "edit_file",
        .description = "Replace an exact string in a file. `old_string` must match exactly once unless replace_all is set. Prefer this for surgical edits.",
        .schema_src = \\{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["path","old_string","new_string"]}
        ,
        .danger = .write,
        .run = runEditFile,
        .summary = sumPath,
    },
    .{
        .name = "list_dir",
        .description = "List entries in a directory with their type (file/dir).",
        .schema_src = \\{"type":"object","properties":{"path":{"type":"string"}}}
        ,
        .danger = .safe,
        .run = runListDir,
        .summary = sumPath,
    },
    .{
        .name = "glob",
        .description = "Find files matching a glob pattern (e.g. 'src/**/*.zig'). '*' does not cross '/', '**' does.",
        .schema_src = \\{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string","description":"directory to search (default: workspace root)"}},"required":["pattern"]}
        ,
        .danger = .safe,
        .run = runGlob,
        .summary = sumPattern,
    },
    .{
        .name = "grep",
        .description = "Search file contents for a substring. Returns 'file:line: text' matches. Options: file_glob filter, ignore_case.",
        .schema_src = \\{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string","description":"file or directory to search (default: workspace)"},"file_glob":{"type":"string","description":"e.g. '*.zig'"},"ignore_case":{"type":"boolean"}},"required":["pattern"]}
        ,
        .danger = .safe,
        .run = runGrep,
        .summary = sumPattern,
    },
    .{
        .name = "delete",
        .description = "Delete a file or directory. Set recursive=true to remove a non-empty directory. Destructive — confirm intent.",
        .schema_src = \\{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean","description":"required true to delete a non-empty directory"}},"required":["path"]}
        ,
        .danger = .write,
        .run = runDelete,
        .summary = sumPath,
    },
    .{
        .name = "make_dir",
        .description = "Create a directory (and any missing parents).",
        .schema_src = \\{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
        ,
        .danger = .write,
        .run = runMakeDir,
        .summary = sumPath,
    },
    .{
        .name = "web_fetch",
        .description = "Fetch a URL and return its content as text (HTML is stripped to readable text). Use for docs, APIs, and pages.",
        .schema_src = \\{"type":"object","properties":{"url":{"type":"string"},"max_chars":{"type":"integer","description":"cap on returned text, default 30000"}},"required":["url"]}
        ,
        .danger = .safe,
        .run = runWebFetch,
        .summary = sumUrl,
    },
    .{
        .name = "web_search",
        .description = "Search the web (DuckDuckGo). Returns title/url/snippet hits.",
        .schema_src = \\{"type":"object","properties":{"query":{"type":"string"},"max_results":{"type":"integer","description":"default 8"}},"required":["query"]}
        ,
        .danger = .safe,
        .run = runWebSearch,
        .summary = sumPattern,
    },
    .{
        .name = "run_shell",
        .description = "Run a shell command in the workspace (cmd /c on Windows, /bin/sh -c elsewhere). Foreground: returns stdout, stderr, exit code (timeout_ms, default 120s). background:true returns a shell id immediately — use it for servers/watchers/long builds; a [shell-done] notice arrives on exit. Manage with {id: sh-N} snapshot, {id, wait_ms} block-for-finish, {id, kill:true}.",
        .schema_src = \\{"type":"object","properties":{"command":{"type":"string"},"timeout_ms":{"type":"integer"},"cwd":{"type":"string"},"background":{"type":"boolean"},"id":{"type":"string"},"wait_ms":{"type":"integer"},"kill":{"type":"boolean"}}}
        ,
        .danger = .shell,
        .run = runShell,
        .summary = sumCommand,
    },
    .{
        .name = "memory_append",
        .description = "Append a short durable fact to long-term memory (harness/memory.md). Persists across sessions.",
        .schema_src = \\{"type":"object","properties":{"text":{"type":"string","description":"one or two sentences to remember"}},"required":["text"]}
        ,
        .danger = .write,
        .run = runMemoryAppend,
        .summary = sumMemory,
    },
    .{
        .name = "skill_load",
        .description = "Load a named skill's full instructions from the harness (skills/<name>.md).",
        .schema_src = \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
        ,
        .danger = .safe,
        .run = runSkillLoad,
        .summary = sumName,
    },
    .{
        .name = "task",
        .description = "Delegate a self-contained subtask to a subagent with fresh context and the same harness. Give complete context in the prompt — it cannot see this conversation. Optional `agent` selects a named persona from sub_agents/<name>/AGENT.md. op=run (default): blocking. op=spawn: async — returns a handle ('<aid>/sub-N'); the report lands in your inbox on completion. op=collect: results (id optional, wait_ms to block). op=list: spawned subagents' status.",
        .schema_src = \\{"type":"object","properties":{"op":{"type":"string","enum":["run","spawn","collect","list"]},"prompt":{"type":"string"},"title":{"type":"string"},"agent":{"type":"string","description":"optional persona name from harness sub_agents/<name>/AGENT.md"},"id":{"type":"string"},"wait_ms":{"type":"integer"}}}
        ,
        .danger = .safe,
        .run = runTask,
        .summary = sumTask,
    },
    .{
        .name = "coord",
        .description = "Coordinate with sibling agents in this shared session (threads + subagents). op: claim(pat=workspace glob you will write — enforced, persists until rel/~10m) · rel(pat? omit=all) · status (agents/notes/claims) · send(to=aid|'*', m) · poll (drain events+inbox; auto-delivered anyway) · note(m=what you're doing). Terse messages only.",
        .schema_src = \\{"type":"object","properties":{"op":{"type":"string","enum":["claim","rel","status","send","poll","note"]},"pat":{"type":"string","description":"workspace-relative glob; comma-separated ok"},"to":{"type":"string"},"m":{"type":"string"}},"required":["op"]}
        ,
        .danger = .safe,
        .run = runCoord,
        .summary = sumCoord,
    },
    .{
        .name = "memory",
        .description = "Long-term memory (supermemory). op: remember(m=fact/preference/decision to keep — instant) · recall(q=query, n?=limit default 6) · profile (who is this workspace/user) · forget(ids=comma-sep memory ids) · forget_q(q=what to forget, apply=true to confirm; previews first). Recall runs automatically per prompt; use remember for durable facts the turn doesn't capture.",
        .schema_src = \\{"type":"object","properties":{"op":{"type":"string","enum":["remember","recall","profile","forget","forget_q"]},"q":{"type":"string"},"m":{"type":"string"},"ids":{"type":"string"},"n":{"type":"integer"},"apply":{"type":"boolean"}},"required":["op"]}
        ,
        .danger = .safe,
        .run = runMemory,
        .summary = sumMem,
    },
    .{
        .name = "goal",
        .description = "Persistent objective kept in context until done — the harness re-surfaces it every turn. op: set(objective=what you must fully accomplish) · get · complete (objective actually achieved — required to end it) · clear. Set one for long multi-turn work; complete it honestly.",
        .schema_src = \\{"type":"object","properties":{"op":{"type":"string","enum":["set","get","complete","clear"]},"objective":{"type":"string"}},"required":["op"]}
        ,
        .danger = .safe,
        .run = runGoal,
        .summary = sumMem,
    },
    .{
        .name = "refine",
        .description = "Schedule continual-harness refinement: a separate model pass reviews this trajectory and applies small evidence-backed edits to harness files (memory.md, lessons.md, guardrails.md, tool_guidance.md, skills/, sub_agents/). Runs at turn end; result arrives in the next sync. Use after a repeated failure, reusable tactic, or policy worth persisting — not for one-off noise. global=true targets the global (cross-workspace) harness.",
        .schema_src = \\{"type":"object","properties":{"instructions":{"type":"string"},"global":{"type":"boolean"}}}
        ,
        .danger = .safe,
        .run = runRefine,
        .summary = sumMem,
    },
    .{
        .name = "traces",
        .description = "Query this harness's own telemetry — every turn's tool calls, skills, token usage and self-check verdicts across sessions. op: stats (aggregate summary + median tokens/answer) · failures (recent verified failures + gaps) · space (3D projection of all turns: x=tool calls, y=tokens out, z=skills, with failure clusters — navigate it to find patterns) · dataset (export ok turns as an SFT jsonl, returns path+count).",
        .schema_src = \\{"type":"object","properties":{"op":{"type":"string","enum":["stats","failures","space","dataset"]},"path":{"type":"string","description":"dataset: output jsonl path (default ~/.xera-rsi/train/data/)"}},"required":["op"]}
        ,
        .danger = .safe,
        .run = runTraces,
        .summary = sumMem,
    },
    .{
        .name = "train",
        .description = "Model training pipeline — no code needed. op: dataset (build SFT jsonl from verified-good traces) · check (audit a jsonl: duplicates, echo/leak rows, degenerate rows, contamination vs optional ref file) · clean (drop flagged rows → clean copy) · hf (search HuggingFace datasets, q=query) · pull (download HF dataset, id=repo id) · presets (list launch configs) · launch (start an ETM evict-train-merge run: path=dataset or auto-build, preset=quick|standard|deep, focus=what the data teaches, epochs/lr/rank overrides) · status (tail the running/last training log). Typical flow: traces{op:dataset} → train{op:check} → train{op:launch,preset:standard}.",
        .schema_src = \\{"type":"object","properties":{"op":{"type":"string","enum":["dataset","check","clean","hf","pull","presets","launch","status"]},"path":{"type":"string","description":"dataset jsonl (check/clean/launch)"},"out":{"type":"string","description":"clean output path (default <path>.clean.jsonl)"},"ref":{"type":"string","description":"reference jsonl of prompts that must NOT appear (contamination check)"},"q":{"type":"string","description":"hf: search query"},"id":{"type":"string","description":"pull: HF repo id e.g. org/name"},"preset":{"type":"string","enum":["quick","standard","deep"]},"focus":{"type":"string","description":"launch: what the dataset teaches"},"epochs":{"type":"integer"},"lr":{"type":"number"},"rank":{"type":"integer"}},"required":["op"]}
        ,
        .danger = .shell,
        .run = runTrain,
        .summary = sumTrain,
    },
    .{
        .name = "harness",
        .description = "Edit this agent's own harness directly — no code needed, every write auto-snapshots for rollback. op: list (files, skills, agents, validation problems) · read (file=harness-relative path e.g. lessons.md, skills/x/SKILL.md) · append (file + content, adds a dated entry) · write (file + content, replaces) · delete (file) · skill_add (name + desc + content → skills/<name>/SKILL.md with proper frontmatter) · agent_add (name + desc + content → sub_agents/<name>/AGENT.md) · history (recent refinement log) · snapshots (rollback points) · rollback (restore a snapshot, ts= timestamp from snapshots, default newest). Targets: memory.md lessons.md guardrails.md tool_guidance.md system_prompt.md subagents.md skills/… sub_agents/…",
        .schema_src = \\{"type":"object","properties":{"op":{"type":"string","enum":["list","read","append","write","delete","skill_add","agent_add","history","snapshots","rollback"]},"file":{"type":"string","description":"harness-relative path e.g. lessons.md or skills/x/SKILL.md"},"name":{"type":"string","description":"skill_add/agent_add: component name"},"desc":{"type":"string","description":"skill_add/agent_add: frontmatter description (required for triggering)"},"content":{"type":"string"},"ts":{"type":"integer","description":"rollback: snapshot timestamp (default: newest)"}},"required":["op"]}
        ,
        .danger = .write,
        .run = runHarnessTool,
        .summary = sumHarness,
    },
};

pub fn schemas(alloc: std.mem.Allocator) ![]llm.ToolSchema {
    var out = try alloc.alloc(llm.ToolSchema, tools.len);
    for (tools, 0..) |t, i| {
        out[i] = .{
            .name = t.name,
            .description = t.description,
            .params = try util.jparse(alloc, t.schema_src),
        };
    }
    return out;
}

pub fn find(name: []const u8) ?*const Tool {
    for (&tools) |*t| {
        if (std.mem.eql(u8, t.name, name)) return t;
    }
    return null;
}

test "all tool schemas parse as JSON" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const s = try schemas(arena.allocator());
    try std.testing.expect(s.len == tools.len);
    try std.testing.expect(find("train") != null);
    try std.testing.expect(find("harness") != null);
    try std.testing.expect(find("traces") != null);
}

// ---------- helpers ----------

fn argStr(args: std.json.Value, key: []const u8) ?[]const u8 {
    return util.jstr(args, key);
}

fn reqPath(ctx: *ToolCtx, args: std.json.Value) ![]u8 {
    const p = argStr(args, "path") orelse return error.MissingPath;
    return util.resolvePath(ctx.scratch, ctx.cwd, p);
}

fn writeGuard(ctx: *ToolCtx, abs_path: []const u8) !void {
    if (!util.pathWithin(ctx.workspace, abs_path)) return error.OutsideWorkspace;
}

/// workspace-relative '/'-normalized form of an abs path; null when outside.
fn relPath(ctx: *ToolCtx, abs: []const u8) ?[]const u8 {
    if (!util.pathWithin(ctx.workspace, abs)) return null;
    const ws = ctx.workspace;
    const rel = abs[@min(ws.len, abs.len)..];
    return std.mem.trimLeft(u8, rel, "/");
}

/// Hard enforcement: produce an error string when `abs` is claimed by
/// another agent. Returns null when the write may proceed.
fn claimConflict(ctx: *ToolCtx, abs: []const u8) ?[]const u8 {
    const co = ctx.coord orelse return null;
    const rel = relPath(ctx, abs) orelse return null;
    const holder = co.conflict(ctx.aid, rel) orelse return null;
    if (ctx.livelog) |lv| lv.coordEvent("blocked", rel, ctx.depth);
    return std.fmt.allocPrint(ctx.alloc,
        "error: '{s}' is claimed by agent '{s}' — coord send to negotiate, wait for their turn (claims auto-release), or work elsewhere",
        .{ rel, holder }) catch null;
}

/// Emit a feed event for tools that mutate the workspace, so sibling
/// agents see "who touched what" for free.
pub fn emitWriteEvent(ctx: *ToolCtx, name: []const u8, args: std.json.Value, out: []const u8) void {
    const co = ctx.coord orelse return;
    if (std.mem.startsWith(u8, out, "error:")) return;
    var buf: [512]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    if (std.mem.eql(u8, name, "write_file") or std.mem.eql(u8, name, "edit_file") or
        std.mem.eql(u8, name, "delete") or std.mem.eql(u8, name, "make_dir"))
    {
        const p = argStr(args, "path") orelse return;
        const abs = util.resolvePath(fba.allocator(), ctx.cwd, p) catch return;
        const rel = relPath(ctx, abs) orelse p;
        co.emit("w", ctx.aid, rel);
    } else if (std.mem.eql(u8, name, "run_shell")) {
        const c = argStr(args, "command") orelse return;
        co.emit("$", ctx.aid, util.firstLine(c));
    }
}

fn sumPath(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    const p = argStr(args, "path") orelse return "?";
    _ = ctx;
    return p;
}

fn sumPattern(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    _ = ctx;
    const p = argStr(args, "pattern") orelse return "?";
    return p;
}

fn sumCommand(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    _ = ctx;
    const p = argStr(args, "command") orelse return "?";
    return util.firstLine(p);
}

fn sumMemory(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    _ = ctx;
    const p = argStr(args, "text") orelse return "?";
    return util.firstLine(p);
}

fn sumName(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    _ = ctx;
    return argStr(args, "name") orelse "?";
}

fn sumTask(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    _ = ctx;
    if (argStr(args, "title")) |t| return t;
    const p = argStr(args, "prompt") orelse return "?";
    return util.firstLine(p);
}

fn sumCoord(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    const op = argStr(args, "op") orelse return "?";
    const extra = argStr(args, "pat") orelse argStr(args, "to") orelse "";
    return std.fmt.allocPrint(ctx.scratch, "{s} {s}", .{ op, extra }) catch op;
}

fn sumMem(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    const op = argStr(args, "op") orelse return "?";
    const extra = argStr(args, "q") orelse argStr(args, "m") orelse argStr(args, "ids") orelse
        argStr(args, "objective") orelse argStr(args, "instructions") orelse "";
    const short = util.truncateMiddle(ctx.scratch, extra, 60) catch extra;
    return std.fmt.allocPrint(ctx.scratch, "{s} {s}", .{ op, short }) catch op;
}

fn sumTrain(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    const op = argStr(args, "op") orelse return "?";
    const extra = argStr(args, "path") orelse argStr(args, "q") orelse
        argStr(args, "id") orelse argStr(args, "preset") orelse "";
    const short = util.truncateMiddle(ctx.scratch, extra, 60) catch extra;
    return std.fmt.allocPrint(ctx.scratch, "{s} {s}", .{ op, short }) catch op;
}

fn sumHarness(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    const op = argStr(args, "op") orelse return "?";
    const extra = argStr(args, "file") orelse argStr(args, "name") orelse "";
    const short = util.truncateMiddle(ctx.scratch, extra, 60) catch extra;
    return std.fmt.allocPrint(ctx.scratch, "{s} {s}", .{ op, short }) catch op;
}

fn sumUrl(ctx: *ToolCtx, args: std.json.Value) []const u8 {
    _ = ctx;
    return argStr(args, "url") orelse "?";
}

const skip_dirs = [_][]const u8{
    ".git",         ".hg",          ".svn",        "node_modules", ".zig-cache",
    "zig-cache",    "zig-out",      "target",      ".venv",        "venv",
    "__pycache__",  "dist",         "build",       ".idea",        ".vs",
    ".xera-rsi", "runs",
};

fn skipDir(name: []const u8) bool {
    for (skip_dirs) |d| if (std.mem.eql(u8, name, d)) return true;
    return false;
}

// ---------- implementations ----------

fn runReadFile(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const abs = try reqPath(ctx, args);
    const f = std.fs.cwd().openFile(abs, .{}) catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: cannot open {s}: {s}", .{ abs, @errorName(e) });
    };
    defer f.close();
    const raw = f.readToEndAlloc(ctx.scratch, 8 << 20) catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: cannot read {s}: {s}", .{ abs, @errorName(e) });
    };

    const offset = util.jint(args, "offset") orelse 0;
    const limit = util.jint(args, "limit") orelse 0;
    var out = std.ArrayList(u8).init(ctx.scratch);
    var total_lines: usize = 1;
    for (raw) |c| {
        if (c == '\n') total_lines += 1;
    }
    if (offset > 0 or limit > 0) {
        var it = std.mem.splitScalar(u8, raw, '\n');
        var ln: usize = 1;
        var emitted: usize = 0;
        const start: usize = if (offset > 0) @intCast(offset) else 1;
        const max: usize = if (limit > 0) @intCast(limit) else 0;
        while (it.next()) |line| : (ln += 1) {
            if (ln < start) continue;
            if (max > 0 and emitted >= max) break;
            try out.appendSlice(line);
            try out.append('\n');
            emitted += 1;
        }
        const end = if (emitted == 0) start else start + emitted - 1;
        const hdr = try std.fmt.allocPrint(ctx.scratch, "[lines {d}-{d} of {d}]\n", .{
            start, end, total_lines,
        });
        return std.fmt.allocPrint(ctx.alloc, "{s}{s}", .{ hdr, out.items });
    }
    const truncated = try util.truncateMiddle(ctx.scratch, raw, 100_000);
    return std.fmt.allocPrint(ctx.alloc, "[{d} bytes, {d} lines]\n{s}", .{
        raw.len, total_lines, truncated,
    });
}

fn runWriteFile(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const abs = try reqPath(ctx, args);
    const content = argStr(args, "content") orelse return "error: missing content";
    writeGuard(ctx, abs) catch {
        return std.fmt.allocPrint(ctx.alloc, "error: {s} is outside the workspace ({s}); denied", .{ abs, ctx.workspace });
    };
    if (claimConflict(ctx, abs)) |msg| return msg;
    if (std.fs.path.dirname(abs)) |parent| try std.fs.cwd().makePath(parent);
    const f = try std.fs.cwd().createFile(abs, .{});
    defer f.close();
    try f.writeAll(content);
    return std.fmt.allocPrint(ctx.alloc, "wrote {d} bytes to {s}", .{ content.len, abs });
}

fn runEditFile(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const abs = try reqPath(ctx, args);
    const old = argStr(args, "old_string") orelse return "error: missing old_string";
    const new = argStr(args, "new_string") orelse return "error: missing new_string";
    const replace_all = util.jbool(args, "replace_all") orelse false;
    if (old.len == 0) return "error: old_string is empty";
    writeGuard(ctx, abs) catch {
        return std.fmt.allocPrint(ctx.alloc, "error: {s} is outside the workspace ({s}); denied", .{ abs, ctx.workspace });
    };
    if (claimConflict(ctx, abs)) |msg| return msg;

    const raw = std.fs.cwd().readFileAlloc(ctx.scratch, abs, 64 << 20) catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: cannot read {s}: {s}", .{ abs, @errorName(e) });
    };

    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, raw, pos, old)) |idx| {
        count += 1;
        pos = idx + old.len;
    }
    if (count == 0) {
        return std.fmt.allocPrint(ctx.alloc, "error: old_string not found in {s}", .{abs});
    }
    if (count > 1 and !replace_all) {
        return std.fmt.allocPrint(ctx.alloc, "error: old_string matches {d} places in {s}; make it unique or set replace_all", .{ count, abs });
    }

    var out = std.ArrayList(u8).init(ctx.scratch);
    if (replace_all) {
        var it = std.mem.splitSequence(u8, raw, old);
        var first = true;
        while (it.next()) |part| {
            if (!first) try out.appendSlice(new);
            try out.appendSlice(part);
            first = false;
        }
    } else {
        const idx = std.mem.indexOf(u8, raw, old).?;
        try out.appendSlice(raw[0..idx]);
        try out.appendSlice(new);
        try out.appendSlice(raw[idx + old.len ..]);
    }
    const f = try std.fs.cwd().createFile(abs, .{});
    defer f.close();
    try f.writeAll(out.items);
    const n_rep: usize = if (replace_all) count else 1;
    return std.fmt.allocPrint(ctx.alloc, "edited {s}: replaced {d} occurrence(s)", .{ abs, n_rep });
}

fn runListDir(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const p = argStr(args, "path") orelse ".";
    const abs = try util.resolvePath(ctx.scratch, ctx.cwd, p);
    var d = std.fs.cwd().openDir(abs, .{ .iterate = true }) catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: cannot open dir {s}: {s}", .{ abs, @errorName(e) });
    };
    defer d.close();
    var out = std.ArrayList(u8).init(ctx.scratch);
    const w = out.writer();
    var it = d.iterate();
    var n: usize = 0;
    while (try it.next()) |e| {
        if (n >= 2000) {
            try w.writeAll("… (truncated)\n");
            break;
        }
        const kind: []const u8 = switch (e.kind) {
            .directory => "dir ",
            .file => "file",
            .sym_link => "link",
            else => "??? ",
        };
        try w.print("{s}  {s}\n", .{ kind, e.name });
        n += 1;
    }
    return std.fmt.allocPrint(ctx.alloc, "[{s}] {d} entries\n{s}", .{ abs, n, out.items });
}

fn runGlob(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const pattern = argStr(args, "pattern") orelse return "error: missing pattern";
    const base = argStr(args, "path") orelse ".";
    const abs = try util.resolvePath(ctx.scratch, ctx.cwd, base);
    var d = std.fs.cwd().openDir(abs, .{ .iterate = true }) catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: cannot open {s}: {s}", .{ abs, @errorName(e) });
    };
    defer d.close();

    const pat = util.normalizePath(ctx.scratch, pattern);
    var out = std.ArrayList(u8).init(ctx.scratch);
    var walker = try d.walk(ctx.scratch);
    defer walker.deinit();
    var n: usize = 0;
    while (try walker.next()) |e| {
        if (e.kind == .directory) continue;
        const rel = util.normalizePath(ctx.scratch, e.path);
        // prune skip dirs cheaply: check each segment
        var skip = false;
        var sit = std.mem.tokenizeScalar(u8, rel, '/');
        while (sit.next()) |seg| {
            if (skipDir(seg)) {
                skip = true;
                break;
            }
        }
        if (skip) continue;
        if (util.globMatch(pat, rel)) {
            try out.appendSlice(rel);
            try out.append('\n');
            n += 1;
            if (n >= 1000) {
                try out.appendSlice("… (truncated)\n");
                break;
            }
        }
    }
    if (n == 0) return std.fmt.allocPrint(ctx.alloc, "no matches for '{s}' under {s}", .{ pattern, abs });
    return std.fmt.allocPrint(ctx.alloc, "{d} match(es):\n{s}", .{ n, out.items });
}

fn runGrep(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const pattern = argStr(args, "pattern") orelse return "error: missing pattern";
    const base = argStr(args, "path") orelse ".";
    const file_glob = argStr(args, "file_glob");
    const icase = util.jbool(args, "ignore_case") orelse false;
    const abs = try util.resolvePath(ctx.scratch, ctx.cwd, base);

    var out = std.ArrayList(u8).init(ctx.scratch);
    const w = out.writer();
    var matches: usize = 0;
    var files_hit: usize = 0;

    const st = std.fs.cwd().statFile(abs) catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: cannot stat {s}: {s}", .{ abs, @errorName(e) });
    };
    if (st.kind == .file) {
        try grepFile(ctx, abs, std.fs.path.basename(abs), pattern, file_glob, icase, w, &matches);
        files_hit = 1;
    } else {
        var d = std.fs.cwd().openDir(abs, .{ .iterate = true }) catch |e| {
            return std.fmt.allocPrint(ctx.alloc, "error: cannot open {s}: {s}", .{ abs, @errorName(e) });
        };
        defer d.close();
        var walker = try d.walk(ctx.scratch);
        defer walker.deinit();
        while (try walker.next()) |e| {
            if (e.kind != .file) continue;
            const rel = util.normalizePath(ctx.scratch, e.path);
            var skip = false;
            var sit = std.mem.tokenizeScalar(u8, rel, '/');
            while (sit.next()) |seg| {
                if (skipDir(seg)) {
                    skip = true;
                    break;
                }
            }
            if (skip) continue;
            const full = try std.fs.path.join(ctx.scratch, &.{ abs, e.path });
            const before = matches;
            try grepFile(ctx, full, rel, pattern, file_glob, icase, w, &matches);
            if (matches > before) files_hit += 1;
            if (matches >= 300) {
                try w.writeAll("… (match limit reached)\n");
                break;
            }
        }
    }
    if (matches == 0) {
        return std.fmt.allocPrint(ctx.alloc, "no matches for '{s}' under {s}", .{ pattern, abs });
    }
    return std.fmt.allocPrint(ctx.alloc, "{d} match(es) in {d} file(s):\n{s}", .{ matches, files_hit, out.items });
}

fn grepFile(
    ctx: *ToolCtx,
    abs: []const u8,
    rel: []const u8,
    pattern: []const u8,
    file_glob: ?[]const u8,
    icase: bool,
    w: anytype,
    matches: *usize,
) !void {
    if (file_glob) |g| {
        if (!util.globMatch(g, std.fs.path.basename(rel)) and !util.globMatch(g, rel)) return;
    }
    const raw = std.fs.cwd().readFileAlloc(ctx.scratch, abs, 4 << 20) catch return;
    if (raw.len == 0) return;
    // crude binary check
    const probe = raw[0..@min(raw.len, 8192)];
    if (std.mem.indexOfScalar(u8, probe, 0) != null) return;

    var it = std.mem.splitScalar(u8, raw, '\n');
    var ln: usize = 1;
    while (it.next()) |line| : (ln += 1) {
        const found = if (icase)
            std.ascii.indexOfIgnoreCase(line, pattern) != null
        else
            std.mem.indexOf(u8, line, pattern) != null;
        if (!found) continue;
        var l = line;
        if (l.len > 0 and l[l.len - 1] == '\r') l = l[0 .. l.len - 1];
        if (l.len > 400) l = l[0..400];
        w.print("{s}:{d}: {s}\n", .{ rel, ln, l }) catch return;
        matches.* += 1;
        if (matches.* >= 300) return;
    }
}

// ---------- shell ----------

const PipeRead = struct {
    file: std.fs.File,
    list: *std.ArrayList(u8),
    max: usize,
    /// optional lock for live tails while a background process streams
    mu: ?*std.Thread.Mutex = null,

    fn run(self: PipeRead) void {
        const mu = self.mu orelse {
            self.file.reader().readAllArrayList(self.list, self.max) catch {};
            return;
        };
        // chunk loop: lock per chunk so mid-run `out` snapshots stay safe.
        // Past the cap we keep draining so the child never blocks on a
        // full pipe — output just stops accumulating.
        var buf: [8192]u8 = undefined;
        while (true) {
            const n = self.file.read(&buf) catch break;
            if (n == 0) break;
            mu.lock();
            if (self.list.items.len < self.max) {
                const room = self.max - self.list.items.len;
                self.list.appendSlice(buf[0..@min(room, n)]) catch {};
            }
            mu.unlock();
        }
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
        if (!self.done.load(.acquire)) {
            _ = self.child.kill() catch return;
        }
    }
};

fn buildArgv(alloc: std.mem.Allocator, ctx: *ToolCtx, command: []const u8) ![][]const u8 {
    const builtin = @import("builtin");
    var argv = std.ArrayList([]const u8).init(alloc);
    if (builtin.os.tag == .windows) {
        if (ctx.cfg.shell) |sh| {
            try argv.append(sh);
            try argv.append(command);
        } else {
            try argv.append("cmd.exe");
            try argv.append("/c");
            try argv.append(command);
        }
    } else {
        const sh = ctx.cfg.shell orelse "/bin/sh";
        try argv.append(sh);
        try argv.append("-c");
        try argv.append(command);
    }
    return argv.toOwnedSlice();
}

/// Detached worker for run_shell{background:true}: spawns, streams
/// output into the handle, publishes the result, then pushes a
/// [shell-done] notice the agent sees at the next sync boundary.
const BgJob = struct {
    sh: *BgShell,
    reg: *Shells,
    argv: [][]const u8,
    cwd: []const u8,
};

fn bgShellMain(job: *BgJob) void {
    const alloc = job.reg.alloc;
    var child = std.process.Child.init(job.argv, alloc);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;
    child.cwd = job.cwd;
    child.spawn() catch |e| {
        job.sh.mu.lock();
        job.sh.done = true;
        job.sh.code = -1;
        job.sh.err.writer().print("spawn failed: {s}", .{@errorName(e)}) catch {};
        job.sh.mu.unlock();
        bgShellNotice(job);
        return;
    };
    job.sh.child = &child;
    const t_out = std.Thread.spawn(.{}, PipeRead.run, .{PipeRead{
        .file = child.stdout.?, .list = &job.sh.out, .max = 4 << 20, .mu = &job.sh.mu,
    }}) catch return;
    const t_err = std.Thread.spawn(.{}, PipeRead.run, .{PipeRead{
        .file = child.stderr.?, .list = &job.sh.err, .max = 4 << 20, .mu = &job.sh.mu,
    }}) catch return;
    const term = child.wait() catch {
        job.sh.mu.lock();
        job.sh.done = true;
        job.sh.code = -1;
        job.sh.mu.unlock();
        bgShellNotice(job);
        return;
    };
    t_out.join();
    t_err.join();
    job.sh.mu.lock();
    job.sh.done = true;
    job.sh.code = switch (term) {
        .Exited => |c| @intCast(c),
        else => -1,
    };
    job.sh.mu.unlock();
    bgShellNotice(job);
}

fn bgShellNotice(job: *BgJob) void {
    const reg = job.reg;
    const nq = reg.notices orelse return;
    const tail = blk: {
        job.sh.mu.lock();
        defer job.sh.mu.unlock();
        const o = job.sh.out.items;
        const t = if (o.len > 300) o[o.len - 300 ..] else o;
        break :blk util.firstLine(t);
    };
    const msg = std.fmt.allocPrint(reg.alloc, "[shell-done {s} exit {d}] {s}", .{
        job.sh.id, job.sh.code, tail,
    }) catch return;
    nq.push(reg.alloc, msg);
}

fn runShell(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    // ---- manage an existing background shell ----
    if (argStr(args, "id")) |id| {
        const reg = ctx.shells orelse return "error: no shell registry on this agent";
        reg.mu.lock();
        const sh = reg.map.get(id);
        reg.mu.unlock();
        const s = sh orelse return std.fmt.allocPrint(ctx.alloc, "error: no shell '{s}'", .{id});

        var kill = false;
        if (util.jget(args, "kill")) |k| {
            if (k == .bool) kill = k.bool;
        }
        if (kill) {
            s.mu.lock();
            const running = !s.done;
            const c = s.child;
            s.mu.unlock();
            if (running and c != null) _ = c.?.kill() catch {};
            return std.fmt.allocPrint(ctx.alloc, "{s} kill requested", .{id});
        }

        const wait_ms: i64 = @intCast(@max(0, util.jint(args, "wait_ms") orelse 0));
        const deadline = std.time.milliTimestamp() + wait_ms;
        while (true) {
            s.mu.lock();
            const d = s.done;
            s.mu.unlock();
            if (d or std.time.milliTimestamp() >= deadline) break;
            std.time.sleep(25 * std.time.ns_per_ms);
        }

        s.mu.lock();
        defer s.mu.unlock();
        const age_s = @divTrunc(std.time.milliTimestamp() - s.started_ms, 1000);
        const out_t = try util.truncateMiddle(ctx.scratch, s.out.items, 20_000);
        const err_t = try util.truncateMiddle(ctx.scratch, s.err.items, 8_000);
        if (!s.done) {
            return std.fmt.allocPrint(ctx.alloc, "{s} still running ({d}s)\n--- stdout ---\n{s}\n--- stderr ---\n{s}", .{ s.id, age_s, out_t, err_t });
        }
        return std.fmt.allocPrint(ctx.alloc, "{s} exit {d} ({d}s)\n--- stdout ---\n{s}\n--- stderr ---\n{s}", .{ s.id, s.code, age_s, out_t, err_t });
    }

    const command = argStr(args, "command") orelse return "error: missing command (or pass id)";
    const timeout_ms: u64 = @intCast(@max(1000, util.jint(args, "timeout_ms") orelse 120_000));
    var run_cwd = ctx.cwd;
    if (argStr(args, "cwd")) |c| {
        run_cwd = try util.resolvePath(ctx.scratch, ctx.cwd, c);
    }

    var background = false;
    if (util.jget(args, "background")) |b| {
        if (b == .bool) background = b.bool;
    }

    // ---- background: detached process + handle ----
    if (background) {
        const reg = ctx.shells orelse return "error: background shells unavailable on this agent";
        const a = reg.alloc; // page_allocator — thread-safe, session lifetime
        const sh = try a.create(BgShell);
        const command_copy = try a.dupe(u8, command);
        sh.* = .{
            .id = undefined,
            .command = command_copy,
            .out = std.ArrayList(u8).init(a),
            .err = std.ArrayList(u8).init(a),
            .started_ms = std.time.milliTimestamp(),
        };
        reg.mu.lock();
        sh.id = try std.fmt.allocPrint(a, "sh-{d}", .{reg.next});
        reg.next += 1;
        try reg.map.put(try a.dupe(u8, sh.id), sh);
        reg.mu.unlock();

        const job = try a.create(BgJob);
        job.* = .{
            .sh = sh,
            .reg = reg,
            .argv = try buildArgv(a, ctx, command_copy),
            .cwd = try a.dupe(u8, run_cwd),
        };
        const th = std.Thread.spawn(.{}, bgShellMain, .{job}) catch |e| {
            return std.fmt.allocPrint(ctx.alloc, "error: spawn thread failed: {s}", .{@errorName(e)});
        };
        th.detach();
        if (ctx.livelog) |lv| lv.event("shell", "spawn", util.firstLine(command), ctx.depth);
        if (ctx.coord) |co| co.emit("S", ctx.aid, std.fmt.allocPrint(ctx.scratch, "{s} bg {s}", .{ sh.id, util.firstLine(command) }) catch sh.id);
        return std.fmt.allocPrint(ctx.alloc, "{s} running (background). You'll get [shell-done] on exit; check with run_shell{{id:\"{s}\"}} or wait.", .{ sh.id, sh.id });
    }

    // ---- foreground ----
    const argv = try buildArgv(ctx.scratch, ctx, command);
    var child = std.process.Child.init(argv, ctx.scratch);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;
    child.cwd = run_cwd;
    child.spawn() catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: spawn failed: {s}", .{@errorName(e)});
    };

    var out_list = std.ArrayList(u8).init(ctx.scratch);
    var err_list = std.ArrayList(u8).init(ctx.scratch);

    var done = std.atomic.Value(bool).init(false);
    const watch = try std.Thread.spawn(.{}, KillWatch.run, .{KillWatch{
        .child = &child,
        .done = &done,
        .timeout_ms = timeout_ms,
    }});
    const t_out = try std.Thread.spawn(.{}, PipeRead.run, .{PipeRead{
        .file = child.stdout.?,
        .list = &out_list,
        .max = 4 << 20,
    }});
    const t_err = try std.Thread.spawn(.{}, PipeRead.run, .{PipeRead{
        .file = child.stderr.?,
        .list = &err_list,
        .max = 4 << 20,
    }});

    const term = child.wait() catch |e| {
        done.store(true, .release);
        watch.join();
        t_out.join();
        t_err.join();
        return std.fmt.allocPrint(ctx.alloc, "error: wait failed: {s}", .{@errorName(e)});
    };
    done.store(true, .release);
    watch.join();
    t_out.join();
    t_err.join();

    const code: i64 = switch (term) {
        .Exited => |c| @intCast(c),
        else => -1,
    };
    const stdout_t = try util.truncateMiddle(ctx.scratch, out_list.items, 30_000);
    const stderr_t = try util.truncateMiddle(ctx.scratch, err_list.items, 15_000);
    return std.fmt.allocPrint(ctx.alloc, "$ {s}\nexit: {d}\n--- stdout ---\n{s}\n--- stderr ---\n{s}", .{
        command, code, stdout_t, stderr_t,
    });
}

fn runMemoryAppend(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const text = argStr(args, "text") orelse return "error: missing text";
    try ctx.h.ensure();
    try ctx.h.applyEdit(ctx.h.memoryFile(), "append", text);
    return std.fmt.allocPrint(ctx.alloc, "memory updated: {s}", .{util.firstLine(text)});
}

fn runSkillLoad(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const name = argStr(args, "name") orelse return "error: missing name";
    const body = ctx.h.skillBody(name) catch |e| {
        const names = ctx.h.listSkills() catch &[_]@import("harness.zig").NamedItem{};
        var hint = std.ArrayList(u8).init(ctx.scratch);
        for (names) |n| {
            if (hint.items.len > 0) try hint.appendSlice(", ");
            try hint.appendSlice(n.name);
        }
        return std.fmt.allocPrint(ctx.alloc, "error: skill '{s}' not found ({s}). available: {s}", .{ name, @errorName(e), hint.items });
    };
    return std.fmt.allocPrint(ctx.alloc, "<skill name=\"{s}\">\n{s}\n</skill>", .{ name, body });
}

fn runTask(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const op = argStr(args, "op") orelse "run";

    if (std.mem.eql(u8, op, "run")) {
        if (ctx.depth >= 2) return "error: subagent depth limit reached";
        const prompt = argStr(args, "prompt") orelse return "error: missing prompt";
        const agent_name = argStr(args, "agent");
        const runner = ctx.subagent_run orelse return "error: subagents unavailable";
        return runner(ctx, prompt, agent_name);
    }

    if (std.mem.eql(u8, op, "spawn")) {
        if (ctx.depth >= 2) return "error: subagent depth limit reached";
        const prompt = argStr(args, "prompt") orelse return "error: spawn needs prompt";
        const agent_name = argStr(args, "agent");
        const title = argStr(args, "title") orelse util.firstLine(prompt);
        const reg = ctx.subs orelse return "error: async subagents unavailable on this agent";
        const runner = ctx.subagent_run orelse return "error: subagents unavailable";
        const a = reg.alloc; // page_allocator — safe after the caller returns

        const sub = try a.create(BgSub);
        const parent = if (ctx.aid.len > 0) ctx.aid else "agent";
        sub.* = .{
            .aid = undefined,
            .title = try a.dupe(u8, title),
            .started_ms = std.time.milliTimestamp(),
        };

        // heap-copy the ctx: the parent's stack ctx dies when the call
        // returns. alloc swaps to the registry's (a depth>0 ctx.alloc is
        // an arena that dies with the parent's run — page survives).
        const tctx = try a.create(ToolCtx);
        tctx.* = ctx.*;
        tctx.alloc = a;
        tctx.scratch = a;

        reg.mu.lock();
        sub.aid = try std.fmt.allocPrint(a, "{s}/sub-{d}", .{ parent, reg.next });
        reg.next += 1;
        try reg.map.put(try a.dupe(u8, sub.aid), sub);
        reg.mu.unlock();
        tctx.sub_aid = sub.aid;
        // sub-subs would outlive a transient parent arena — keep async
        // delegation to channel agents only
        tctx.subs = null;

        const job = try a.create(SubJob);
        job.* = .{ .sub = sub, .tctx = tctx, .prompt = try a.dupe(u8, prompt), .agent = agent_name, .runner = runner };
        const th = std.Thread.spawn(.{}, bgSubMain, .{job}) catch |e| {
            return std.fmt.allocPrint(ctx.alloc, "error: spawn failed: {s}", .{@errorName(e)});
        };
        th.detach();
        if (ctx.livelog) |lv| lv.subagent(agent_name orelse "sub", util.firstLine(prompt), ctx.depth);
        if (ctx.coord) |co| co.emit("S", ctx.aid, std.fmt.allocPrint(ctx.scratch, "spawn {s}", .{sub.aid}) catch sub.aid);
        return std.fmt.allocPrint(ctx.alloc, "spawned {s} — result arrives in your inbox; check with task{{op:\"collect\",id:\"{s}\"}}", .{ sub.aid, sub.aid });
    }

    if (std.mem.eql(u8, op, "list") or std.mem.eql(u8, op, "collect")) {
        const reg = ctx.subs orelse return "error: async subagents unavailable on this agent";
        const want_id = argStr(args, "id");
        const wait_ms: i64 = if (std.mem.eql(u8, op, "list")) 0 else @intCast(@max(0, util.jint(args, "wait_ms") orelse 0));
        const deadline = std.time.milliTimestamp() + wait_ms;

        var out = std.ArrayList(u8).init(ctx.scratch);
        while (true) {
            out.clearRetainingCapacity();
            var pending: usize = 0;
            var matched: usize = 0;
            reg.mu.lock();
            var it = reg.map.iterator();
            while (it.next()) |kv| {
                const s = kv.value_ptr.*;
                if (want_id) |w| {
                    if (!std.mem.eql(u8, s.aid, w) and !std.mem.eql(u8, kv.key_ptr.*, w)) continue;
                }
                matched += 1;
                s.mu.lock();
                const status: []const u8 = if (!s.done) "running" else if (s.is_err) "error" else "done";
                if (!s.done) pending += 1;
                try out.writer().print("{s} [{s}] {s}", .{ s.aid, status, s.title });
                if (s.done and s.result.len > 0) {
                    const r = try util.truncateMiddle(ctx.scratch, s.result, 3000);
                    try out.writer().print("\n--- result ---\n{s}\n---\n", .{r});
                }
                s.mu.unlock();
                try out.append('\n');
            }
            reg.mu.unlock();

            if (want_id != null and matched == 0)
                return std.fmt.allocPrint(ctx.alloc, "error: no spawned subagent '{s}'", .{want_id.?});
            if (matched == 0) return "(no spawned subagents)";
            if (pending == 0 or std.time.milliTimestamp() >= deadline) break;
            std.time.sleep(25 * std.time.ns_per_ms);
        }
        return try ctx.alloc.dupe(u8, std.mem.trimRight(u8, out.items, "\n"));
    }

    return std.fmt.allocPrint(ctx.alloc, "error: unknown op '{s}' (run|spawn|collect|list)", .{op});
}

/// Detached worker for task op=spawn: runs the full subagent loop and
/// publishes the report — into the parent's notices and its inbox.
const SubJob = struct {
    sub: *BgSub,
    tctx: *ToolCtx,
    prompt: []const u8,
    agent: ?[]const u8,
    runner: *const fn (ctx: *ToolCtx, prompt: []const u8, agent_name: ?[]const u8) anyerror![]const u8,
};

fn bgSubMain(job: *SubJob) void {
    const a = job.tctx.alloc;
    const res = job.runner(job.tctx, job.prompt, job.agent) catch |e| blk: {
        job.sub.is_err = true;
        break :blk std.fmt.allocPrint(a, "(subagent failed: {s})", .{@errorName(e)}) catch "";
    };
    job.sub.mu.lock();
    job.sub.result = res;
    job.sub.done = true;
    job.sub.mu.unlock();

    const first = util.firstLine(res);
    const tail = if (first.len > 200) first[0..200] else first;
    const msg = std.fmt.allocPrint(a, "[sub-done {s}{s}] {s}", .{
        job.sub.aid, if (job.sub.is_err) " error" else "", tail,
    }) catch return;
    if (job.tctx.notices) |nq| nq.push(a, msg);
    if (job.tctx.coord) |co| {
        co.emit("S", job.sub.aid, "done");
        co.send(job.sub.aid, job.tctx.aid, msg) catch {};
    }
}

fn runGoal(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const gs = ctx.goal_state orelse return "error: goals unavailable on this agent";
    const op = argStr(args, "op") orelse return "error: missing op";

    if (std.mem.eql(u8, op, "set")) {
        const obj = argStr(args, "objective") orelse return "error: set needs objective";
        const copy = try ctx.alloc.dupe(u8, obj);
        gs.mu.lock();
        gs.objective = copy;
        gs.set_ms = std.time.milliTimestamp();
        gs.turns = 0;
        gs.mu.unlock();
        if (ctx.coord) |co| co.setGoal(ctx.aid, obj);
        if (ctx.livelog) |lv| lv.memEvent("goal_set", util.firstLine(obj), ctx.depth);
        return std.fmt.allocPrint(ctx.alloc, "goal set: {s}\n(it re-surfaces every turn until you `goal complete`)", .{obj});
    }
    if (std.mem.eql(u8, op, "get")) {
        gs.mu.lock();
        defer gs.mu.unlock();
        if (gs.objective.len == 0) return "(no active goal)";
        return std.fmt.allocPrint(ctx.alloc, "goal: {s} ({d} turns)", .{ gs.objective, gs.turns });
    }
    if (std.mem.eql(u8, op, "complete") or std.mem.eql(u8, op, "clear")) {
        gs.mu.lock();
        const had = gs.objective.len > 0;
        gs.objective = "";
        gs.turns = 0;
        gs.mu.unlock();
        if (ctx.coord) |co| co.setGoal(ctx.aid, null);
        if (ctx.livelog) |lv| lv.memEvent(if (had) "goal_done" else "goal_clear", "", ctx.depth);
        return if (had)
            (if (std.mem.eql(u8, op, "complete")) "goal completed" else "goal cleared")
        else
            "(no active goal)";
    }
    return std.fmt.allocPrint(ctx.alloc, "error: unknown op '{s}' (set|get|complete|clear)", .{op});
}

fn runRefine(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const req = ctx.refine_req orelse return "error: refinement unavailable on this agent";
    const instr = argStr(args, "instructions") orelse "";
    var glob = false;
    if (util.jget(args, "global")) |g| {
        if (g == .bool) glob = g.bool;
    }
    req.mu.lock();
    req.pending = if (instr.len > 0) try ctx.alloc.dupe(u8, instr) else "";
    req.scope_global = glob;
    req.mu.unlock();
    if (ctx.livelog) |lv| lv.memEvent("refine_sched", util.firstLine(instr), ctx.depth);
    return "refinement scheduled — it runs when this turn ends and applies evidence-backed edits to the harness";
}

// ---------- traces (self-telemetry) ----------

fn runTraces(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    _ = ctx.traces; // traces work even if this agent's store is detached
    const op = argStr(args, "op") orelse return "error: missing op";
    if (std.mem.eql(u8, op, "stats")) {
        return traces_mod.summary(ctx.scratch) catch |e|
            try std.fmt.allocPrint(ctx.scratch, "error: {s}", .{@errorName(e)});
    }
    if (std.mem.eql(u8, op, "failures")) {
        const st = try traces_mod.stats(ctx.scratch);
        if (st.failures.len == 0) return "no verified failures on record";
        var out = std.ArrayList(u8).init(ctx.scratch);
        for (st.failures) |f| {
            try out.writer().print("- \"{s}\" → {s}\n", .{ f.prompt, f.gap });
        }
        return out.items;
    }
    if (std.mem.eql(u8, op, "space")) {
        return traces_mod.spaceJson(ctx.scratch) catch |e|
            try std.fmt.allocPrint(ctx.scratch, "error: {s}", .{@errorName(e)});
    }
    if (std.mem.eql(u8, op, "dataset")) {
        const default_dir = blk: {
            const home = util.homeDir(ctx.scratch) catch break :blk "";
            break :blk std.fs.path.join(ctx.scratch,
                &.{ home, ".xera-rsi", "train", "data" }) catch "";
        };
        const path = argStr(args, "path") orelse
            (try std.fs.path.join(ctx.scratch, &.{ default_dir,
                try std.fmt.allocPrint(ctx.scratch, "traces-{d}.jsonl", .{std.time.timestamp()}) }));
        const n = traces_mod.exportDataset(ctx.scratch, path, true) catch |e|
            return try std.fmt.allocPrint(ctx.scratch, "error: {s}", .{@errorName(e)});
        return try std.fmt.allocPrint(ctx.scratch,
            "dataset written: {s} ({d} samples)", .{ path, n });
    }
    return "error: op must be stats|failures|space|dataset";
}

// ---------- training (dataset → check → launch → status) ----------

/// Resolve a dataset path: absolute stays, relative resolves against
/// ~/.xera-rsi/train/data first (where `dataset` writes), then cwd.
fn trainPath(ctx: *ToolCtx, p: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(p)) return ctx.scratch.dupe(u8, p);
    const root = train_mod.rootDir(ctx.scratch) catch null;
    if (root) |r| {
        const cand = try std.fs.path.join(ctx.scratch, &.{ r, "data", p });
        if (std.fs.cwd().access(cand, .{})) |_| return cand else |_| {}
    }
    return util.resolvePath(ctx.scratch, ctx.cwd, p);
}

fn runTrain(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const a = ctx.scratch;
    const op = argStr(args, "op") orelse return "error: missing op";

    if (std.mem.eql(u8, op, "dataset")) {
        const ds = train_mod.buildDataset(a, "tool") catch |e|
            return try std.fmt.allocPrint(a, "error: {s}", .{@errorName(e)});
        if (ds.samples == 0)
            return "dataset empty — no verified-good turns yet; run some tasks first (or pull a HF dataset with train{op:pull})";
        return try std.fmt.allocPrint(a,
            "dataset written: {s} ({d} samples)\nnext: train{{op:\"check\",path:\"{s}\"}}", .{ ds.path, ds.samples, ds.path });
    }

    if (std.mem.eql(u8, op, "check")) {
        const p = argStr(args, "path") orelse return "error: check needs path";
        const abs = try trainPath(ctx, p);
        const ref = if (argStr(args, "ref")) |rp| try trainPath(ctx, rp) else null;
        return train_mod.checkDataset(a, abs, ref) catch |e|
            try std.fmt.allocPrint(a, "error: {s}", .{@errorName(e)});
    }

    if (std.mem.eql(u8, op, "clean")) {
        const p = argStr(args, "path") orelse return "error: clean needs path";
        const abs = try trainPath(ctx, p);
        const out = argStr(args, "out") orelse
            try std.fmt.allocPrint(a, "{s}.clean.jsonl", .{abs});
        const ref = if (argStr(args, "ref")) |rp| try trainPath(ctx, rp) else null;
        return train_mod.cleanDataset(a, abs, out, ref) catch |e|
            try std.fmt.allocPrint(a, "error: {s}", .{@errorName(e)});
    }

    if (std.mem.eql(u8, op, "hf")) {
        const q = argStr(args, "q") orelse return "error: hf needs q (search query)";
        const hc = ctx.hclient orelse return "error: no http client on this agent";
        const hits = train_mod.hfSearch(a, hc, q, 8) catch |e|
            return try std.fmt.allocPrint(a, "error: {s}", .{@errorName(e)});
        if (hits.len == 0) return "no datasets matched";
        var out = std.ArrayList(u8).init(a);
        for (hits) |hh| {
            try out.writer().print("{s}  ({d} likes, {d} dl)\n", .{ hh.id, hh.likes, hh.downloads });
        }
        try out.appendSlice("pull with: train{op:\"pull\",id:\"<id>\"}");
        return out.items;
    }

    if (std.mem.eql(u8, op, "pull")) {
        const id = argStr(args, "id") orelse return "error: pull needs id (org/name)";
        const hc = ctx.hclient orelse return "error: no http client on this agent";
        const p = train_mod.hfDownload(a, hc, id) catch |e|
            return try std.fmt.allocPrint(a, "error: {s} (dataset has no .jsonl/.json/.csv/.txt file?)", .{@errorName(e)});
        return try std.fmt.allocPrint(a,
            "downloaded: {s}\nnext: train{{op:\"check\",path:\"{s}\"}}", .{ p, p });
    }

    if (std.mem.eql(u8, op, "presets")) {
        var out = std.ArrayList(u8).init(a);
        for (train_mod.presets) |p| {
            try out.writer().print("{s: <9} {s}\n           args: {s}\n", .{ p.name, p.desc, p.args });
        }
        try out.appendSlice("launch with: train{op:\"launch\",preset:\"<name>\",focus:\"<what it teaches>\"}");
        return out.items;
    }

    if (std.mem.eql(u8, op, "launch")) {
        // data: explicit path, or auto-build from traces
        var data: []const u8 = "";
        if (argStr(args, "path")) |p| {
            data = try trainPath(ctx, p);
        } else {
            const ds = train_mod.buildDataset(a, "tool") catch |e|
                return try std.fmt.allocPrint(a, "error: auto-dataset failed ({s}) — pass path= explicitly", .{@errorName(e)});
            if (ds.samples == 0) return "error: no dataset path given and no trace samples to auto-build";
            data = ds.path;
        }
        // args: preset + overrides + focus — the model picks, we compose
        var extra = std.ArrayList(u8).init(a);
        const preset = argStr(args, "preset") orelse "standard";
        const pargs = train_mod.presetArgs(preset) orelse
            return "error: unknown preset (presets|quick|standard|deep)";
        try extra.appendSlice(pargs);
        if (util.jget(args, "epochs")) |e| {
            if (e == .integer) try extra.writer().print(" --epochs {d}", .{e.integer});
        }
        if (util.jget(args, "lr")) |e| {
            if (e == .float) try extra.writer().print(" --lr {d}", .{e.float});
        }
        if (util.jget(args, "rank")) |e| {
            if (e == .integer) try extra.writer().print(" --lora-rank {d}", .{e.integer});
        }
        if (argStr(args, "focus")) |f| {
            try extra.writer().print(" --focus \"{s}\"", .{f});
        }
        const tp = train_mod.launchEtm(a, data, extra.items) catch |e|
            return try std.fmt.allocPrint(a,
                "error: launch failed ({s}) — needs python + the ETM repo (clones on first launch)", .{@errorName(e)});
        return try std.fmt.allocPrint(a,
            "etm launched: preset {s} on {s}\nlog: {s}\nwatch with train{{op:\"status\"}}", .{ preset, data, tp.log_path });
    }

    if (std.mem.eql(u8, op, "status")) {
        const st = train_mod.status(a);
        var out = std.ArrayList(u8).init(a);
        if (st.log_path.len == 0) return "no training run yet";
        try out.writer().print("{s} · {d}s elapsed · exit {d}\nlog: {s}\n", .{
            if (st.running) "running" else "finished", st.elapsed_s, st.code, st.log_path });
        if (st.log_tail.len > 0) try out.writer().print("--- log tail ---\n{s}", .{st.log_tail});
        return out.items;
    }

    return "error: op must be dataset|check|clean|hf|pull|presets|launch|status";
}

// ---------- harness (direct self-editing, auto-snapshotted) ----------

fn runHarnessTool(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const a = ctx.scratch;
    const h = ctx.h;
    const op = argStr(args, "op") orelse return "error: missing op";
    try h.ensure();

    if (std.mem.eql(u8, op, "list")) {
        var out = std.ArrayList(u8).init(a);
        try out.writer().print("harness: {s} ({s})\n", .{ h.dir, if (h.global) "global" else "project" });
        for (harness.component_files) |f| {
            const body = h.read(f);
            try out.writer().print("  {s: <20} {d} b\n", .{ f, body.len });
        }
        const skills = h.listSkills() catch &[_]harness.NamedItem{};
        for (skills) |s| try out.writer().print("  skill {s}{s}\n", .{ s.name,
            if (s.desc.len > 0) try std.fmt.allocPrint(a, " — {s}", .{s.desc}) else "" });
        const agents = h.listAgents() catch &[_]harness.NamedItem{};
        for (agents) |ag| try out.writer().print("  agent {s}{s}\n", .{ ag.name,
            if (ag.desc.len > 0) try std.fmt.allocPrint(a, " — {s}", .{ag.desc}) else "" });
        const scripts = h.listScripts() catch &[_][]const u8{};
        for (scripts) |s| try out.writer().print("  script {s}\n", .{s});
        const problems = h.validate(a) catch &[_][]const u8{};
        for (problems) |p| try out.writer().print("  ! {s}\n", .{p});
        return out.items;
    }

    if (std.mem.eql(u8, op, "read")) {
        const f = argStr(args, "file") orelse return "error: read needs file";
        const rel = h.safeRelPath(f) catch return "error: bad path";
        const body = h.read(rel);
        if (body.len == 0) return "error: file empty or missing";
        return if (body.len > 60_000) body[0..60_000] else body;
    }

    if (std.mem.eql(u8, op, "history")) {
        const p = try std.fs.path.join(a, &.{ h.dir, "refinement_history.jsonl" });
        const body = std.fs.cwd().readFileAlloc(a, p, 4 << 20) catch
            return "no refinement history yet";
        // last ~10 lines
        var lines = std.ArrayList([]const u8).init(a);
        var it = std.mem.splitScalar(u8, body, '\n');
        while (it.next()) |ln| {
            if (ln.len > 2) try lines.append(ln);
        }
        var out = std.ArrayList(u8).init(a);
        const start = if (lines.items.len > 10) lines.items.len - 10 else 0;
        for (lines.items[start..]) |ln| try out.writer().print("{s}\n", .{ln});
        return if (out.items.len > 0) out.items else "no refinement history yet";
    }

    if (std.mem.eql(u8, op, "snapshots")) {
        const sdir = try std.fmt.allocPrint(a, "{s}/snapshots", .{h.dir});
        var d = std.fs.cwd().openDir(sdir, .{ .iterate = true }) catch
            return "no snapshots yet — one is taken automatically before every harness write";
        defer d.close();
        var names = std.ArrayList([]const u8).init(a);
        var it = d.iterate();
        while (try it.next()) |e| {
            if (e.kind == .directory) try names.append(try a.dupe(u8, e.name));
        }
        std.mem.sort([]const u8, names.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.order(u8, x, y) == .gt;
            }
        }.lt);
        var out = std.ArrayList(u8).init(a);
        for (names.items[0..@min(names.items.len, 12)]) |n|
            try out.writer().print("{s}\n", .{n});
        try out.appendSlice("restore with: harness{op:\"rollback\",ts:<name>}");
        return out.items;
    }

    if (std.mem.eql(u8, op, "rollback")) {
        const sdir = try std.fmt.allocPrint(a, "{s}/snapshots", .{h.dir});
        var ts_name: []const u8 = "";
        if (util.jget(args, "ts")) |t| {
            if (t == .integer) ts_name = try std.fmt.allocPrint(a, "{d}", .{t.integer});
        }
        if (ts_name.len == 0) {
            // newest snapshot wins
            var d = std.fs.cwd().openDir(sdir, .{ .iterate = true }) catch
                return "error: no snapshots to roll back to";
            defer d.close();
            var it = d.iterate();
            while (try it.next()) |e| {
                if (e.kind == .directory and
                    (ts_name.len == 0 or std.mem.order(u8, e.name, ts_name) == .gt))
                    ts_name = try a.dupe(u8, e.name);
            }
        }
        if (ts_name.len == 0) return "error: no snapshots to roll back to";
        const snap_dir = try std.fmt.allocPrint(a, "{s}/{s}", .{ sdir, ts_name });
        var d = std.fs.cwd().openDir(snap_dir, .{ .iterate = true }) catch
            return "error: snapshot not found";
        defer d.close();
        var walker = try d.walk(a);
        defer walker.deinit();
        var n: usize = 0;
        while (try walker.next()) |e| {
            if (e.kind != .file) continue;
            const body = d.readFileAlloc(a, e.path, 4 << 20) catch continue;
            const rel = try std.fmt.allocPrint(a, "{s}", .{util.normalizePath(a, e.path)});
            h.applyEdit(rel, "write", body) catch continue;
            n += 1;
        }
        return try std.fmt.allocPrint(a,
            "rolled back {d} file(s) from snapshot {s}", .{ n, ts_name });
    }

    // ---- mutating ops: validate first (a bad op mustn't litter snapshots),
    // then snapshot, apply, and record history ----
    const mutating = std.mem.eql(u8, op, "append") or std.mem.eql(u8, op, "write") or
        std.mem.eql(u8, op, "delete") or std.mem.eql(u8, op, "skill_add") or
        std.mem.eql(u8, op, "agent_add");
    if (!mutating)
        return "error: op must be list|read|append|write|delete|skill_add|agent_add|history|snapshots|rollback";
    const snap = try h.snapshot();
    const ts = std.time.milliTimestamp();
    try refine_mod.writeSnapshot(a, h.dir, ts, snap);
    var applied: []const u8 = "";

    if (std.mem.eql(u8, op, "append") or std.mem.eql(u8, op, "write") or
        std.mem.eql(u8, op, "delete"))
    {
        const f = argStr(args, "file") orelse return "error: needs file";
        const rel = h.safeRelPath(f) catch return "error: bad path";
        // keep the tool out of its own bookkeeping dirs
        if (std.mem.startsWith(u8, rel, "snapshots/") or
            std.mem.eql(u8, rel, "refinement_history.jsonl"))
            return "error: cannot modify snapshots/history";
        if (!std.mem.eql(u8, op, "delete")) {
            const content = argStr(args, "content") orelse return "error: needs content";
            if (content.len == 0 or content.len > 64_000) return "error: content empty or >64k";
            try h.applyEdit(rel, op, content);
        } else {
            try h.applyEdit(rel, "delete", "");
        }
        applied = try std.fmt.allocPrint(a, "{s}:{s}", .{ op, rel });
    } else if (std.mem.eql(u8, op, "skill_add")) {
        const name = argStr(args, "name") orelse return "error: needs name";
        const desc = argStr(args, "desc") orelse return "error: needs desc (frontmatter description)";
        const content = argStr(args, "content") orelse return "error: needs content";
        const rel = try std.fmt.allocPrint(a, "skills/{s}/SKILL.md", .{name});
        const body = try std.fmt.allocPrint(a,
            "---\nname: {s}\ndescription: {s}\n---\n\n{s}\n", .{ name, desc, content });
        try h.applyEdit(rel, "write", body);
        applied = try std.fmt.allocPrint(a, "skill_add:{s}", .{rel});
    } else if (std.mem.eql(u8, op, "agent_add")) {
        const name = argStr(args, "name") orelse return "error: needs name";
        const desc = argStr(args, "desc") orelse return "error: needs desc (frontmatter description)";
        const content = argStr(args, "content") orelse return "error: needs content";
        const rel = try std.fmt.allocPrint(a, "sub_agents/{s}/AGENT.md", .{name});
        const body = try std.fmt.allocPrint(a,
            "---\nname: {s}\ndescription: {s}\n---\n\n{s}\n", .{ name, desc, content });
        try h.applyEdit(rel, "write", body);
        applied = try std.fmt.allocPrint(a, "agent_add:{s}", .{rel});
    } else unreachable; // `mutating` guard above

    try refine_mod.appendHistory(a, h.dir, .{
        .ts = ts,
        .scope = if (h.global) "global" else "local",
        .instructions = "direct edit via harness tool",
        .summary = applied,
        .edits = applied,
    });
    return try std.fmt.allocPrint(a,
        "harness updated: {s} (snapshot {d} — harness{{op:\"rollback\"}} to undo)", .{ applied, ts });
}

// ---------- coordination ----------

/// Normalize a claim pattern to workspace-relative '/'-separated glob.
/// Trailing '/' expands to "/**"; the workspace root itself becomes "**".
fn coordNormPat(ctx: *ToolCtx, pat0: []const u8) ![]const u8 {
    const pat = std.mem.trim(u8, pat0, " \t\"'");
    if (pat.len == 0) return error.EmptyPattern;
    const dir_claim = pat[pat.len - 1] == '/' or pat[pat.len - 1] == '\\';
    const abs = try util.resolvePath(ctx.scratch, ctx.cwd, pat);
    var rel = relPath(ctx, abs) orelse return error.OutsideWorkspace;
    if (rel.len == 0) return "**";
    if (dir_claim) rel = try std.fmt.allocPrint(ctx.scratch, "{s}/**", .{rel});
    return rel;
}

fn runCoord(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const co = ctx.coord orelse return "error: coord inactive (no shared session)";
    const op = argStr(args, "op") orelse return "error: missing op";
    const aid = ctx.aid;

    if (std.mem.eql(u8, op, "claim")) {
        const pat0 = argStr(args, "pat") orelse return "error: claim needs pat (workspace glob)";
        var out = std.ArrayList(u8).init(ctx.scratch);
        var it = std.mem.tokenizeScalar(u8, pat0, ',');
        while (it.next()) |p0| {
            const norm = coordNormPat(ctx, p0) catch {
                try out.writer().print("{s}: bad pattern\n", .{std.mem.trim(u8, p0, " ")});
                continue;
            };
            if (co.claim(aid, norm)) |holder| {
                if (ctx.livelog) |lv| lv.coordEvent("claim_conflict", norm, ctx.depth);
                try out.writer().print("{s}: CONFLICT held by {s}\n", .{ norm, holder });
            } else {
                if (ctx.livelog) |lv| lv.coordEvent("claim", norm, ctx.depth);
                try out.writer().print("{s}: ok\n", .{norm});
            }
        }
        return try ctx.alloc.dupe(u8, std.mem.trimRight(u8, out.items, "\n"));
    }
    if (std.mem.eql(u8, op, "rel")) {
        const n = if (argStr(args, "pat")) |p0| blk: {
            const norm = coordNormPat(ctx, p0) catch return "error: bad pat";
            break :blk co.release(aid, norm);
        } else co.release(aid, null);
        return std.fmt.allocPrint(ctx.alloc, "released {d} claim(s)", .{n});
    }
    if (std.mem.eql(u8, op, "status")) {
        const s = try co.status(ctx.scratch, aid);
        return try ctx.alloc.dupe(u8, s);
    }
    if (std.mem.eql(u8, op, "send")) {
        const to = argStr(args, "to") orelse return "error: send needs to (aid or '*')";
        const m = argStr(args, "m") orelse return "error: send needs m";
        co.send(aid, to, m) catch |e| switch (e) {
            error.UnknownAgent => {
                const s = co.status(ctx.scratch, aid) catch "";
                return std.fmt.allocPrint(ctx.alloc, "error: unknown agent '{s}'. {s}", .{ to, s });
            },
            else => return e,
        };
        if (ctx.livelog) |lv| lv.coordEvent("send", to, ctx.depth);
        return std.fmt.allocPrint(ctx.alloc, "sent → {s}", .{to});
    }
    if (std.mem.eql(u8, op, "poll")) {
        const s = try co.drain(ctx.scratch, aid);
        if (s.len == 0) return "no new events";
        return try ctx.alloc.dupe(u8, s);
    }
    if (std.mem.eql(u8, op, "note")) {
        const m = argStr(args, "m") orelse return "error: note needs m";
        co.setNote(aid, m);
        return "note set";
    }
    return std.fmt.allocPrint(ctx.alloc, "error: unknown op '{s}' (claim|rel|status|send|poll|note)", .{op});
}

// ---------- memory (supermemory) ----------

fn runMemory(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const mem = ctx.memory orelse
        return "error: memory inactive (no SUPERMEMORY_API_KEY / memory_key configured)";
    const op = argStr(args, "op") orelse return "error: missing op";

    if (std.mem.eql(u8, op, "remember")) {
        const m = argStr(args, "m") orelse return "error: remember needs m";
        const meta = try std.fmt.allocPrint(ctx.scratch,
            "{{\"src\":\"agent\",\"aid\":\"{s}\"}}", .{ctx.aid});
        const doc = try std.fmt.allocPrint(ctx.scratch, "xera-notes-{s}", .{ctx.aid});
        const out = try mem.add(ctx.scratch, m, doc, meta, "instant");
        if (ctx.livelog) |lv| lv.memEvent("remember", util.firstLine(m), ctx.depth);
        if (ctx.coord) |co| co.emit("M", ctx.aid, util.firstLine(m));
        return try ctx.alloc.dupe(u8, out);
    }
    if (std.mem.eql(u8, op, "recall")) {
        const q = argStr(args, "q") orelse return "error: recall needs q";
        var limit: usize = 6;
        if (util.jget(args, "n")) |n| {
            if (n == .integer and n.integer > 0 and n.integer <= 100) limit = @intCast(n.integer);
        }
        const out = try mem.search(ctx.scratch, q, limit);
        if (ctx.livelog) |lv| lv.memEvent("recall", q, ctx.depth);
        return try ctx.alloc.dupe(u8, if (out.len == 0) "(no relevant memories)" else out);
    }
    if (std.mem.eql(u8, op, "profile")) {
        const out = try mem.profile(ctx.scratch);
        return try ctx.alloc.dupe(u8, if (out.len == 0) "(empty profile)" else out);
    }
    if (std.mem.eql(u8, op, "forget")) {
        const ids = argStr(args, "ids") orelse return "error: forget needs ids (comma-separated)";
        const out = try mem.forget(ctx.scratch, ids);
        if (ctx.livelog) |lv| lv.memEvent("forget", ids, ctx.depth);
        return try ctx.alloc.dupe(u8, out);
    }
    if (std.mem.eql(u8, op, "forget_q")) {
        const q = argStr(args, "q") orelse return "error: forget_q needs q";
        var apply = false;
        if (util.jget(args, "apply")) |a| {
            if (a == .bool) apply = a.bool;
        }
        const out = try mem.forgetQuery(ctx.scratch, q, apply);
        if (ctx.livelog) |lv| lv.memEvent(if (apply) "forget_q" else "forget_q_dry", q, ctx.depth);
        return try ctx.alloc.dupe(u8, out);
    }
    return std.fmt.allocPrint(ctx.alloc,
        "error: unknown op '{s}' (remember|recall|profile|forget|forget_q)", .{op});
}

// ---------- filesystem: delete / mkdir ----------

fn runDelete(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const abs = try reqPath(ctx, args);
    writeGuard(ctx, abs) catch {
        return std.fmt.allocPrint(ctx.alloc, "error: {s} is outside the workspace ({s}); denied", .{ abs, ctx.workspace });
    };
    if (claimConflict(ctx, abs)) |msg| return msg;
    if (std.mem.eql(u8, abs, ctx.workspace)) {
        return "error: refusing to delete the workspace root itself";
    }
    const recursive = util.jbool(args, "recursive") orelse false;
    const st = std.fs.cwd().statFile(abs) catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: cannot stat {s}: {s}", .{ abs, @errorName(e) });
    };
    if (st.kind == .directory) {
        if (recursive) {
            try std.fs.cwd().deleteTree(abs);
            return std.fmt.allocPrint(ctx.alloc, "deleted directory tree {s}", .{abs});
        }
        std.fs.cwd().deleteDir(abs) catch |e| {
            return std.fmt.allocPrint(ctx.alloc,
                "error: cannot delete dir {s}: {s} (set recursive=true for non-empty)", .{ abs, @errorName(e) });
        };
        return std.fmt.allocPrint(ctx.alloc, "deleted directory {s}", .{abs});
    }
    std.fs.cwd().deleteFile(abs) catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: cannot delete {s}: {s}", .{ abs, @errorName(e) });
    };
    return std.fmt.allocPrint(ctx.alloc, "deleted {s}", .{abs});
}

fn runMakeDir(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const abs = try reqPath(ctx, args);
    writeGuard(ctx, abs) catch {
        return std.fmt.allocPrint(ctx.alloc, "error: {s} is outside the workspace ({s}); denied", .{ abs, ctx.workspace });
    };
    if (claimConflict(ctx, abs)) |msg| return msg;
    try std.fs.cwd().makePath(abs);
    return std.fmt.allocPrint(ctx.alloc, "created {s}", .{abs});
}

// ---------- web ----------

fn webClient(scratch: std.mem.Allocator, url: []const u8, max_bytes: usize) !struct { body: []u8, status: std.http.Status } {
    var client = std.http.Client{ .allocator = scratch };
    defer client.deinit();
    var body = std.ArrayList(u8).init(scratch);
    errdefer body.deinit();
    const res = try client.fetch(.{
        .location = .{ .url = url },
        .method = .GET,
        .extra_headers = &.{
            .{ .name = "user-agent", .value = "xera-rsi/0.1 (+https://github.com) terminal agent" },
            .{ .name = "accept", .value = "text/html,application/xhtml+xml,text/plain,application/json,*/*" },
        },
        .response_storage = .{ .dynamic = &body },
        .max_append_size = max_bytes,
    });
    return .{ .body = try body.toOwnedSlice(), .status = res.status };
}

/// RFC-3986 percent-encode (unreserved chars pass through).
fn urlEncode(scratch: std.mem.Allocator, s: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(scratch);
    for (s) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '~') {
            try out.append(c);
        } else {
            try out.writer().print("%{X:0>2}", .{c});
        }
    }
    return out.toOwnedSlice();
}

/// Crude HTML → text: drop scripts/styles/tags, decode common entities, collapse whitespace.
fn htmlToText(scratch: std.mem.Allocator, html: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(scratch);
    var i: usize = 0;
    var skip_depth: u32 = 0;
    while (i < html.len) {
        // skip <script>/<style> blocks
        if (skip_depth == 0 and i + 6 < html.len and html[i] == '<') {
            if (std.ascii.startsWithIgnoreCase(html[i + 1 ..], "script") or
                std.ascii.startsWithIgnoreCase(html[i + 1 ..], "style"))
            {
                skip_depth = 1;
            }
        }
        if (html[i] == '<') {
            const end = std.mem.indexOfScalarPos(u8, html, i, '>') orelse html.len - 1;
            const tag = html[i .. @min(end + 1, html.len)];
            if (skip_depth > 0) {
                if (std.ascii.indexOfIgnoreCase(tag, "/script") != null or
                    std.ascii.indexOfIgnoreCase(tag, "/style") != null) skip_depth = 0;
            } else {
                // block-level tags → newline
                for ([_][]const u8{ "p>", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "ul", "ol", "table", "section", "article", "header", "footer", "blockquote" }) |t| {
                    if (std.ascii.indexOfIgnoreCase(tag, t) != null) {
                        try out.append('\n');
                        break;
                    }
                }
            }
            i = end + 1;
            continue;
        }
        if (skip_depth > 0) {
            i += 1;
            continue;
        }
        if (html[i] == '&') {
            const ent = [_]struct { e: []const u8, r: []const u8 }{
                .{ .e = "&amp;", .r = "&" },  .{ .e = "&lt;", .r = "<" },
                .{ .e = "&gt;", .r = ">" },   .{ .e = "&quot;", .r = "\"" },
                .{ .e = "&#39;", .r = "'" },  .{ .e = "&apos;", .r = "'" },
                .{ .e = "&nbsp;", .r = " " },
            };
            var hit = false;
            for (ent) |e| {
                if (std.mem.startsWith(u8, html[i..], e.e)) {
                    try out.appendSlice(e.r);
                    i += e.e.len;
                    hit = true;
                    break;
                }
            }
            if (hit) continue;
        }
        if (html[i] == ' ' or html[i] == '\t') {
            if (out.items.len > 0 and out.items[out.items.len - 1] != ' ' and
                out.items[out.items.len - 1] != '\n') try out.append(' ');
            i += 1;
            continue;
        }
        try out.append(html[i]);
        i += 1;
    }
    return out.toOwnedSlice();
}

fn runWebFetch(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const url = argStr(args, "url") orelse return "error: missing url";
    const max_chars: usize = @intCast(@max(1000, util.jint(args, "max_chars") orelse 30_000));
    if (!std.mem.startsWith(u8, url, "http://") and !std.mem.startsWith(u8, url, "https://")) {
        return "error: url must start with http:// or https://";
    }
    const res = webClient(ctx.scratch, url, 4 << 20) catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: fetch failed: {s}", .{@errorName(e)});
    };
    if (res.status != .ok) {
        return std.fmt.allocPrint(ctx.alloc, "error: HTTP {d} for {s}", .{ @intFromEnum(res.status), url });
    }
    const body = res.body;
    const looks_html = std.ascii.indexOfIgnoreCase(body[0..@min(body.len, 4096)], "<html") != null or
        std.ascii.indexOfIgnoreCase(body[0..@min(body.len, 4096)], "<!doctype") != null;
    const text = if (looks_html) try htmlToText(ctx.scratch, body) else body;
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const capped = try util.truncateMiddle(ctx.scratch, trimmed, max_chars);
    return std.fmt.allocPrint(ctx.alloc, "[{s} · {d} bytes → {d} chars]\n{s}", .{
        url, body.len, capped.len, capped,
    });
}

fn runWebSearch(ctx: *ToolCtx, args: std.json.Value) ![]const u8 {
    const query = argStr(args, "query") orelse return "error: missing query";
    const max_results: usize = @intCast(@max(1, @min(20, util.jint(args, "max_results") orelse 8)));
    const url = try std.fmt.allocPrint(ctx.scratch,
        "https://lite.duckduckgo.com/lite/?q={s}", .{try urlEncode(ctx.scratch, query)});
    const res = webClient(ctx.scratch, url, 2 << 20) catch |e| {
        return std.fmt.allocPrint(ctx.alloc, "error: search failed: {s}", .{@errorName(e)});
    };
    if (res.status != .ok) {
        return std.fmt.allocPrint(ctx.alloc, "error: search HTTP {d}", .{@intFromEnum(res.status)});
    }
    // duckduckgo lite: <a ... class='result-link' href='URL'>TITLE</a>
    //                  <td class='result-snippet'>SNIP</td>
    var out = std.ArrayList(u8).init(ctx.scratch);
    var pos: usize = 0;
    var n: usize = 0;
    while (n < max_results) {
        const link_idx = std.ascii.indexOfIgnoreCasePos(res.body, pos, "result-link") orelse break;
        // backtrack to the href= of the enclosing <a>
        const a_start = std.mem.lastIndexOfScalar(u8, res.body[0..link_idx], '<') orelse break;
        const tag = res.body[a_start..];
        const href_pos = std.ascii.indexOfIgnoreCasePos(tag, 0, "href") orelse break;
        const q1 = std.mem.indexOfAnyPos(u8, tag, href_pos, "\"'") orelse break;
        const q2 = std.mem.indexOfAnyPos(u8, tag, q1 + 1, "\"'") orelse break;
        const href = tag[q1 + 1 .. q2];
        const a_end = std.mem.indexOfPos(u8, res.body, link_idx, "</a>") orelse break;
        const title_start = std.mem.indexOfScalarPos(u8, res.body, link_idx, '>') orelse break;
        const title = try htmlToText(ctx.scratch, res.body[title_start + 1 .. a_end]);
        // snippet follows the link row
        var snippet: []const u8 = "";
        if (std.ascii.indexOfIgnoreCasePos(res.body, a_end, "result-snippet")) |si| {
            if (si < a_end + 4096) {
                if (std.mem.indexOfScalarPos(u8, res.body, si, '>')) |s1| {
                    if (std.mem.indexOfPos(u8, res.body, s1, "</td>")) |s2| {
                        snippet = std.mem.trim(u8, try htmlToText(ctx.scratch, res.body[s1 + 1 .. s2]), " \t\r\n");
                    }
                }
            }
        }
        try out.writer().print("{d}. {s}\n   {s}\n   {s}\n\n", .{
            n + 1, std.mem.trim(u8, title, " \t\r\n"), href, snippet,
        });
        n += 1;
        pos = a_end;
    }
    if (n == 0) {
        return std.fmt.allocPrint(ctx.alloc, "no results for '{s}' (search backend may be unreachable)", .{query});
    }
    return std.fmt.allocPrint(ctx.alloc, "{d} result(s) for '{s}':\n\n{s}", .{ n, query, out.items });
}
