const std = @import("std");
const util = @import("util.zig");
const config = @import("config.zig");
const http = @import("http.zig");
const llm = @import("llm.zig");
const tools = @import("tools.zig");
const harness = @import("harness.zig");
const session = @import("session.zig");
const hooks_mod = @import("hooks.zig");
const live = @import("live.zig");
const coord_mod = @import("coord.zig");
const memory_mod = @import("memory.zig");
const refine_mod = @import("refine.zig");
const traces_mod = @import("traces.zig");

pub const Ui = tools.Ui;

pub const RunOutcome = struct {
    text: []const u8,
    iterations: usize,
    stop_reason: []const u8,
    usage: llm.Usage,
};

pub const Agent = struct {
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    h: *harness.Harness,
    cwd: []const u8,
    workspace: []const u8,
    depth: u8 = 0,
    messages: std.ArrayList(std.json.Value),
    tool_schemas: []llm.ToolSchema,
    ui: ?*const Ui = null,
    approver: ?*const tools.Approver = null,
    interactive: bool = false,
    slog: ?*session.Log = null,
    /// live observability stream (~/.xera-rsi/live/<id>.jsonl)
    livelog: ?*live.Live = null,
    usage: llm.Usage = .{},
    trajectory: std.ArrayList(u8),
    always_allow: *std.StringHashMap(void),
    /// lifecycle hooks (harness c8), loaded lazily on first use
    hooks: ?hooks_mod.Hooks = null,
    hooks_loaded: bool = false,
    /// set externally to request a graceful stop at the next loop boundary
    interrupted: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    /// shared-session coordination fabric (claims, feed, inbox); null when
    /// the agent isn't attached to a session (plain exec/improve runs)
    coord: ?*coord_mod.Coordinator = null,
    /// this agent's id inside the coordinator ("main", "t-1", "main/s")
    aid: []const u8 = "",
    /// whether the coordinator join + protocol intro already happened
    coord_joined: bool = false,
    /// supermemory client; null when unconfigured (memory tool inert,
    /// no auto-recall/capture)
    memory: ?*memory_mod.Memory = null,
    /// stable document id for turn capture ("xera-<session>-<aid>");
    /// empty = this agent doesn't capture turns (subagents don't)
    mem_doc_id: []const u8 = "",
    /// auto-recall + auto-capture toggle (config memory_auto)
    mem_auto: bool = true,
    /// background shells (run_shell background:true) owned by this agent.
    /// Heap-owned: detached workers may write here after the agent's
    /// stack frame is gone (spawned subs return while shells run on).
    shells: *tools.Shells,
    /// async subagents spawned via task op=spawn (same lifetime rule)
    subs: *tools.SubRegistry,
    /// pushed notices ([shell-done], [sub-done]) merged into <sync>
    notices: *tools.NoticeQueue,
    /// persistent objective re-surfaced each turn (goal tool)
    goal_state: tools.GoalState = .{},
    /// scheduled continual-harness refinement (refine tool), runs at turn end
    refine_req: tools.RefineReq = .{},
    /// telemetry store (~/.xera-rsi/traces) — records every turn/tool/skill
    traces: ?*traces_mod.Store = null,
    /// session id used in trace records
    sid: []const u8 = "",
    /// 1-based turn counter for trace records
    turn_no: usize = 0,
    /// tool names + skills used during the in-flight turn (flushed at turn end)
    turn_tools: std.ArrayList([]const u8),
    turn_skills: std.ArrayList([]const u8),

    pub fn init(
        alloc: std.mem.Allocator,
        cfg: *const config.Config,
        hclient: *http.HttpClient,
        h: *harness.Harness,
        cwd: []const u8,
        workspace: []const u8,
        depth: u8,
        extra_system: []const u8,
    ) !Agent {
        const sys = try h.systemPrompt(cwd, extra_system);
        var msgs = std.ArrayList(std.json.Value).init(alloc);
        try msgs.append(try llm.systemMessage(alloc, sys));
        var self = Agent{
            .alloc = alloc,
            .cfg = cfg,
            .hclient = hclient,
            .h = h,
            .cwd = cwd,
            .workspace = workspace,
            .depth = depth,
            .messages = msgs,
            .tool_schemas = try tools.schemas(alloc),
            .trajectory = std.ArrayList(u8).init(alloc),
            .turn_tools = std.ArrayList([]const u8).init(alloc),
            .turn_skills = std.ArrayList([]const u8).init(alloc),
            .always_allow = blk: {
                const m = try alloc.create(std.StringHashMap(void));
                m.* = std.StringHashMap(void).init(alloc);
                break :blk m;
            },
            .shells = undefined,
            .subs = undefined,
            .notices = undefined,
        };
        // cross-thread plumbing: heap-owned + page_allocator — detached
        // shell/sub workers may write to these after this agent's frame
        // (and its arena) is gone.
        const pa = std.heap.page_allocator;
        self.notices = try pa.create(tools.NoticeQueue);
        self.notices.* = tools.NoticeQueue.init(pa);
        self.shells = try pa.create(tools.Shells);
        self.shells.* = tools.Shells.init(pa, self.notices);
        self.subs = try pa.create(tools.SubRegistry);
        self.subs.* = tools.SubRegistry.init(pa);
        // c8: SessionStart hook fires once at agent creation
        const pl = try std.fmt.allocPrint(alloc,
            "{{\"event\":\"SessionStart\",\"cwd\":\"{s}\"}}", .{try util.jescape(alloc, cwd)});
        const note = self.fireHook("SessionStart", pl, "");
        if (note.len > 0) {
            try self.messages.append(try llm.systemMessage(alloc,
                try std.fmt.allocPrint(alloc, "[session hook context]\n{s}", .{note})));
        }
        return self;
    }

    fn status(self: *Agent, msg: []const u8) void {
        if (self.ui) |u| {
            if (u.on_status) |f| f(u.ctx, msg);
        }
    }

    fn fireHook(self: *Agent, event: []const u8, payload: []const u8, tool_name: []const u8) []const u8 {
        if (!self.hooks_loaded) {
            self.hooks_loaded = true;
            self.hooks = hooks_mod.Hooks.load(self.alloc, self.h.hooksPath());
        }
        if (self.hooks) |*hk| {
            var arena = std.heap.ArenaAllocator.init(self.alloc);
            defer arena.deinit();
            return self.alloc.dupe(u8, hk.fire(arena.allocator(), event, payload, tool_name)) catch "";
        }
        return "";
    }

    /// Push a user message and run the tool loop to completion.
    /// Returns the final assistant text.
    pub fn runTask(self: *Agent, prompt: []const u8) !RunOutcome {
        return self.runTaskAuthored(prompt, "");
    }

    /// runTask with an author tag for observability.
    pub fn runTaskAuthored(self: *Agent, prompt: []const u8, author: []const u8) !RunOutcome {
        self.turn_no += 1;
        self.turn_tools.clearRetainingCapacity();
        self.turn_skills.clearRetainingCapacity();
        const t0 = std.time.milliTimestamp();
        // UserPromptSubmit hook (c8): may attach context or block
        const pl = try std.fmt.allocPrint(self.alloc,
            "{{\"event\":\"UserPromptSubmit\",\"cwd\":\"{s}\"}}", .{try util.jescape(self.alloc, self.cwd)});
        const extra = self.fireHook("UserPromptSubmit", pl, "");
        const final_prompt = if (extra.len > 0)
            try std.fmt.allocPrint(self.alloc, "{s}\n\n<hook_context>\n{s}\n</hook_context>", .{ prompt, extra })
        else
            prompt;
        const um = try llm.userMessage(self.alloc, final_prompt);
        try self.messages.append(um);
        if (self.slog) |s| try s.append(um);
        if (self.livelog) |lv| lv.user(prompt, final_prompt, author, self.depth);
        try self.recallMemory(prompt);
        const outcome = try self.loop();
        if (self.livelog) |lv| lv.turnEnd(outcome.stop_reason, outcome.iterations,
            self.usage.prompt, self.usage.completion, self.depth);
        self.captureMemory(prompt, author, outcome.text);
        if (self.traces) |ts| ts.turn(self.alloc, self.sid, self.turn_no,
            prompt, outcome.text, self.turn_tools.items, self.turn_skills.items,
            self.usage.prompt, self.usage.completion, outcome.iterations,
            outcome.stop_reason, std.time.milliTimestamp() - t0);
        self.runPendingRefine();
        return outcome;
    }

    /// Auto-recall: search long-term memory for this prompt and inject
    /// hits as a <memory> system message. Best-effort — the turn must
    /// never fail because the memory layer hiccuped.
    fn recallMemory(self: *Agent, prompt: []const u8) !void {
        const mem = self.memory orelse return;
        if (!self.mem_auto) return;
        const hits = mem.search(self.alloc, prompt, 6) catch return;
        if (hits.len == 0 or std.mem.startsWith(u8, hits, "error:")) return;
        if (self.livelog) |lv| lv.memEvent("recall_auto", prompt, self.depth);
        try self.messages.append(try llm.systemMessage(self.alloc,
            try std.fmt.allocPrint(self.alloc,
                "<memory> durable context from prior sessions (id [sim] text):\n{s}\n</memory>",
                .{hits})));
    }

    /// Auto-capture: hand the turn to supermemory as an append to this
    /// agent's session document — it extracts/links the memories itself.
    /// Detached: nothing user-visible awaits the network.
    fn captureMemory(self: *Agent, prompt: []const u8, author: []const u8, answer: []const u8) void {
        const mem = self.memory orelse return;
        if (!self.mem_auto or self.mem_doc_id.len == 0) return;
        const pa = std.heap.page_allocator;
        const who = if (author.len > 0) author else "user";
        const tail = if (answer.len > 8_000) answer[answer.len - 8_000 ..] else answer;
        const turn = std.fmt.allocPrint(pa,
            "{s}: {s}\nassistant[{s}]: {s}", .{ who, prompt, self.aid, tail }) catch return;
        const meta = std.fmt.allocPrint(pa,
            "{{\"src\":\"xera-rsi\",\"agent\":\"{s}\"}}", .{self.aid}) catch return;
        const job = pa.create(CaptureJob) catch return;
        job.* = .{ .mem = mem, .doc = self.mem_doc_id, .turn = turn, .meta = meta };
        const th = std.Thread.spawn(.{}, captureThread, .{job}) catch return;
        th.detach();
        if (self.livelog) |lv| lv.memEvent("capture", self.mem_doc_id, self.depth);
    }

    const CaptureJob = struct {
        mem: *memory_mod.Memory,
        doc: []const u8,
        turn: []const u8,
        meta: []const u8,
    };

    fn captureThread(job: *CaptureJob) void {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        _ = job.mem.add(arena.allocator(), job.turn, job.doc, job.meta, null) catch {};
        const p = std.heap.page_allocator;
        p.free(job.turn);
        p.free(job.meta);
        p.destroy(job);
    }

    /// A `refine` tool call schedules refinement; it runs here, at turn
    /// end — a separate model pass applies evidence-backed harness edits,
    /// then the result is injected so the next turn sees it.
    fn runPendingRefine(self: *Agent) void {
        self.refine_req.mu.lock();
        const instr = self.refine_req.pending orelse {
            self.refine_req.mu.unlock();
            return;
        };
        const glob = self.refine_req.scope_global;
        self.refine_req.pending = null;
        self.refine_req.mu.unlock();

        const summary = refine_mod.run(self.alloc, self.cfg, self.hclient, self.h,
            self.messages.items, self.trajectory.items, .{
                .instructions = instr,
                .scope_global = glob,
            }) catch |e|
            std.fmt.allocPrint(self.alloc, "refine failed: {s}", .{@errorName(e)}) catch "refine failed";

        const note = std.fmt.allocPrint(self.alloc,
            "[refine{s}] {s}", .{ if (glob) " global" else "", summary }) catch return;
        self.messages.append(llm.systemMessage(self.alloc, note) catch return) catch return;
        if (self.coord) |co| co.emit("R", self.aid, summary);
        if (self.livelog) |lv| lv.memEvent("refine", summary, self.depth);
    }

    /// One-time coordinator join: pick an aid, register, inject the
    /// compact protocol briefing so the model knows the rules.
    fn coordSetup(self: *Agent) !void {
        const co = self.coord orelse return;
        if (self.coord_joined) return;
        self.coord_joined = true;
        if (self.aid.len == 0) self.aid = "main";
        co.register(self.aid);
        try self.messages.append(try llm.systemMessage(self.alloc,
            try std.fmt.allocPrint(self.alloc,
                "<coord> shared session — you are agent '{s}'. Sibling agents (subagents) work this same workspace.\n" ++
                "1) claim before you write: coord{{op:\"claim\",pat:\"src/*.zig\"}} — writes into paths claimed by others are refused.\n" ++
                "2) claims persist until coord{{op:\"rel\"}} or ~10min expiry — release when done.\n" ++
                "3) coord{{op:\"status\"}} = roster · coord{{op:\"send\",to:\"aid\"|\"*\",m:\"...\"}} = message (keep m to facts/paths, no prose).\n" ++
                "4) <sync> messages push the event feed between your turns: seq|kind|from|data — " ++
                "k: b=turn-start e=turn-end w=write $=shell c=claim r=release !=contention m=msg n=note +=join.</coord>",
                .{self.aid})));
    }

    /// Push-delivery: drain feed + inbox + pushed notices (shell/sub
    /// completions) and inject as a <sync> system message. A persistent
    /// goal is re-surfaced here too. Runs before every model call so
    /// nothing depends on the model remembering to poll.
    fn syncCoord(self: *Agent) !void {
        var block = std.ArrayList(u8).init(self.alloc);
        {
            self.goal_state.mu.lock();
            defer self.goal_state.mu.unlock();
            if (self.goal_state.objective.len > 0) {
                self.goal_state.turns += 1;
                try block.writer().print("goal|{s}|turn {d}\n", .{
                    self.goal_state.objective, self.goal_state.turns });
            }
        }
        for (self.notices.drain()) |n| {
            try block.appendSlice(n);
            try block.append('\n');
        }
        if (self.coord) |co| {
            const lines = try co.drain(self.alloc, self.aid);
            if (lines.len > 0) {
                try block.appendSlice(std.mem.trimRight(u8, lines, "\n"));
                try block.append('\n');
            }
        }
        if (block.items.len == 0) return;
        const m = try std.fmt.allocPrint(self.alloc, "<sync>\n{s}</sync>",
            .{std.mem.trimRight(u8, block.items, "\n")});
        try self.messages.append(try llm.systemMessage(self.alloc, m));
    }

    /// Continue the loop with whatever is already in history.
    pub fn loop(self: *Agent) !RunOutcome {
        // c8: Stop fires when the tool loop exits for any reason
        defer _ = self.fireHook("Stop", "{\"event\":\"Stop\"}", "");
        try self.coordSetup();
        if (self.coord) |co| {
            co.emit("b", self.aid, "");
            defer co.emit("e", self.aid, "");
        }
        var iters: usize = 0;
        var last_text: []const u8 = "";
        var stop_reason: []const u8 = "stop";
        while (iters < self.cfg.max_iters) {
            if (self.interrupted.load(.acquire)) {
                stop_reason = "interrupted";
                break;
            }
            iters += 1;
            try self.compactIfNeeded();
            try self.syncCoord();

            const on_delta = self.makeDeltaCb();
            const on_reason = self.makeReasoningCb();
            const res = llm.chat(self.hclient, self.alloc, self.cfg, self.messages.items, self.tool_schemas, .{
                .max_tokens = self.cfg.max_tokens,
                .temperature = self.cfg.temperature,
                .stream = self.cfg.stream,
                .on_delta = on_delta,
                .on_reasoning = on_reason,
            }) catch |e| return e;
            self.usage.prompt = res.usage.prompt;
            self.usage.completion += res.usage.completion;

            try self.messages.append(res.assistant_message);
            if (self.slog) |s| try s.append(res.assistant_message);
            if (self.livelog) |lv| {
                if (res.reasoning.len > 0) lv.reasoning(res.reasoning, self.depth);
                var names = std.ArrayList(u8).init(self.alloc);
                for (res.tool_calls, 0..) |call, ci| {
                    if (ci > 0) try names.append(',');
                    try names.appendSlice(call.name);
                }
                lv.assistant(res.content, names.items, self.depth);
                lv.usage(res.usage.prompt, res.usage.completion, self.depth);
            }
            if (res.content.len > 0) {
                last_text = res.content;
                try self.trajectory.appendSlice("\n## assistant\n");
                try self.trajectory.appendSlice(res.content);
                try self.trajectory.append('\n');
            }

            if (res.tool_calls.len == 0) {
                return .{
                    .text = last_text,
                    .iterations = iters,
                    .stop_reason = res.finish_reason,
                    .usage = self.usage,
                };
            }

            // execute each tool call and append results
            for (res.tool_calls) |call| {
                const result = try self.dispatch(call);
                const tm = try llm.toolResultMessage(self.alloc, call.id, result);
                try self.messages.append(tm);
                if (self.slog) |s| try s.append(tm);
            }
        }
        stop_reason = "max_iters";
        self.status("reached iteration limit; stopping");
        return .{
            .text = last_text,
            .iterations = iters,
            .stop_reason = stop_reason,
            .usage = self.usage,
        };
    }

    fn makeDeltaCb(self: *Agent) ?llm.DeltaCb {
        const u = self.ui orelse return null;
        const f = u.on_text orelse return null;
        return .{ .ctx = u.ctx, .f = f };
    }

    fn makeReasoningCb(self: *Agent) ?llm.DeltaCb {
        const u = self.ui orelse return null;
        const f = u.on_reasoning orelse return null;
        return .{ .ctx = u.ctx, .f = f };
    }

    fn dispatch(self: *Agent, call: llm.ToolCall) ![]const u8 {
        const t = tools.find(call.name) orelse {
            return std.fmt.allocPrint(self.alloc, "error: unknown tool '{s}'", .{call.name});
        };

        var scratch_arena = std.heap.ArenaAllocator.init(self.alloc);
        defer scratch_arena.deinit();
        const scratch = scratch_arena.allocator();

        const args = util.jparse(scratch, call.arguments) catch {
            return std.fmt.allocPrint(self.alloc, "error: could not parse arguments for {s}", .{call.name});
        };
        var tctx = self.toolCtxFor(scratch);
        const summary = t.summary(&tctx, args);

        // c8 PreToolUse hook — a block skips the call and feeds the reason back
        const pre_payload = try std.fmt.allocPrint(scratch,
            "{{\"event\":\"PreToolUse\",\"tool_name\":\"{s}\",\"cwd\":\"{s}\"}}",
            .{ t.name, try util.jescape(scratch, self.cwd) });
        const pre = self.fireHook("PreToolUse", pre_payload, t.name);
        if (pre.len > 0) {
            const msg = try std.fmt.allocPrint(self.alloc, "error: hook blocked {s}: {s}", .{ t.name, pre });
            self.reportResult(t.name, msg, false, call.id);
            try self.trajTool(t.name, summary, msg);
            return msg;
        }

        // approval gate
        if (try self.needsApproval(t.*)) {
            const approved = try self.askApproval(t.*, summary);
            if (!approved) {
                self.reportResult(t.name, "denied by user", false, call.id);
                try self.trajTool(t.name, summary, "denied by user");
                return std.fmt.allocPrint(self.alloc, "error: user denied {s} ({s})", .{ t.name, summary });
            }
        }

        if (self.ui) |u| {
            if (u.on_tool_call) |f| f(u.ctx, t.name, summary);
        }
        try self.trajCall(t.name, summary, call.arguments);
        if (self.livelog) |lv| {
            lv.toolCall(t.name, summary, call.arguments, call.id, self.depth);
            if (std.mem.eql(u8, t.name, "skill_load")) {
                if (util.jstr(args, "name")) |sk| lv.skillUsed(sk, self.depth);
            } else if (std.mem.eql(u8, t.name, "task")) {
                const persona = util.jstr(args, "agent") orelse "";
                const title = util.jstr(args, "title") orelse
                    (util.jstr(args, "prompt") orelse "");
                lv.subagent(persona, util.firstLine(title), self.depth);
            }
        }

        const out = t.run(&tctx, args) catch |e| {
            const msg = try std.fmt.allocPrint(self.alloc, "error: {s} failed: {s}", .{ t.name, @errorName(e) });
            self.reportResult(t.name, msg, false, call.id);
            try self.trajTool(t.name, summary, msg);
            if (self.traces) |ts| {
                ts.tool(self.alloc, self.sid, self.turn_no, t.name, false, msg);
                self.turn_tools.append(t.name) catch {};
            }
            return msg;
        };
        // free awareness for siblings: successful writes land on the feed
        tools.emitWriteEvent(&tctx, t.name, args, out);
        // c8 PostToolUse hook — feedback is appended as context the model sees
        const post_payload = try std.fmt.allocPrint(scratch,
            "{{\"event\":\"PostToolUse\",\"tool_name\":\"{s}\",\"cwd\":\"{s}\"}}",
            .{ t.name, try util.jescape(scratch, self.cwd) });
        const post = self.fireHook("PostToolUse", post_payload, t.name);
        const final_out = if (post.len > 0)
            try std.fmt.allocPrint(self.alloc, "{s}\n\n<hook_feedback>\n{s}\n</hook_feedback>", .{ out, post })
        else
            out;
        const preview = try util.truncateMiddle(scratch, final_out, 400);
        const ok = !std.mem.startsWith(u8, out, "error:");
        self.reportResult(t.name, preview, ok, call.id);
        try self.trajTool(t.name, summary, preview);
        if (self.traces) |ts| {
            ts.tool(self.alloc, self.sid, self.turn_no, t.name, ok, summary);
            self.turn_tools.append(t.name) catch {};
            if (std.mem.eql(u8, t.name, "skill_load")) {
                if (util.jstr(args, "name")) |sk| {
                    ts.skill(self.alloc, self.sid, self.turn_no, sk);
                    self.turn_skills.append(sk) catch {};
                }
            }
        }
        return final_out;
    }

    fn toolCtxFor(self: *Agent, scratch: std.mem.Allocator) tools.ToolCtx {
        return .{
            .alloc = self.alloc,
            .scratch = scratch,
            .cwd = self.cwd,
            .workspace = self.workspace,
            .h = self.h,
            .cfg = self.cfg,
            .approver = self.approver,
            .depth = self.depth,
            .ui = self.ui,
            .livelog = self.livelog,
            .always_allow = self.always_allow,
            .subagent_run = subagentRun,
            .coord = self.coord,
            .aid = self.aid,
            .memory = self.memory,
            .shells = self.shells,
            .subs = self.subs,
            .notices = self.notices,
            .goal_state = &self.goal_state,
            .refine_req = &self.refine_req,
            .traces = self.traces,
            .sid = self.sid,
            .hclient = self.hclient,
        };
    }

    fn needsApproval(self: *Agent, t: tools.Tool) !bool {
        if (t.danger == .safe) return false;
        if (self.always_allow.contains(t.name)) return false;
        return self.cfg.approval == .ask;
    }

    fn askApproval(self: *Agent, t: tools.Tool, summary: []const u8) !bool {
        const ap = self.approver orelse return true; // nobody to ask -> allow
        const decision = try ap.ask(ap.ctx, t.name, summary, t.danger);
        switch (decision) {
            .allow => return true,
            .always => {
                try self.always_allow.put(try self.alloc.dupe(u8, t.name), {});
                return true;
            },
            .deny => return false,
        }
    }

    /// Report a tool result to both the UI and the live event stream.
    fn reportResult(self: *Agent, name: []const u8, preview: []const u8, ok: bool, call_id: []const u8) void {
        if (self.ui) |u| {
            if (u.on_tool_result) |f| f(u.ctx, name, preview, ok);
        }
        if (self.livelog) |lv| lv.toolResult(name, ok, preview, call_id, self.depth);
    }

    fn trajCall(self: *Agent, name: []const u8, summary: []const u8, args: []const u8) !void {
        const w = self.trajectory.writer();
        try w.print("\n> {s}({s})\n", .{ name, summary });
        _ = args;
    }

    fn trajTool(self: *Agent, name: []const u8, summary: []const u8, result: []const u8) !void {
        const w = self.trajectory.writer();
        _ = name;
        _ = summary;
        try w.print("{s}\n", .{try util.truncateMiddle(self.alloc, result, 1200)});
    }

    fn subagentRun(tctx: *tools.ToolCtx, prompt: []const u8, agent_name: ?[]const u8) anyerror![]const u8 {
        var arena = std.heap.ArenaAllocator.init(tctx.alloc);
        defer arena.deinit();
        const aalloc = arena.allocator();

        var hc = http.HttpClient.init(aalloc);
        defer hc.deinit();

        // c5: optional named persona from sub_agents/<name>/AGENT.md
        var extra = std.ArrayList(u8).init(aalloc);
        try extra.appendSlice("You are a subagent of xera-rsi. Complete the delegated task and report back " ++
            "concisely with concrete results (paths, numbers, findings).");
        if (agent_name) |an| {
            const persona = tctx.h.agentBody(an) catch |e| {
                const avail = tctx.h.listAgents() catch &[_]harness.NamedItem{};
                var hint = std.ArrayList(u8).init(aalloc);
                for (avail) |ag| {
                    if (hint.items.len > 0) try hint.appendSlice(", ");
                    try hint.appendSlice(ag.name);
                }
                return try tctx.alloc.dupe(u8, try std.fmt.allocPrint(aalloc,
                    "error: subagent persona '{s}' not found ({s}). available: {s}",
                    .{ an, @errorName(e), hint.items }));
            };
            try extra.writer().print("\n\nYou are the '{s}' sub-agent:\n{s}", .{ an, persona });
        }

        var sub = try Agent.init(
            aalloc, tctx.cfg, &hc, tctx.h, tctx.cwd, tctx.workspace,
            tctx.depth + 1, extra.items,
        );
        sub.approver = tctx.approver;
        sub.ui = tctx.ui;
        sub.livelog = tctx.livelog;
        // join the shared fabric as a child of this agent ("main" -> "main/s",
        // or the explicit "main/sub-3" a spawn assigned)
        sub.coord = tctx.coord;
        sub.aid = if (tctx.sub_aid) |a|
            a
        else if (tctx.aid.len > 0)
            try std.fmt.allocPrint(aalloc, "{s}/s", .{tctx.aid})
        else
            "s";
        // memory tool works for subagents; turn capture stays on channel agents
        sub.memory = tctx.memory;
        sub.traces = tctx.traces;
        sub.sid = try std.fmt.allocPrint(aalloc, "{s}/{s}", .{ tctx.sid, sub.aid });
        if (tctx.always_allow) |aa| sub.always_allow = aa;
        _ = sub.fireHook("SubagentStart", "{\"event\":\"SubagentStart\"}", "");
        const outcome = try sub.runTask(prompt);
        _ = sub.fireHook("SubagentStop", "{\"event\":\"SubagentStop\"}", "");
        return try tctx.alloc.dupe(u8, outcome.text);
    }

    /// Force a history compaction now (the /summarize command).
    pub fn compactNow(self: *Agent) !void {
        try self.compactImpl(true);
    }

    /// Drop oldest middle messages when serialized history exceeds the budget.
    fn compactIfNeeded(self: *Agent) !void {
        try self.compactImpl(false);
    }

    fn compactImpl(self: *Agent, force: bool) !void {
        _ = self.fireHook("PreCompact", "{\"event\":\"PreCompact\"}", "");
        var size: usize = 0;
        var sizes = try self.alloc.alloc(usize, self.messages.items.len);
        for (self.messages.items, 0..) |m, i| {
            const s = try util.jstringify(self.alloc, m);
            sizes[i] = s.len;
            size += s.len;
        }
        if (!force and size <= self.cfg.history_budget) return;

        const keep_budget = self.cfg.history_budget * 3 / 5;
        var start: usize = self.messages.items.len;
        var acc: usize = 0;
        while (start > 1) {
            const s = sizes[start - 1];
            if (acc + s > keep_budget) break;
            acc += s;
            start -= 1;
        }
        // never keep a leading "tool" result (its call would be missing)
        while (start < self.messages.items.len) {
            const role = util.jstr(self.messages.items[start], "role") orelse break;
            if (!std.mem.eql(u8, role, "tool")) break;
            start += 1;
        }
        if (start <= 1) return; // nothing to drop
        var kept = std.ArrayList(std.json.Value).init(self.alloc);
        try kept.append(self.messages.items[0]);
        const note = try std.fmt.allocPrint(self.alloc, "[xera-rsi: {d} earlier messages ({d} bytes) were dropped to fit the context window]", .{ start - 1, size });
        try kept.append(try llm.systemMessage(self.alloc, note));
        for (self.messages.items[start..]) |m| try kept.append(m);
        self.messages.deinit();
        self.messages = kept;
        self.status("compacted conversation history");
        _ = self.fireHook("PostCompact", "{\"event\":\"PostCompact\"}", "");
    }
};
