//! Codex-style TUI: scrollback history, bordered composer, status bar,
//! spinner, streaming, tool cells, key-driven approvals.
//! The agent runs on a worker thread; this thread renders + handles input.

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
const tui = @import("tui.zig");
const coord_mod = @import("coord.zig");
const memory_mod = @import("memory.zig");
const traces_mod = @import("traces.zig");
const rsi = @import("rsi.zig");

const sty = tui.Style;
const Key = tui.Key;

const spinner_frames = "⠋⠙⠚⠞⠖⠦⠴⠲⠳⠓";

// ---------- banner ----------

const glyph5 = std.StaticStringMap([5][]const u8).initComptime(.{
    .{ "x", .{ "█   █", " █ █ ", "  █  ", " █ █ ", "█   █" } },
    .{ "e", .{ "█████", "█    ", "████ ", "█    ", "█████" } },
    .{ "r", .{ "████ ", "█   █", "████ ", "█ █  ", "█  █ " } },
    .{ "a", .{ " ███ ", "█   █", "█████", "█   █", "█   █" } },
    .{ "-", .{ "     ", "     ", "█████", "     ", "     " } },
    .{ "o", .{ " ███ ", "█   █", "█   █", "█   █", " ███ " } },
    .{ "p", .{ "████ ", "█   █", "████ ", "█    ", "█    " } },
    .{ "t", .{ "█████", "  █  ", "  █  ", "  █  ", "  █  " } },
    .{ "i", .{ "█████", "  █  ", "  █  ", "  █  ", "█████" } },
    .{ "m", .{ "█   █", "██ ██", "█ █ █", "█   █", "█   █" } },
    .{ "u", .{ "█   █", "█   █", "█   █", "█   █", " ███ " } },
    .{ "s", .{ " ████", "█    ", " ███ ", "    █", "████ " } },
});

const banner_text = "xera-rsi";

const banner_rows: [5][]const u8 = blk: {
    var rows: [5][]const u8 = .{ "", "", "", "", "" };
    for (0..5) |r| {
        for (0..banner_text.len) |i| {
            rows[r] = rows[r] ++ glyph5.get(banner_text[i .. i + 1]).?[r] ++ " ";
        }
    }
    break :blk rows;
};

const banner_cols = banner_text.len * 6;
const banner_anim_frames: usize = 30;

/// comptime lit-pixel mask for the glyph grid (5 rows x banner_cols)
const banner_lit: [5][banner_cols]bool = blk: {
    var m: [5][banner_cols]bool = .{.{false} ** banner_cols} ** 5;
    for (0..5) |r| {
        var col: usize = 0;
        var i: usize = 0;
        while (i < banner_rows[r].len) {
            const c = banner_rows[r][i];
            const len = if (c < 0x80) @as(usize, 1) else (std.unicode.utf8ByteSequenceLength(c) catch 1);
            if (c != ' ' and col < banner_cols) m[r][col] = true;
            col += 1;
            i += len;
        }
    }
    break :blk m;
};

/// gradient stops for the banner: cyan -> blue -> white
fn gradRGB(t: f32) [3]u8 {
    const c0 = [3]f32{ 0, 229, 255 };
    const c1 = [3]f32{ 56, 140, 255 };
    const c2 = [3]f32{ 250, 250, 255 };
    var a = c0;
    var b = c1;
    var u = t * 2;
    if (t >= 0.5) {
        a = c1;
        b = c2;
        u = (t - 0.5) * 2;
    }
    var out: [3]u8 = undefined;
    for (0..3) |i| out[i] = @intFromFloat(@min(255, @max(0, a[i] + (b[i] - a[i]) * u)));
    return out;
}

const scramble_chars = [_][]const u8{
    "░", "▒", "▓", "<", ">", "/", "\\", "|", "=", "+", "*", "^", "?", "#", "%", "&", "x", "X", "0", "1",
};

/// frame at which the bright cell (r,c) locks in — sweeps left→right with jitter
fn lockFrame(r: usize, c: usize) usize {
    const base = c * (banner_anim_frames - 8) / @max(1, banner_cols - 1);
    const jitter = (r * 13 + c * 7) % 5;
    return base + jitter;
}

/// Render the 3D banner: gradient front face + extruded down-right `░` depth.
/// `frame` < banner_anim_frames → matrix-scramble reveal.
fn renderBanner(fa: std.mem.Allocator, w: usize, frame: usize, out: *std.ArrayList([]const u8)) !void {
    const lead = if (w > banner_cols + 1) (w - banner_cols - 1) / 2 else 0;
    const pad = try spaces(fa, lead);
    const animating = frame < banner_anim_frames;

    try out.append("");
    var r: usize = 0;
    while (r < 6) : (r += 1) {
        var line = std.ArrayList(u8).init(fa);
        try line.appendSlice(pad);
        var c: usize = 0;
        while (c < banner_cols + 1 and lead + c < w) : (c += 1) {
            const bright = r < 5 and c < banner_cols and banner_lit[r][c];
            // extruded depth: shadow of the cell up-left that isn't covered by a glyph
            const shade = r > 0 and c > 0 and r - 1 < 5 and c - 1 < banner_cols and
                banner_lit[r - 1][c - 1] and !bright;
            if (bright) {
                const lf = lockFrame(r, c);
                if (!animating or frame >= lf) {
                    const t = @as(f32, @floatFromInt(c)) / @as(f32, @floatFromInt(banner_cols - 1));
                    const rgb = gradRGB(t);
                    try line.writer().print("\x1b[38;2;{d};{d};{d}m█", .{ rgb[0], rgb[1], rgb[2] });
                } else if (frame + 10 >= lf) {
                    // shimmer zone just ahead of the lock front
                    const g = scramble_chars[(r * 31 + c * 17 + frame * 13) % scramble_chars.len];
                    const glow = 90 + ((r * 31 + c * 17 + frame * 13) % 120);
                    try line.writer().print("\x1b[38;2;40;{d};{d}m{s}", .{ glow, @min(255, glow + 60), g });
                } else {
                    try line.append(' ');
                }
            } else if (shade) {
                const lf = lockFrame(r - 1, c - 1);
                if (!animating or frame >= lf) {
                    try line.writer().print("{s}░", .{sty.shadow});
                } else {
                    try line.append(' ');
                }
            } else {
                try line.append(' ');
            }
        }
        try line.appendSlice(sty.reset);
        try out.append(line.items);
    }
}

// ---------- slash commands ----------

const SlashCmd = struct {
    name: []const u8,
    desc: []const u8,
    takes_arg: bool = false,
};

const slash_cmds = [_]SlashCmd{
    .{ .name = "/agents", .desc = "named sub-agent personas" },
    .{ .name = "/auto", .desc = "toggle auto-approve for writes/shell" },
    .{ .name = "/clear", .desc = "fresh context + clear screen" },
    .{ .name = "/exit", .desc = "quit" },
    .{ .name = "/harness", .desc = "harness components + validation" },
    .{ .name = "/help", .desc = "commands and keys" },
    .{ .name = "/improve", .desc = "harness² recursion (own command)" },
    .{ .name = "/library", .desc = "adopted harness snapshots" },
    .{ .name = "/model", .desc = "search and switch provider model" },
    .{ .name = "/resume", .desc = "load a session: /resume <id>", .takes_arg = true },
    .{ .name = "/rsi-loop", .desc = "run an rsi loop: /rsi-loop [focus]", .takes_arg = true },
    .{ .name = "/sessions", .desc = "list saved sessions" },
    .{ .name = "/skills", .desc = "list harness skills" },
    .{ .name = "/status", .desc = "session info" },
    .{ .name = "/summarize", .desc = "compact the conversation now" },
    .{ .name = "/usage", .desc = "token usage this session" },
};

// ---------- events (worker -> ui) ----------

const Approval = struct {
    mu: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    decision: ?tools.ApprovalDecision = null,
    label: []const u8,
    danger: tools.Danger,
};

const Event = union(enum) {
    text: []const u8,
    think: []const u8,
    tool_start: struct { name: []const u8, summary: []const u8 },
    tool_result: struct { preview: []const u8, ok: bool },
    status: []const u8,
    done: agent.RunOutcome,
    failed: []const u8,
    approval: *Approval,
    cleared,
    /// self-check failed → offer an rsi loop (accept/reject panel)
    rsi_proposal: *rsi.Proposal,
    /// rsi loop finished on the worker
    rsi_done,
    /// "model is improving itself" indicator
    training: struct { on: bool, label: []const u8 },
    models: llm.ModelCatalog,
    model_changed: []const u8,
};

const Job = union(enum) {
    prompt: []const u8,
    /// run the rsi loop on an accepted/manual proposal
    rsi_run: *rsi.Proposal,
    reset,
    usage,
    summarize,
    load_msgs: std.ArrayList(std.json.Value),
    list_models,
    set_model: []const u8,
    quit,
};

const Queue = struct {
    mu: std.Thread.Mutex = .{},
    items: std.ArrayList(Event),
    jobs: std.ArrayList(Job),
    cond: std.Thread.Condition = .{},

    fn init(alloc: std.mem.Allocator) Queue {
        return .{ .items = std.ArrayList(Event).init(alloc), .jobs = std.ArrayList(Job).init(alloc) };
    }
    fn push(self: *Queue, ev: Event) void {
        self.mu.lock();
        defer self.mu.unlock();
        self.items.append(ev) catch {};
    }
    fn drain(self: *Queue, into: *std.ArrayList(Event)) void {
        self.mu.lock();
        defer self.mu.unlock();
        into.appendSlice(self.items.items) catch {};
        self.items.clearRetainingCapacity();
    }
    fn pushJob(self: *Queue, j: Job) void {
        self.mu.lock();
        self.jobs.append(j) catch {};
        self.mu.unlock();
        self.cond.signal();
    }
    fn popJob(self: *Queue) Job {
        self.mu.lock();
        defer self.mu.unlock();
        while (self.jobs.items.len == 0) self.cond.wait(&self.mu);
        const j = self.jobs.items[0];
        _ = self.jobs.orderedRemove(0);
        return j;
    }
    fn queuedJobs(self: *Queue) usize {
        self.mu.lock();
        defer self.mu.unlock();
        return self.jobs.items.len;
    }
};

// ---------- scrollback blocks ----------

const ToolBlk = struct {
    name: []const u8,
    summary: []const u8,
    preview: []const u8 = "",
    ok: bool = true,
    finished: bool = false,
};

const UserBlk = struct {
    text: []const u8,
};

const AccBlk = struct {
    text: std.ArrayList(u8),
};

const RsiBlk = struct {
    prop: *rsi.Proposal,
    decided: enum { pending, accepted, rejected } = .pending,
};

const Block = union(enum) {
    user: UserBlk,
    assistant: AccBlk, // streams in
    thinking: AccBlk,
    tool: ToolBlk,
    rsi: RsiBlk,
    sys: []const u8,
    banner, // the gradient logo
};

// ---------- click zones / overlays ----------

const ClickAction = union(enum) {
    accept_rsi: *rsi.Proposal,
    reject_rsi: *rsi.Proposal,
    select_model: []const u8,
    close_overlay,
};

const ClickZone = struct {
    row: usize,
    col0: usize,
    col1: usize,
    act: ClickAction,
};

/// a click zone inside the chat scrollback, recorded against a chat-line
/// index (translated to a terminal row by render() after scrolling)
const ChatZone = struct {
    line: usize,
    col0: usize,
    col1: usize,
    act: ClickAction,
};

const Overlay = enum { none, model_picker };

// ---------- app ----------

pub const App = struct {
    alloc: std.mem.Allocator,
    cfg: *config.Config,
    hclient: *http.HttpClient,
    h: *harness.Harness,
    cwd: []const u8,
    term: *tui.Term,

    q: Queue,
    blocks: std.ArrayList(Block),
    hist: std.ArrayList([]const u8),
    hist_idx: usize = 0,
    composer: std.ArrayList(u8),
    cursor: usize = 0, // byte index into composer
    scroll: usize = 0, // lines up from bottom
    busy: bool = false,
    busy_start: i64 = 0,
    tool_count: usize = 0,
    ag_ptr: std.atomic.Value(?*agent.Agent) = std.atomic.Value(?*agent.Agent).init(null),
    pending_approval: ?*Approval = null,
    auto: bool = false,
    banner_anim: usize = 0, // frames elapsed of the matrix reveal
    slash_sel: usize = 0,
    slash_off: bool = false, // esc-dismissed until composer changes
    quit: bool = false,
    dirty: bool = true,
    full_redraw: bool = true,
    /// click zones inside the chat, recorded against chat-line indices and
    /// translated to terminal rows in render()
    chat_zones: std.ArrayList(ChatZone) = undefined,
    session_id: []const u8 = "",
    slog: ?*session.Log = null,
    livelog: ?*live.Live = null,
    /// coordination fabric shared by every channel agent + subagent
    coord: coord_mod.Coordinator,
    /// supermemory client shared by all agents (null = unconfigured)
    mem: ?*memory_mod.Memory = null,
    /// telemetry store shared by all agents (~/.xera-rsi/traces)
    traces_store: ?*traces_mod.Store = null,
    /// rsi: proposal awaiting accept/reject (one at a time)
    pending_rsi: ?*rsi.Proposal = null,
    /// "model is improving itself" — training indicator
    training: bool = false,
    train_label: []const u8 = "",
    /// last user prompt (fallback focus for /rsi-loop)
    last_prompt: []const u8 = "",

    // ---- ui state ----
    name: []const u8 = "you",
    overlay: Overlay = .none,
    clicks: std.ArrayList(ClickZone),
    model: []const u8,
    models: [][]const u8 = &.{},
    model_filter: std.ArrayList(u8),
    model_filter_cursor: usize = 0,
    model_sel: usize = 0,
    model_loading: bool = false,
    model_error: []const u8 = "",
    overlay_cursor: ?Pos = null,

    prev: std.ArrayList([]const u8),
    prev_arena: std.heap.ArenaAllocator,

    pub fn run(
        alloc: std.mem.Allocator,
        cfg: *config.Config,
        hclient: *http.HttpClient,
        h: *harness.Harness,
        cwd: []const u8,
        resume_msgs: ?std.ArrayList(std.json.Value),
    ) !void {
        var term = try tui.Term.enter(alloc);
        defer term.leave();

        var log = try session.create(alloc, cfg, cwd);

        var app = App{
            .alloc = alloc,
            .cfg = cfg,
            .hclient = hclient,
            .h = h,
            .cwd = cwd,
            .term = &term,
            .q = Queue.init(alloc),
            .blocks = std.ArrayList(Block).init(alloc),
            .hist = std.ArrayList([]const u8).init(alloc),
            .composer = std.ArrayList(u8).init(alloc),
            .prev = std.ArrayList([]const u8).init(alloc),
            .prev_arena = std.heap.ArenaAllocator.init(alloc),
            .session_id = log.id,
            .slog = &log,
            .auto = cfg.approval == .full_auto,
            .clicks = std.ArrayList(ClickZone).init(alloc),
            .chat_zones = std.ArrayList(ChatZone).init(alloc),
            .model = cfg.model,
            .coord = coord_mod.Coordinator.init(alloc),
            .model_filter = std.ArrayList(u8).init(alloc),
            .name = std.process.getEnvVarOwned(alloc, "XERA_NAME") catch
                std.process.getEnvVarOwned(alloc, "USERNAME") catch
                std.process.getEnvVarOwned(alloc, "USER") catch "anon",
        };
        defer app.prev_arena.deinit();

        // telemetry: every turn/tool/skill lands in ~/.xera-rsi/traces
        app.traces_store = traces_mod.Store.create(alloc);

        if (live.Live.create(alloc, log.id) catch null) |lv| {
            app.livelog = lv;
            lv.sessionStart(cwd, cfg.model, h.dir, app.name);
        }

        // long-term memory: one supermemory namespace per workspace,
        // shared by every channel agent + subagent
        app.mem = memory_mod.Memory.create(alloc, hclient, cfg.memory_key, cfg.memory_base,
            if (cfg.memory_namespace.len > 0) cfg.memory_namespace
            else memory_mod.Memory.defaultNs(alloc, cwd));

        // welcome: gradient logo + tagline
        try app.blocks.append(.banner);
        try app.addSysFmt("◆ xera-rsi — model {s} · harness {s}", .{
            cfg.model,
            if (h.global) "global" else "project",
        });
        try app.addSys("type a task, /help for commands, ctrl+c to quit");

        if (resume_msgs) |msgs| {
            app.q.pushJob(.{ .load_msgs = msgs });
            try app.addSysFmt("resuming session…", .{});
        }

        const worker = try std.Thread.spawn(.{}, workerMain, .{&app});
        defer worker.join();

        var keys = std.ArrayList(Key).init(alloc);
        var evs = std.ArrayList(Event).init(alloc);

        while (!app.quit) {
            keys.clearRetainingCapacity();
            app.term.pollKeys(alloc, &keys, 40) catch {};
            for (keys.items) |k| app.onKey(k) catch {};
            app.q.drain(&evs);
            for (evs.items) |ev| app.onEvent(ev) catch {};
            evs.clearRetainingCapacity();

            // drive the banner reveal animation
            if (app.banner_anim < banner_anim_frames) {
                app.banner_anim += 1;
                app.dirty = true;
            }

            const ow = app.term.w;
            const oh = app.term.h;
            app.term.querySize() catch {};
            if (app.term.w != ow or app.term.h != oh) app.full_redraw = true;

            if (app.dirty or app.busy or app.full_redraw) {
                try app.render();
                app.dirty = false;
                app.full_redraw = false;
            }
        }
        app.q.pushJob(.quit);
        // if an approval is outstanding, release the worker
        if (app.pending_approval) |a| {
            a.mu.lock();
            a.decision = .deny;
            a.cond.signal();
            a.mu.unlock();
        }
    }

    // ---------- event handling ----------

    fn onEvent(self: *App, ev: Event) !void {
        self.dirty = true;
        switch (ev) {
            .text => |t| {
                const b = try self.openAcc(.assistant);
                try b.text.appendSlice(t);
            },
            .think => |t| {
                const b = try self.openAcc(.thinking);
                try b.text.appendSlice(t);
            },
            .tool_start => |ts| {
                try self.blocks.append(.{ .tool = .{
                    .name = ts.name, .summary = ts.summary,
                } });
                self.tool_count += 1;
            },
            .tool_result => |tr| {
                // attach to most recent unfinished tool block
                var i = self.blocks.items.len;
                while (i > 0) : (i -= 1) {
                    switch (self.blocks.items[i - 1]) {
                        .tool => |*tb| {
                            if (!tb.finished) {
                                tb.preview = tr.preview;
                                tb.ok = tr.ok;
                                tb.finished = true;
                                break;
                            }
                        },
                        else => {},
                    }
                }
            },
            .status => |s| try self.addSysFmt("· {s}", .{s}),
            .done => |outcome| {
                self.busy = false;
                try self.addSysFmt("— done · {d} turn(s) · {d} tools · {d}s", .{
                    outcome.iterations,
                    self.tool_count,
                    @divTrunc(std.time.milliTimestamp() - self.busy_start, 1000),
                });
                self.maybeNextJob();
            },
            .failed => |e| {
                self.busy = false;
                try self.addSysFmt("✗ {s}", .{e});
                self.maybeNextJob();
            },
            .approval => |a| self.pending_approval = a,
            .cleared => {
                self.blocks.clearRetainingCapacity();
                try self.addSys("history cleared");
            },
            .models => |catalog| {
                self.models = catalog.models;
                self.model_error = catalog.error_msg;
                self.model_loading = false;
                self.model_sel = 0;
                if (catalog.error_msg.len > 0) try self.addSysFmt("✗ {s}", .{catalog.error_msg});
            },
            .model_changed => |model| {
                self.model = model;
                try self.updateWelcomeModel(model);
                try self.addSysFmt("model switched to {s}", .{model});
            },
            .rsi_proposal => |prop| {
                self.pending_rsi = prop;
                try self.blocks.append(.{ .rsi = .{ .prop = prop } });
            },
            .rsi_done => {
                self.busy = false;
                self.maybeNextJob();
            },
            .training => |t| {
                self.training = t.on;
                self.train_label = t.label;
            },
        }
    }

    const AccKind = enum { assistant, thinking };

    /// If the worker already popped a queued prompt, keep the busy UI alive.
    fn maybeNextJob(self: *App) void {
        if (self.q.queuedJobs() > 0) {
            self.busy = true;
            self.busy_start = std.time.milliTimestamp();
            self.tool_count = 0;
        }
    }

    fn openAcc(self: *App, kind: AccKind) !*AccBlk {
        if (self.blocks.items.len > 0) {
            const last = &self.blocks.items[self.blocks.items.len - 1];
            switch (last.*) {
                .assistant => |*al| if (kind == .assistant) return al,
                .thinking => |*al| if (kind == .thinking) return al,
                else => {},
            }
        }
        const blk: Block = if (kind == .assistant)
            .{ .assistant = .{ .text = std.ArrayList(u8).init(self.alloc) } }
        else
            .{ .thinking = .{ .text = std.ArrayList(u8).init(self.alloc) } };
        try self.blocks.append(blk);
        const last = &self.blocks.items[self.blocks.items.len - 1];
        return switch (last.*) {
            .assistant => |*al| al,
            .thinking => |*al| al,
            else => unreachable,
        };
    }

    fn addSys(self: *App, s: []const u8) !void {
        try self.blocks.append(.{ .sys = s });
    }
    fn addSysFmt(self: *App, comptime fmt: []const u8, args: anytype) !void {
        try self.blocks.append(.{ .sys = try std.fmt.allocPrint(self.alloc, fmt, args) });
    }

    fn updateWelcomeModel(self: *App, model: []const u8) !void {
        for (self.blocks.items) |*block| {
            switch (block.*) {
                .sys => |*text| {
                    if (std.mem.startsWith(u8, text.*, "◆ xera-rsi — model ")) {
                        text.* = try std.fmt.allocPrint(self.alloc, "◆ xera-rsi — model {s} · harness {s}", .{
                            model, if (self.h.global) "global" else "project",
                        });
                        return;
                    }
                },
                else => {},
            }
        }
    }

    // ---------- input ----------

    fn onKey(self: *App, k: Key) !void {
        self.dirty = true;
        // approval modal consumes keys first
        if (self.pending_approval) |a| {
            const dec: ?tools.ApprovalDecision = switch (k) {
                .char => |c| switch (c) {
                    'y', 'Y' => .allow,
                    'n', 'N' => .deny,
                    'a', 'A' => .always,
                    else => null,
                },
                .enter => .allow,
                .escape => .deny,
                else => null,
            };
            if (dec) |d| {
                a.mu.lock();
                a.decision = d;
                a.cond.signal();
                a.mu.unlock();
                self.pending_approval = null;
                try self.addSysFmt("{s} {s}", .{
                    if (d == .deny) "denied" else if (d == .always) "always allow" else "approved",
                    a.label,
                });
            }
            return;
        }
        // an rsi proposal awaiting a decision consumes its keys first
        if (self.pending_rsi) |prop| {
            const accept: ?bool = switch (k) {
                .char => |c| switch (c) {
                    'a', 'A', 'y', 'Y' => true,
                    'r', 'R', 'n', 'N' => false,
                    else => null,
                },
                .enter => true,
                .escape => false,
                else => null,
            };
            if (accept) |v| {
                try self.decideRsi(prop, v);
                return;
            }
        }
        if (self.overlay == .model_picker) {
            try self.modelPickerKey(k);
            return;
        }
        // other overlays consume keys (esc closes, clicks hit-test below)
        if (self.overlay != .none) {
            switch (k) {
                .escape, .enter => self.overlay = .none,
                .mouse => |m| {
                    if (m.move) self.hoverAt(m.col, m.row) else if (m.down) self.clickAt(m.col, m.row);
                },
                else => {},
            }
            return;
        }
        if (k == .mouse) {
            const m = k.mouse;
            if (m.move) self.hoverAt(m.col, m.row) else if (m.down) self.clickAt(m.col, m.row);
            return;
        }
        // slash dropdown owns navigation while open
        if (self.slashOpen()) {
            var mbuf: [slash_cmds.len]usize = undefined;
            const n = self.slashMatches(&mbuf);
            if (self.slash_sel >= n) self.slash_sel = 0;
            switch (k) {
                .up => {
                    if (n > 0) self.slash_sel = (self.slash_sel + n - 1) % n;
                    return;
                },
                .down => {
                    if (n > 0) self.slash_sel = (self.slash_sel + 1) % n;
                    return;
                },
                .tab, .enter => {
                    if (n > 0) {
                        const cmd = slash_cmds[mbuf[self.slash_sel]];
                        if (k == .tab or cmd.takes_arg) {
                            self.composer.clearRetainingCapacity();
                            try self.composer.appendSlice(cmd.name);
                            try self.composer.append(' ');
                            self.cursor = self.composer.items.len;
                            self.slash_off = true;
                        } else {
                            self.composer.clearRetainingCapacity();
                            self.cursor = 0;
                            try self.runSlash(try self.alloc.dupe(u8, cmd.name));
                        }
                        return;
                    }
                },
                .escape => {
                    self.slash_off = true;
                    return;
                },
                else => {},
            }
        }
        switch (k) {
            .char => |c| try self.insertChar(c),
            .enter => try self.submit(),
            .backspace => self.backspace(),
            .delete => self.delForward(),
            .left => self.moveCursor(-1),
            .right => self.moveCursor(1),
            .home => self.cursor = 0,
            .end => self.cursor = self.composer.items.len,
            .up => self.histMove(-1),
            .down => self.histMove(1),
            .page_up => self.scrollBy(10),
            .page_down => self.scrollBy(-10),
            .escape => {
                if (self.busy) {
                    if (self.ag_ptr.load(.acquire)) |a| a.interrupted.store(true, .release);
                } else {
                    self.composer.clearRetainingCapacity();
                    self.cursor = 0;
                }
            },
            .tab => {},
            .ctrl_c => {
                if (self.busy) {
                    if (self.ag_ptr.load(.acquire)) |a| a.interrupted.store(true, .release);
                } else self.quit = true;
            },
            .ctrl_d => if (self.composer.items.len == 0) {
                self.quit = true;
            },
            .ctrl_l => self.full_redraw = true,
            .resize => self.full_redraw = true,
            .paste => |p| try self.insertText(p),
            .mouse => {}, // handled above
        }
    }

    /// mark the pending proposal decided, then queue the loop (accept) or drop it (reject)
    fn decideRsi(self: *App, prop: *rsi.Proposal, accept: bool) !void {
        self.pending_rsi = null;
        for (self.blocks.items) |*b| {
            switch (b.*) {
                .rsi => |*rb| {
                    if (rb.prop == prop) {
                        rb.decided = if (accept) .accepted else .rejected;
                        break;
                    }
                },
                else => {},
            }
        }
        if (accept) {
            if (!self.busy) {
                self.busy = true;
                self.busy_start = std.time.milliTimestamp();
                self.tool_count = 0;
            }
            self.q.pushJob(.{ .rsi_run = prop });
            try self.addSys("◈ rsi loop accepted — analyzing traces…");
        } else {
            try self.addSys("◈ rsi loop rejected");
        }
    }

    /// dropdown is live while the composer is exactly `/partial-name`
    fn slashOpen(self: *App) bool {
        if (self.slash_off) return false;
        const b = self.composer.items;
        return b.len > 0 and b[0] == '/' and
            std.mem.indexOfAny(u8, b, " \n") == null;
    }

    /// indices into slash_cmds whose name starts with the typed prefix
    fn slashMatches(self: *App, buf: *[slash_cmds.len]usize) usize {
        const typed = self.composer.items;
        var n: usize = 0;
        for (slash_cmds, 0..) |c, i| {
            if (std.mem.startsWith(u8, c.name, typed)) {
                buf[n] = i;
                n += 1;
            }
        }
        return n;
    }

    fn insertChar(self: *App, cp: u21) !void {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(cp, &buf) catch return;
        try self.composer.insertSlice(self.cursor, buf[0..n]);
        self.cursor += n;
        self.slash_off = false;
    }

    fn insertText(self: *App, text: []const u8) !void {
        var i: usize = 0;
        while (i < text.len) {
            const c = text[i];
            // keep every line break: a paste must not silently join lines
            if (c == '\r') {
                if (i + 1 < text.len and text[i + 1] == '\n') i += 1; // CRLF -> LF
                try self.composer.insertSlice(self.cursor, "\n");
                self.cursor += 1;
                i += 1;
                continue;
            }
            if (c == '\n') {
                try self.composer.insertSlice(self.cursor, "\n");
                self.cursor += 1;
                i += 1;
                continue;
            }
            const len = if (c < 0x80) 1 else (std.unicode.utf8ByteSequenceLength(c) catch 1);
            try self.composer.insertSlice(self.cursor, text[i .. i + @min(len, text.len - i)]);
            self.cursor += len;
            i += len;
        }
        self.slash_off = false;
    }

    fn backspace(self: *App) void {
        if (self.cursor == 0) return;
        var s = self.cursor - 1;
        while (s > 0 and (self.composer.items[s] & 0xC0) == 0x80) s -= 1;
        const new_len = self.composer.items.len - (self.cursor - s);
        std.mem.copyForwards(u8, self.composer.items[s..new_len], self.composer.items[self.cursor..]);
        self.composer.items.len = new_len;
        self.cursor = s;
        self.slash_off = false;
    }

    fn delForward(self: *App) void {
        if (self.cursor >= self.composer.items.len) return;
        var e = self.cursor + 1;
        while (e < self.composer.items.len and (self.composer.items[e] & 0xC0) == 0x80) e += 1;
        std.mem.copyForwards(
            u8,
            self.composer.items[self.cursor .. self.composer.items.len - (e - self.cursor)],
            self.composer.items[e..],
        );
        self.composer.items.len -= e - self.cursor;
        self.slash_off = false;
    }

    fn moveCursor(self: *App, dir: i32) void {
        if (dir < 0) {
            var s = self.cursor;
            if (s == 0) return;
            s -= 1;
            while (s > 0 and (self.composer.items[s] & 0xC0) == 0x80) s -= 1;
            self.cursor = s;
        } else {
            var e = self.cursor;
            if (e >= self.composer.items.len) return;
            e += 1;
            while (e < self.composer.items.len and (self.composer.items[e] & 0xC0) == 0x80) e += 1;
            self.cursor = e;
        }
    }

    fn histMove(self: *App, dir: i32) void {
        if (self.hist.items.len == 0) return;
        var idx: i64 = @intCast(self.hist_idx);
        idx += dir;
        const hi: i64 = @intCast(self.hist.items.len);
        if (idx < 0) idx = 0;
        if (idx > hi) idx = hi;
        self.hist_idx = @intCast(idx);
        self.composer.clearRetainingCapacity();
        if (self.hist_idx < self.hist.items.len) {
            self.composer.appendSlice(self.hist.items[self.hist_idx]) catch {};
        }
        self.cursor = self.composer.items.len;
    }

    fn scrollBy(self: *App, d: i32) void {
        var s: i64 = @intCast(self.scroll);
        s += d;
        if (s < 0) s = 0;
        self.scroll = @intCast(s);
    }

    fn modelMatches(model: []const u8, query: []const u8) bool {
        if (query.len == 0) return true;
        if (query.len > model.len) return false;
        for (0..model.len - query.len + 1) |i| {
            if (std.ascii.eqlIgnoreCase(model[i .. i + query.len], query)) return true;
        }
        return false;
    }

    fn filteredModelCount(self: *const App) usize {
        var count: usize = 0;
        for (self.models) |model| if (modelMatches(model, self.model_filter.items)) {
            count += 1;
        };
        return count;
    }

    fn filteredModel(self: *const App, selected: usize) ?[]const u8 {
        var index: usize = 0;
        for (self.models) |model| {
            if (!modelMatches(model, self.model_filter.items)) continue;
            if (index == selected) return model;
            index += 1;
        }
        return null;
    }

    fn modelPickerKey(self: *App, k: Key) !void {
        switch (k) {
            .escape => self.overlay = .none,
            .enter => if (self.filteredModel(self.model_sel)) |model| self.chooseModel(model),
            .up => self.moveModelSelection(-1),
            .down => self.moveModelSelection(1),
            .page_up => self.moveModelSelection(-10),
            .page_down => self.moveModelSelection(10),
            .char => |cp| try self.insertModelFilter(cp),
            .paste => |text| try self.insertModelFilterText(text),
            .backspace => self.modelFilterBackspace(),
            .delete => self.modelFilterDelete(),
            .left => self.modelFilterMove(-1),
            .right => self.modelFilterMove(1),
            .home => self.model_filter_cursor = 0,
            .end => self.model_filter_cursor = self.model_filter.items.len,
            .mouse => |m| {
                if (m.move) self.hoverAt(m.col, m.row) else if (m.down) self.clickAt(m.col, m.row);
            },
            .resize => self.full_redraw = true,
            else => {},
        }
        self.dirty = true;
    }

    fn moveModelSelection(self: *App, delta: i32) void {
        const count = self.filteredModelCount();
        if (count == 0) return;
        const n: i64 = @intCast(count);
        var next: i64 = @intCast(self.model_sel);
        next = @mod(next + delta, n);
        self.model_sel = @intCast(next);
    }

    fn modelFilterMaxBytes(self: *App) usize {
        return @max(1, @min(48, @as(usize, self.term.w) -| 18));
    }

    fn insertModelFilter(self: *App, cp: u21) !void {
        var buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(cp, &buf) catch return;
        if (self.model_filter.items.len + n > self.modelFilterMaxBytes()) return;
        try self.model_filter.insertSlice(self.model_filter_cursor, buf[0..n]);
        self.model_filter_cursor += n;
        self.model_sel = 0;
    }

    fn insertModelFilterText(self: *App, text: []const u8) !void {
        var i: usize = 0;
        while (i < text.len) {
            if (text[i] < 0x20 or text[i] == 0x7f) {
                i += 1;
                continue;
            }
            const len = if (text[i] < 0x80) 1 else (std.unicode.utf8ByteSequenceLength(text[i]) catch 1);
            const take = @min(len, text.len - i);
            if (self.model_filter.items.len + take <= self.modelFilterMaxBytes()) {
                try self.model_filter.insertSlice(self.model_filter_cursor, text[i .. i + take]);
                self.model_filter_cursor += take;
            }
            i += len;
        }
        self.model_sel = 0;
    }

    fn modelFilterBackspace(self: *App) void {
        if (self.model_filter_cursor == 0) return;
        var start = self.model_filter_cursor - 1;
        while (start > 0 and (self.model_filter.items[start] & 0xC0) == 0x80) start -= 1;
        const removed = self.model_filter_cursor - start;
        const new_len = self.model_filter.items.len - removed;
        std.mem.copyForwards(u8, self.model_filter.items[start..new_len], self.model_filter.items[self.model_filter_cursor..]);
        self.model_filter.items.len = new_len;
        self.model_filter_cursor = start;
        self.model_sel = 0;
    }

    fn modelFilterDelete(self: *App) void {
        if (self.model_filter_cursor >= self.model_filter.items.len) return;
        var end = self.model_filter_cursor + 1;
        while (end < self.model_filter.items.len and (self.model_filter.items[end] & 0xC0) == 0x80) end += 1;
        const removed = end - self.model_filter_cursor;
        std.mem.copyForwards(u8, self.model_filter.items[self.model_filter_cursor .. self.model_filter.items.len - removed], self.model_filter.items[end..]);
        self.model_filter.items.len -= removed;
        self.model_sel = 0;
    }

    fn modelFilterMove(self: *App, direction: i32) void {
        if (direction < 0) {
            if (self.model_filter_cursor == 0) return;
            self.model_filter_cursor -= 1;
            while (self.model_filter_cursor > 0 and (self.model_filter.items[self.model_filter_cursor] & 0xC0) == 0x80)
                self.model_filter_cursor -= 1;
        } else {
            if (self.model_filter_cursor >= self.model_filter.items.len) return;
            self.model_filter_cursor += 1;
            while (self.model_filter_cursor < self.model_filter.items.len and (self.model_filter.items[self.model_filter_cursor] & 0xC0) == 0x80)
                self.model_filter_cursor += 1;
        }
    }

    fn openModelPicker(self: *App) void {
        self.overlay = .model_picker;
        self.model_filter.clearRetainingCapacity();
        self.model_filter_cursor = 0;
        self.model_sel = 0;
        self.model_loading = true;
        self.model_error = "";
        self.q.pushJob(.list_models);
    }

    fn chooseModel(self: *App, model: []const u8) void {
        self.overlay = .none;
        if (std.mem.eql(u8, model, self.model)) {
            self.addSysFmt("{s} is already selected", .{model}) catch {};
            return;
        }
        const owned = self.alloc.dupe(u8, model) catch return;
        self.addSysFmt("switching model to {s}…", .{model}) catch {};
        self.q.pushJob(.{ .set_model = owned });
    }

    fn submit(self: *App) !void {
        const text = std.mem.trim(u8, self.composer.items, " \t\r\n");
        if (text.len == 0) return;
        const cmd = try self.alloc.dupe(u8, text);
        self.composer.clearRetainingCapacity();
        self.cursor = 0;

        if (cmd[0] == '/') {
            try self.runSlash(cmd);
            return;
        }
        try self.hist.append(cmd);
        self.last_prompt = cmd;
        self.hist_idx = self.hist.items.len;
        self.scroll = 0;

        try self.blocks.append(.{ .user = .{ .text = cmd } });
        if (!self.busy) {
            self.busy = true;
            self.busy_start = std.time.milliTimestamp();
            self.tool_count = 0;
        }
        self.q.pushJob(.{ .prompt = cmd });
    }

    fn runSlash(self: *App, cmd: []const u8) !void {
        const sp = std.mem.indexOfScalar(u8, cmd, ' ') orelse cmd.len;
        const name = cmd[0..sp];
        const arg = std.mem.trim(u8, if (sp < cmd.len) cmd[sp + 1 ..] else "", " ");
        if (eqAny(name, &.{ "/exit", "/quit", "/q" })) {
            self.quit = true;
        } else if (eqAny(name, &.{ "/help", "/h", "/?" })) {
            var msg = std.ArrayList(u8).init(self.alloc);
            try msg.appendSlice("commands (type / for the picker):\n");
            for (slash_cmds) |c| {
                try msg.writer().print("  {s: <12} {s}\n", .{ c.name, c.desc });
            }
            try msg.appendSlice("keys: enter send · esc interrupt/clear · pgup/pgdn scroll · " ++
                "up/down history · tab complete · ctrl+l redraw · ctrl+c quit");
            try self.addSys(msg.items);
        } else if (std.mem.eql(u8, name, "/clear")) {
            self.q.pushJob(.reset);
        } else if (std.mem.eql(u8, name, "/auto")) {
            self.auto = !self.auto;
            try self.addSysFmt("auto-approve {s}", .{if (self.auto) "on" else "off"});
        } else if (std.mem.eql(u8, name, "/model")) {
            self.openModelPicker();
        } else if (std.mem.eql(u8, name, "/usage")) {
            self.q.pushJob(.usage);
        } else if (std.mem.eql(u8, name, "/summarize") or std.mem.eql(u8, name, "/compact")) {
            self.q.pushJob(.summarize);
        } else if (std.mem.eql(u8, name, "/skills")) {
            const skills = self.h.listSkills() catch &[_]harness.NamedItem{};
            if (skills.len == 0) {
                try self.addSys("no skills — add skills/<name>/SKILL.md in the harness dir");
            }
            for (skills) |s| {
                try self.addSysFmt("◦ {s}{s}", .{ s.name, if (s.desc.len > 0)
                    try std.fmt.allocPrint(self.alloc, " — {s}", .{s.desc}) else "" });
            }
        } else if (std.mem.eql(u8, name, "/agents")) {
            const agents = self.h.listAgents() catch &[_]harness.NamedItem{};
            if (agents.len == 0) {
                try self.addSys("no named agents — add sub_agents/<name>/AGENT.md in the harness dir");
            }
            for (agents) |a| {
                try self.addSysFmt("◦ {s}{s}", .{ a.name, if (a.desc.len > 0)
                    try std.fmt.allocPrint(self.alloc, " — {s}", .{a.desc}) else "" });
            }
            try self.addSys("spawn: the model calls task(prompt, agent=\"<name>\") · depth ≤ 2");
        } else if (std.mem.eql(u8, name, "/harness")) {
            try self.addSysFmt("harness: {s} ({s})", .{ self.h.dir, if (self.h.global) "global" else "project" });
            for (harness.component_files) |f| {
                const body = self.h.read(f);
                if (body.len > 0) try self.addSysFmt("  {s} · {d} b", .{ f, body.len });
            }
            const skills = self.h.listSkills() catch &[_]harness.NamedItem{};
            const agents = self.h.listAgents() catch &[_]harness.NamedItem{};
            const scripts = self.h.listScripts() catch &[_][]const u8{};
            try self.addSysFmt("  {d} skill(s) · {d} agent(s) · {d} script(s) · hooks {s}", .{
                skills.len, agents.len, scripts.len,
                if (self.h.hooksPath() != null) "on" else "off",
            });
            const problems = self.h.validate(self.alloc) catch &[_][]const u8{};
            for (problems) |p| try self.addSysFmt("  ! {s}", .{p});
        } else if (std.mem.eql(u8, name, "/library")) {
            const home = util.homeDir(self.alloc) catch "";
            const libdir = try std.fs.path.join(self.alloc, &.{ home, ".xera-rsi", "library" });
            var d = std.fs.cwd().openDir(libdir, .{ .iterate = true }) catch {
                try self.addSys("no library yet — adopted harnesses from `improve` land here");
                return;
            };
            defer d.close();
            var it = d.iterate();
            var n: usize = 0;
            while (try it.next()) |e| {
                if (e.kind != .directory) continue;
                try self.addSysFmt("◦ {s}", .{e.name});
                n += 1;
                if (n >= 10) break;
            }
            if (n == 0) try self.addSys("(empty — nothing adopted yet)");
        } else if (std.mem.eql(u8, name, "/sessions")) {
            const infos = session.list(self.alloc, self.cfg) catch {
                try self.addSys("could not list sessions");
                return;
            };
            for (infos[0..@min(infos.len, 8)]) |info| {
                try self.addSysFmt("{s} · {d} turn(s) · {s}", .{ info.id, info.turns, info.cwd });
            }
            if (infos.len == 0) try self.addSys("(no sessions)");
        } else if (std.mem.eql(u8, name, "/resume")) {
            if (arg.len == 0) {
                try self.addSys("usage: /resume <session-id>");
                return;
            }
            const msgs = session.loadMessages(self.alloc, self.cfg, arg) catch |e| {
                try self.addSysFmt("cannot load {s}: {s}", .{ arg, @errorName(e) });
                return;
            };
            self.q.pushJob(.{ .load_msgs = msgs });
        } else if (std.mem.eql(u8, name, "/status")) {
            try self.addSysFmt("session {s} · {d} block(s)", .{ self.session_id, self.blocks.items.len });
        } else if (std.mem.eql(u8, name, "/improve")) {
            try self.addSys("improve runs as its own command:  xera-rsi improve \"task\"");
        } else if (std.mem.eql(u8, name, "/rsi-loop")) {
            try self.cmdRsiLoop(arg);
        } else {
            try self.addSysFmt("unknown command {s} — /help", .{name});
        }
    }

    /// /rsi-loop [focus] — build a proposal and present it for accept/reject,
    /// same as a self-check failure would.
    fn cmdRsiLoop(self: *App, arg: []const u8) !void {
        const focus = if (arg.len > 0) arg else if (self.last_prompt.len > 0)
            self.last_prompt
        else
            "general self-improvement over recent turns";
        if (self.pending_rsi != null) {
            try self.addSys("an rsi proposal is already awaiting a decision (a/enter accept · r/esc reject)");
            return;
        }
        const prop = rsi.buildProposal(self.alloc, focus, "") catch |e| {
            try self.addSysFmt("could not build rsi proposal: {s}", .{@errorName(e)});
            return;
        };
        self.pending_rsi = prop;
        try self.blocks.append(.{ .rsi = .{ .prop = prop } });
    }

    fn hoverAt(self: *App, col: u16, row: u16) void {
        if (self.overlay != .model_picker) return;
        for (self.clicks.items) |zone| {
            if (row != zone.row or col < zone.col0 or col > zone.col1) continue;
            switch (zone.act) {
                .select_model => |model| {
                    for (0..self.filteredModelCount()) |i| {
                        if (self.filteredModel(i)) |candidate| {
                            if (std.mem.eql(u8, candidate, model)) {
                                if (self.model_sel != i) {
                                    self.model_sel = i;
                                    self.dirty = true;
                                }
                                return;
                            }
                        }
                    }
                },
                else => {},
            }
        }
    }

    /// mouse click → hit-test the zones registered by the last render.
    fn clickAt(self: *App, col: u16, row: u16) void {
        for (self.clicks.items) |z| {
            if (row == z.row and col >= z.col0 and col <= z.col1) {
                switch (z.act) {
                    .select_model => |model| self.chooseModel(model),
                    .close_overlay => self.overlay = .none,
                    .accept_rsi => |prop| {
                        if (self.pending_rsi == prop) self.decideRsi(prop, true) catch {};
                    },
                    .reject_rsi => |prop| {
                        if (self.pending_rsi == prop) self.decideRsi(prop, false) catch {};
                    },
                }
                self.dirty = true;
                return;
            }
        }
        // click anywhere while an overlay is open dismisses it
        if (self.overlay != .none) {
            self.overlay = .none;
            self.dirty = true;
        }
    }

    // ---------- worker-facing API (called on worker thread) ----------

    fn pushText(ctx: *anyopaque, text: []const u8) void {
        const app: *App = @ptrCast(@alignCast(ctx));
        const d = app.alloc.dupe(u8, text) catch return;
        app.q.push(.{ .text = d });
    }
    fn pushThink(ctx: *anyopaque, text: []const u8) void {
        const app: *App = @ptrCast(@alignCast(ctx));
        const d = app.alloc.dupe(u8, text) catch return;
        app.q.push(.{ .think = d });
    }
    fn pushToolStart(ctx: *anyopaque, name: []const u8, summary: []const u8) void {
        const app: *App = @ptrCast(@alignCast(ctx));
        app.q.push(.{ .tool_start = .{
            .name = app.alloc.dupe(u8, name) catch return,
            .summary = app.alloc.dupe(u8, summary) catch "",
        } });
    }
    fn pushToolResult(ctx: *anyopaque, name: []const u8, preview: []const u8, ok: bool) void {
        const app: *App = @ptrCast(@alignCast(ctx));
        _ = name;
        app.q.push(.{ .tool_result = .{
            .preview = app.alloc.dupe(u8, preview) catch "",
            .ok = ok,
        } });
    }
    fn pushStatus(ctx: *anyopaque, msg: []const u8) void {
        const app: *App = @ptrCast(@alignCast(ctx));
        const d = app.alloc.dupe(u8, msg) catch return;
        app.q.push(.{ .status = d });
    }
    fn pushTraining(ctx: *anyopaque, on: bool, label: []const u8) void {
        const app: *App = @ptrCast(@alignCast(ctx));
        app.q.push(.{ .training = .{ .on = on, .label = app.alloc.dupe(u8, label) catch "" } });
    }

    fn askApproval(ctx: *anyopaque, tool_name: []const u8, summary: []const u8, danger: tools.Danger) anyerror!tools.ApprovalDecision {
        const app: *App = @ptrCast(@alignCast(ctx));
        if (app.auto) return .allow;
        const a = try app.alloc.create(Approval);
        a.* = .{
            .label = try std.fmt.allocPrint(app.alloc, "{s} {s} ({s})", .{ tool_name, summary, @tagName(danger) }),
            .danger = danger,
        };
        app.q.push(.{ .approval = a });
        a.mu.lock();
        defer a.mu.unlock();
        while (a.decision == null) a.cond.wait(&a.mu);
        return a.decision.?;
    }

    // ---------- rendering ----------

    fn render(self: *App) !void {
        var frame = std.heap.ArenaAllocator.init(self.alloc);
        defer frame.deinit();
        const fa = frame.allocator();

        const w: usize = self.term.w;
        const h: usize = self.term.h;
        self.clicks.clearRetainingCapacity();
        self.chat_zones.clearRetainingCapacity();
        self.overlay_cursor = null;

        var lines = std.ArrayList([]const u8).init(fa);
        try lines.append(try self.statusLine(fa, w));

        // chat
        const chat_lines = try self.renderChat(fa, w);
        const composer = try self.renderComposer(fa, w);
        const ind_row: usize = if (self.busy or self.training) 1 else 0;
        const drop_rows = try self.renderDropdown(fa, w);
        const avail = if (h > 1 + composer.items.len + ind_row + drop_rows.items.len + 1)
            h - 1 - composer.items.len - ind_row - drop_rows.items.len - 1
        else
            1;
        const total = chat_lines.items.len;
        var start: usize = if (total > avail) total - avail else 0;
        if (self.scroll > 0) {
            start = if (self.scroll < start) start - self.scroll else 0;
        }
        // translate chat-embedded zones to the rows they'll actually land on
        for (self.chat_zones.items) |z| {
            if (z.line < start or z.line >= start + avail) continue;
            self.clicks.append(.{
                .row = 1 + z.line - start,
                .col0 = z.col0,
                .col1 = z.col1,
                .act = z.act,
            }) catch {};
        }
        var shown: usize = 0;
        var i = start;
        while (i < total and shown < avail) : (i += 1) {
            try lines.append(chat_lines.items[i]);
            shown += 1;
        }
        // top-pad with scroll hint
        if (start > 0) {
            lines.items[1] = try std.fmt.allocPrint(fa, "{s}↑ {d} earlier line(s) — pgup/pgdn to scroll{s}", .{ sty.gray, start, sty.reset });
        }
        while (shown < avail) : (shown += 1) try lines.append("");

        // busy / training indicator
        if (self.busy) {
            const el: i64 = @divTrunc(std.time.milliTimestamp() - self.busy_start, 1000);
            const frame_i: usize = @intCast(@mod(@divTrunc(std.time.milliTimestamp(), 80), @as(i64, @intCast(spinner_frames.len / 3))));
            const spin = spinner_frames[frame_i * 3 .. frame_i * 3 + 3];
            const qn = self.q.queuedJobs();
            try lines.append(try std.fmt.allocPrint(
                fa,
                " {s}{s}{s} working {s}{d}s{s} · {d} tool call(s){s}{s}",
                .{
                    sty.cyan,
                    spin,
                    sty.reset,
                    sty.gray,
                    el,
                    sty.reset,
                    self.tool_count,
                    if (qn > 0) try std.fmt.allocPrint(fa, " {s}· {d} queued{s}", .{ sty.yellow, qn, sty.reset }) else "",
                    if (self.training) try std.fmt.allocPrint(fa, " {s}◈ {s}{s}", .{ sty.magenta, self.train_label, sty.reset }) else "",
                },
            ));
        } else if (self.training) {
            try lines.append(try std.fmt.allocPrint(
                fa,
                " {s}◈ model is improving itself{s} {s}{s}{s}",
                .{ sty.magenta ++ sty.bold, sty.reset, sty.gray, self.train_label, sty.reset },
            ));
        }

        // dropdown rows sit directly above the composer
        for (drop_rows.items) |dl| try lines.append(dl);

        // composer
        for (composer.items) |cl| try lines.append(cl);

        // hint / approval row
        if (self.pending_approval) |a| {
            try lines.append(try std.fmt.allocPrint(
                fa,
                " {s}⚠ allow?{s} {s}  {s}[y]es [n]o [a]lways{s}",
                .{ sty.yellow ++ sty.bold, sty.reset, a.label, sty.gray, sty.reset },
            ));
        } else {
            try lines.append(try std.fmt.allocPrint(
                fa,
                " {s}enter send · esc {s} · /help{s}",
                .{ sty.gray, if (self.busy) "interrupt" else "clear", sty.reset },
            ));
        }

        // modal overlays draw on top
        if (self.overlay != .none) try self.renderOverlay(fa, &lines, w, h);

        // diff vs previous frame
        const cursor = self.overlay_cursor orelse try self.cursorPos(fa, composer.items.len, ind_row + drop_rows.items.len);
        try self.paint(lines.items, h, cursor);
    }

    /// Draw a centered overlay box over `lines` (mutating rows inside the box).
    /// Content rows may carry a ClickAction (parallel `acts` list).
    fn renderOverlay(self: *App, fa: std.mem.Allocator, lines: *std.ArrayList([]const u8), w: usize, h: usize) !void {
        const bw: usize = @min(w -| 4, 72);
        const bx: usize = (w - bw) / 2;
        var content = std.ArrayList([]const u8).init(fa);
        var acts = std.ArrayList(?ClickAction).init(fa);
        const addc = struct {
            fn f(c: *std.ArrayList([]const u8), acs: *std.ArrayList(?ClickAction), l: []const u8, a: ?ClickAction) !void {
                try c.append(l);
                try acs.append(a);
            }
        }.f;
        switch (self.overlay) {
            .none => return,
            .model_picker => {
                try addc(&content, &acts, try std.fmt.allocPrint(fa, "{s}◈ Select a model · current: {s}{s}", .{
                    sty.accent ++ sty.bold, self.model, sty.reset }), null);
                try addc(&content, &acts, try std.fmt.allocPrint(fa, "{s}provider: {s}{s}", .{
                    sty.gray, try clip(fa, self.cfg.base_url, bw -| 16), sty.reset }), null);
                const search_prefix = "⌕ search: ";
                try addc(&content, &acts, try std.fmt.allocPrint(fa, "{s}{s}{s}{s}", .{
                    sty.cyan, search_prefix, self.model_filter.items, sty.reset }), null);
                try addc(&content, &acts, try std.fmt.allocPrint(fa, "{s}type to filter · ↑↓ navigate · hover/click or enter · esc close{s}", .{
                    sty.gray, sty.reset }), null);
                if (self.model_loading) {
                    try addc(&content, &acts, try std.fmt.allocPrint(fa, "{s}loading models from provider…{s}", .{ sty.gray, sty.reset }), null);
                } else if (self.model_error.len > 0) {
                    try addc(&content, &acts, try std.fmt.allocPrint(fa, "{s}✗ {s}{s}", .{ sty.red, self.model_error, sty.reset }), null);
                } else {
                    const count = self.filteredModelCount();
                    if (count == 0) {
                        const empty_msg = if (self.models.len == 0) "provider returned no models" else "no matching models";
                        try addc(&content, &acts, try std.fmt.allocPrint(fa, "{s}{s}{s}", .{ sty.gray, empty_msg, sty.reset }), null);
                    } else {
                        if (self.model_sel >= count) self.model_sel = 0;
                        const max_rows = @max(1, @min(10, h -| 13));
                        var top: usize = 0;
                        if (count > max_rows and self.model_sel >= max_rows)
                            top = self.model_sel - max_rows + 1;
                        const end = @min(count, top + max_rows);
                        for (top..end) |index| {
                            const model = self.filteredModel(index) orelse continue;
                            const selected = index == self.model_sel;
                            const current = std.mem.eql(u8, model, self.model);
                            const prefix: []const u8 = if (selected) "› " else "  ";
                            const suffix: []const u8 = if (current) "  (current)" else "";
                            const label = try clip(fa, model, bw -| 22);
                            const row = try std.fmt.allocPrint(fa, "{s}{s}{s}{s}{s}{s}", .{
                                if (selected) sty.accent ++ sty.bold else sty.gray,
                                prefix,
                                if (current) sty.green else sty.reset,
                                label,
                                sty.reset,
                                suffix,
                            });
                            try addc(&content, &acts, row, .{ .select_model = model });
                        }
                        if (count > max_rows) {
                            try addc(&content, &acts, try std.fmt.allocPrint(fa, "{s}{d} models · showing {d}–{d}{s}", .{
                                sty.gray, count, top + 1, end, sty.reset }), null);
                        } else {
                            try addc(&content, &acts, try std.fmt.allocPrint(fa, "{s}{d} model(s) available{s}", .{
                                sty.gray, count, sty.reset }), null);
                        }
                    }
                }
            },
        }

        const ch = content.items.len + 2;
        const by = if (h > ch + 4) (h - ch) / 2 else 1;
        if (self.overlay == .model_picker) {
            const search_col = bx + 2 + tui.displayWidth("⌕ search: ") + tui.displayWidth(self.model_filter.items[0..self.model_filter_cursor]);
            self.overlay_cursor = .{
                .row = @min(by + 3, h -| 1),
                .col = @min(search_col, w -| 1),
            };
        }
        for (0..ch) |r| {
            const row = by + r;
            if (row >= lines.items.len) break;
            const box: []const u8 = if (r == 0)
                try std.fmt.allocPrint(fa, "{s}╭{s}╮{s}", .{ sty.accent, try hline(fa, "─", bw - 2), sty.reset })
            else if (r == ch - 1)
                try std.fmt.allocPrint(fa, "{s}╰{s}╯{s}", .{ sty.accent, try hline(fa, "─", bw - 2), sty.reset })
            else blk: {
                const inner: []const u8 = if (r - 1 < content.items.len) content.items[r - 1] else "";
                const dw = tui.displayWidth(inner);
                const pad = if (bw - 3 > dw) bw - 3 - dw else 0;
                break :blk try std.fmt.allocPrint(fa, "{s}│{s} {s}{s}{s}│{s}", .{
                    sty.accent, sty.reset, inner, try spaces(fa, pad), sty.accent, sty.reset,
                });
            };
            lines.items[row] = try std.fmt.allocPrint(fa, "{s}{s}", .{ try spaces(fa, bx), box });
            if (r > 0 and r < ch - 1 and r - 1 < acts.items.len) {
                if (acts.items[r - 1]) |act| {
                    try self.clicks.append(.{ .row = row, .col0 = bx, .col1 = bx + bw -| 1, .act = act });
                }
            }
        }
    }

    fn hline(fa: std.mem.Allocator, ch: []const u8, n: usize) ![]const u8 {
        var out = std.ArrayList(u8).init(fa);
        for (0..n) |_| try out.appendSlice(ch);
        return out.items;
    }

    /// rows of the `/command` picker — empty list when closed
    fn renderDropdown(self: *App, fa: std.mem.Allocator, w: usize) !std.ArrayList([]const u8) {
        var out = std.ArrayList([]const u8).init(fa);
        if (!self.slashOpen()) return out;
        var mbuf: [slash_cmds.len]usize = undefined;
        const n = self.slashMatches(&mbuf);
        if (n == 0) {
            try out.append(try std.fmt.allocPrint(fa, "  {s}no match — esc closes{s}", .{ sty.gray, sty.reset }));
            return out;
        }
        if (self.slash_sel >= n) self.slash_sel = 0;
        const shown_max: usize = 7;
        // keep the selected row in view
        var top: usize = 0;
        if (n > shown_max and self.slash_sel >= shown_max) top = self.slash_sel - shown_max + 1;
        var shown: usize = 0;
        for (mbuf[top..n], top..) |ci, ri| {
            if (shown >= shown_max) break;
            const c = slash_cmds[ci];
            const sel = ri == self.slash_sel;
            const marker = if (sel) "›" else " ";
            const name_sty = if (sel) sty.accent ++ sty.bold else "";
            const ln = try std.fmt.allocPrint(fa, "  {s}{s}{s} {s}{s: <12}{s} {s}{s}{s}", .{
                if (sel) sty.accent else sty.gray,
                marker,
                sty.reset,
                name_sty,
                c.name,
                sty.reset,
                sty.gray,
                c.desc,
                sty.reset,
            });
            try out.append(ln);
            shown += 1;
        }
        if (n > shown_max) {
            try out.append(try std.fmt.allocPrint(fa, "  {s}  … {d} more{s}", .{ sty.gray, n - shown_max, sty.reset }));
        }
        _ = w;
        return out;
    }

    fn statusLine(self: *App, fa: std.mem.Allocator, w: usize) ![]const u8 {
        const left = try std.fmt.allocPrint(fa, " {s}{s} xera-rsi{s} {s}{s}{s} {s}{s}{s}", .{
            sty.bright_cyan,
            "◆",
            sty.reset,
            sty.gray,
            self.model,
            sty.reset,
            sty.gray,
            try util.shortenPath(fa, self.cwd, @max(16, w / 3)),
            sty.reset,
        });
        const right = try std.fmt.allocPrint(fa, "{s}{s}{s} {s}{s}{s} ", .{
            sty.gray,
            self.session_id,
            sty.reset,
            sty.gray,
            if (self.auto) "auto" else "ask",
            sty.reset,
        });
        const lw = tui.displayWidth(left);
        const rw = tui.displayWidth(right);
        const pad = if (w > lw + rw) w - lw - rw else 1;
        return std.fmt.allocPrint(fa, "{s}{s}{s}", .{ left, try spaces(fa, pad), right });
    }

    fn renderChat(self: *App, fa: std.mem.Allocator, w: usize) !std.ArrayList([]const u8) {
        var out = std.ArrayList([]const u8).init(fa);
        const cw = if (w > 4) w - 1 else w;
        for (self.blocks.items) |b| {
            switch (b) {
                .banner => {
                    try renderBanner(fa, w, self.banner_anim, &out);
                    const tag = try std.fmt.allocPrint(fa, "{s}a zig agent harness · harness² inside{s}", .{ sty.gray, sty.reset });
                    const tw = tui.displayWidth(tag);
                    try out.append(try std.fmt.allocPrint(fa, "{s}{s}", .{ try spaces(fa, if (w > tw) (w - tw) / 2 else 0), tag }));
                    try out.append("");
                },
                .user => |u| {
                    const wrapped = try tui.wrapText(fa, u.text, if (cw > 2) cw - 2 else cw);
                    for (wrapped.items, 0..) |ln, i| {
                        if (i == 0) {
                            try out.append(try std.fmt.allocPrint(fa, " {s}❯{s} {s}", .{
                                sty.accent ++ sty.bold, sty.reset, ln,
                            }));
                        } else {
                            try out.append(try std.fmt.allocPrint(fa, "   {s}", .{ln}));
                        }
                    }
                },
                .assistant => |al| {
                    const md = try mdLines(fa, al.text.items);
                    for (md.items) |ln| {
                        const wrapped = try tui.wrapLine(fa, ln, cw);
                        for (wrapped.items) |wl| {
                            try out.append(try std.fmt.allocPrint(fa, " {s}", .{wl}));
                        }
                    }
                },
                .thinking => |al| {
                    const wrapped = try tui.wrapText(fa, al.text.items, if (cw > 2) cw - 2 else cw);
                    for (wrapped.items) |ln| {
                        try out.append(try std.fmt.allocPrint(fa, "  {s}{s}{s}", .{ sty.dim ++ sty.italic, ln, sty.reset }));
                    }
                },
                .tool => |tb| {
                    const sum_max = if (w > 20) w - 14 else w;
                    const sum = if (tui.displayWidth(tb.summary) > sum_max)
                        try clip(fa, tb.summary, sum_max)
                    else
                        tb.summary;
                    try out.append(try std.fmt.allocPrint(fa, " ⚙{s} {s}{s}{s} {s}{s}{s}", .{
                        sty.gray ++ "⚙" ++ sty.reset ++ " ",
                        sty.cyan,
                        tb.name,
                        sty.reset,
                        sty.gray,
                        sum,
                        sty.reset,
                    }));
                    if (tb.finished) {
                        const wrapped = try tui.wrapText(fa, tb.preview, if (cw > 4) cw - 4 else cw);
                        var shown: usize = 0;
                        for (wrapped.items) |ln| {
                            if (shown >= 4) break;
                            if (ln.len == 0) continue;
                            try out.append(try std.fmt.allocPrint(fa, "   {s}{s}{s}", .{ sty.gray, ln, sty.reset }));
                            shown += 1;
                        }
                        if (wrapped.items.len > shown and shown > 0) {
                            try out.append(try std.fmt.allocPrint(fa, "   {s}…{s}", .{ sty.gray, sty.reset }));
                        }
                        try out.append(try std.fmt.allocPrint(fa, "   {s}{s}{s}", .{
                            if (tb.ok) sty.green else sty.red,
                            if (tb.ok) "✓" else "✗",
                            sty.reset,
                        }));
                    }
                },
                .sys => |s| {
                    var it = std.mem.splitScalar(u8, s, '\n');
                    while (it.next()) |ln| {
                        const wrapped = try tui.wrapLine(fa, ln, if (cw > 2) cw - 2 else cw);
                        for (wrapped.items) |wl| {
                            try out.append(try std.fmt.allocPrint(fa, " {s}{s}{s}", .{ sty.gray, wl, sty.reset }));
                        }
                    }
                },
                .rsi => |rb| {
                    const prop = rb.prop;
                    const title: []const u8 = if (prop.gap.len > 0)
                        "◈ rsi proposal — self-check failed"
                    else
                        "◈ rsi proposal";
                    try out.append(try std.fmt.allocPrint(fa, " {s}╭─ {s}{s} {s}", .{
                        sty.magenta, title, try rep(fa, "─", if (cw > tui.displayWidth(title) + 8)
                            cw - tui.displayWidth(title) - 8 else 1), sty.reset,
                    }));
                    if (prop.gap.len > 0) {
                        const g = try clip(fa, prop.gap, if (cw > 12) cw - 12 else cw);
                        try out.append(try std.fmt.allocPrint(fa, " {s}│{s} {s}gap:{s} {s}", .{
                            sty.magenta, sty.reset, sty.yellow, sty.reset, g,
                        }));
                    }
                    const focus = try clip(fa, prop.focus, if (cw > 12) cw - 12 else cw);
                    try out.append(try std.fmt.allocPrint(fa, " {s}│{s} {s}focus:{s} {s}", .{
                        sty.magenta, sty.reset, sty.cyan, sty.reset, focus,
                    }));
                    try out.append(try std.fmt.allocPrint(fa, " {s}│{s} {s}{s}{s}", .{
                        sty.magenta, sty.reset, sty.gray, prop.stats_line, sty.reset,
                    }));
                    for (prop.steps) |step| {
                        try out.append(try std.fmt.allocPrint(fa, " {s}│{s}   {s}· {s}{s}", .{
                            sty.magenta, sty.reset, sty.gray, step, sty.reset,
                        }));
                    }
                    switch (rb.decided) {
                        .pending => {
                            const a_lbl = "[a] accept";
                            const r_lbl = "[r] reject";
                            const row_i = out.items.len;
                            try out.append(try std.fmt.allocPrint(fa, " {s}│{s}   {s}{s}{s}   {s}{s}{s}", .{
                                sty.magenta, sty.reset,
                                sty.green ++ sty.bold, a_lbl, sty.reset,
                                sty.red ++ sty.bold, r_lbl, sty.reset,
                            }));
                            // zones are 0-based display cols; the line starts
                            // with " │   " → 5 cols before the first button
                            try self.chat_zones.append(.{
                                .line = row_i, .col0 = 5, .col1 = 5 + a_lbl.len - 1,
                                .act = .{ .accept_rsi = prop },
                            });
                            const r0 = 5 + a_lbl.len + 3;
                            try self.chat_zones.append(.{
                                .line = row_i, .col0 = r0, .col1 = r0 + r_lbl.len - 1,
                                .act = .{ .reject_rsi = prop },
                            });
                        },
                        .accepted => try out.append(try std.fmt.allocPrint(fa,
                            " {s}│{s}   {s}✓ accepted — loop queued{s}", .{ sty.magenta, sty.reset, sty.green, sty.reset })),
                        .rejected => try out.append(try std.fmt.allocPrint(fa,
                            " {s}│{s}   {s}✗ rejected{s}", .{ sty.magenta, sty.reset, sty.red, sty.reset })),
                    }
                    try out.append(try std.fmt.allocPrint(fa, " {s}╰{s}{s}", .{
                        sty.magenta, try rep(fa, "─", if (cw > 6) cw - 6 else 1), sty.reset,
                    }));
                },
            }
        }
        return out;
    }

    const ComposerRender = struct {
        lines: std.ArrayList([]const u8),
        cur_row: usize,
        cur_col: usize,
    };

    fn renderComposer(self: *App, fa: std.mem.Allocator, w: usize) !std.ArrayList([]const u8) {
        var out = std.ArrayList([]const u8).init(fa);
        const text_w = if (w > 6) w - 6 else w; // "│ ❯ " + text + "│"
        try out.append(try std.fmt.allocPrint(fa, "{s}╭{s}╮{s}", .{ sty.gray, try rep(fa, "─", if (w > 2) w - 2 else 1), sty.reset }));

        const body = self.composer.items;
        if (body.len == 0) {
            const inner = try std.fmt.allocPrint(fa, " {s}❯{s} {s}ask xera anything — / for commands{s}", .{
                sty.accent, sty.reset, sty.gray, sty.reset,
            });
            const iw = tui.displayWidth(inner);
            const pad = if (w > iw + 2) w - iw - 2 else 0;
            try out.append(try std.fmt.allocPrint(fa, "{s}│{s}{s}{s}│{s}", .{
                sty.gray, sty.reset, inner, try spaces(fa, pad), sty.gray,
            }));
        } else {
            const wrapped = try tui.wrapText(fa, body, text_w);
            for (wrapped.items, 0..) |ln, i| {
                const prefix: []const u8 = if (i == 0) sty.accent ++ "❯ " ++ sty.reset else "  ";
                const content = try std.fmt.allocPrint(fa, " {s}{s}", .{ prefix, ln });
                const iw = tui.displayWidth(content);
                const pad = if (w > iw + 2) w - iw - 2 else 0;
                try out.append(try std.fmt.allocPrint(fa, "{s}│{s}{s}{s}│{s}", .{
                    sty.gray, sty.reset, content, try spaces(fa, pad), sty.gray,
                }));
            }
        }
        try out.append(try std.fmt.allocPrint(fa, "{s}╰{s}╯{s}", .{ sty.gray, try rep(fa, "─", if (w > 2) w - 2 else 1), sty.reset }));
        return out;
    }

    const Pos = struct { row: usize, col: usize };

    /// Compute the terminal cursor position (0-based row, col) inside the composer.
    fn cursorPos(self: *App, fa: std.mem.Allocator, comp_h: usize, busy_row: usize) !Pos {
        const w: usize = self.term.w;
        const h: usize = self.term.h;
        const tw = if (w > 6) w - 6 else w;
        const chat_h = if (h > 1 + comp_h + busy_row + 1) h - 1 - comp_h - busy_row - 1 else 1;
        const first_content_row = 1 + chat_h + busy_row + 1; // status + chat + busy + top border

        const cur = @min(self.cursor, self.composer.items.len);
        const before = self.composer.items[0..cur];
        const line_start = if (std.mem.lastIndexOfScalar(u8, before, '\n')) |idx| idx + 1 else 0;
        const head = before[0..line_start]; // complete raw lines incl trailing \n
        var head_lines: usize = 0;
        if (head.len > 0) {
            const wrapped_head = try tui.wrapText(fa, head[0 .. head.len - 1], tw);
            head_lines = wrapped_head.items.len;
        }
        const tail_disp = tui.displayWidth(before[line_start..]);
        const row_in = head_lines + tail_disp / @max(1, tw);
        const col_in = 4 + tail_disp % @max(1, tw);
        return .{ .row = first_content_row + row_in, .col = col_in };
    }

    fn paint(self: *App, lines: []const []const u8, h: usize, cur: Pos) !void {
        var bw = std.io.bufferedWriter(self.term.out.writer());
        const out = bw.writer();
        try out.writeAll("\x1b[?25l");
        var i: usize = 0;
        while (i < h) : (i += 1) {
            const want = if (i < lines.len) lines[i] else "";
            const prev_has = i < self.prev.items.len;
            const prev_line = if (prev_has) self.prev.items[i] else "";
            if (prev_has and std.mem.eql(u8, prev_line, want)) continue;
            try out.print("\x1b[{d};1H\x1b[2K{s}", .{ i + 1, want });
        }
        // place the text cursor inside the composer
        try out.print("\x1b[{d};{d}H\x1b[?25h", .{ cur.row + 1, cur.col + 1 });
        // store for next diff
        _ = self.prev_arena.reset(.retain_capacity);
        self.prev.clearRetainingCapacity();
        const pa = self.prev_arena.allocator();
        for (lines) |l| try self.prev.append(try pa.dupe(u8, l));
        try bw.flush();
    }
};

test "model picker filter is case-insensitive" {
    try std.testing.expect(App.modelMatches("OpenAI/Model-1", "model"));
    try std.testing.expect(!App.modelMatches("OpenAI/Model-1", "missing"));
    try std.testing.expect(App.modelMatches("OpenAI/Model-1", ""));
}

// ---------- worker ----------

fn workerMain(app: *App) void {
    workerLoop(app) catch |e| {
        const msg = std.fmt.allocPrint(app.alloc, "worker failed: {s}", .{@errorName(e)}) catch "worker failed";
        app.q.push(.{ .failed = msg });
    };
}

/// The single agent worker: owns the agent, its ui callback and approver.
const Worker = struct {
    ag: agent.Agent,
    ui: tools.Ui,
    approver: tools.Approver,
};

fn makeWorker(app: *App, alloc: std.mem.Allocator) !*Worker {
    const w = try alloc.create(Worker);
    w.* = .{
        .ag = try agent.Agent.init(alloc, app.cfg, app.hclient, app.h, app.cwd, app.cwd, 0, ""),
        .ui = undefined,
        .approver = undefined,
    };
    w.ui = .{
        .ctx = @ptrCast(app),
        .on_text = App.pushText,
        .on_reasoning = App.pushThink,
        .on_tool_call = App.pushToolStart,
        .on_tool_result = App.pushToolResult,
        .on_status = App.pushStatus,
    };
    w.approver = .{ .ctx = @ptrCast(app), .ask = App.askApproval };
    w.ag.ui = &w.ui;
    w.ag.approver = &w.approver;
    w.ag.slog = app.slog;
    w.ag.livelog = app.livelog;
    w.ag.coord = &app.coord;
    w.ag.aid = "main";
    w.ag.memory = app.mem;
    w.ag.mem_auto = app.cfg.memory_auto;
    w.ag.mem_doc_id = try std.fmt.allocPrint(alloc, "xera-{s}-main", .{app.session_id});
    w.ag.traces = app.traces_store;
    w.ag.sid = app.session_id;
    return w;
}

fn workerLoop(app: *App) !void {
    const alloc = app.alloc;
    const main = try makeWorker(app, alloc);
    app.ag_ptr.store(&main.ag, .release);

    while (true) {
        const job = app.q.popJob();
        switch (job) {
            .quit => return,
            .reset => {
                main.ag = try agent.Agent.init(alloc, app.cfg, app.hclient, app.h, app.cwd, app.cwd, 0, "");
                main.ag.ui = &main.ui;
                main.ag.approver = &main.approver;
                main.ag.slog = app.slog;
                main.ag.livelog = app.livelog;
                main.ag.coord = &app.coord;
                main.ag.aid = "main";
                main.ag.memory = app.mem;
                main.ag.mem_auto = app.cfg.memory_auto;
                main.ag.mem_doc_id = try std.fmt.allocPrint(alloc, "xera-{s}-main", .{app.session_id});
                main.ag.traces = app.traces_store;
                main.ag.sid = app.session_id;
                app.ag_ptr.store(&main.ag, .release);
                app.q.push(.cleared);
            },
            .load_msgs => |msgs| {
                var new_msgs = std.ArrayList(std.json.Value).init(alloc);
                try new_msgs.append(main.ag.messages.items[0]); // keep system
                for (msgs.items) |m| {
                    const role = util.jstr(m, "role") orelse continue;
                    if (std.mem.eql(u8, role, "system")) continue;
                    try new_msgs.append(m);
                }
                main.ag.messages = new_msgs;
                app.q.push(.{ .status = "session loaded into context" });
            },
            .usage => {
                var sz: usize = 0;
                for (main.ag.messages.items) |m| {
                    sz += (util.jstringify(alloc, m) catch "").len;
                }
                app.q.push(.{ .status = try std.fmt.allocPrint(alloc,
                    "{d} msgs · context ~{d}k chars · tokens in {d} · out {d}",
                    .{ main.ag.messages.items.len, sz / 1000, main.ag.usage.prompt, main.ag.usage.completion }) });
            },
            .summarize => {
                const before = main.ag.messages.items.len;
                main.ag.compactNow() catch {};
                app.q.push(.{ .status = try std.fmt.allocPrint(alloc,
                    "compacted context: {d} → {d} msgs", .{ before, main.ag.messages.items.len }) });
            },
            .list_models => {
                const catalog = llm.listModels(app.hclient, alloc, app.cfg) catch |e| {
                    app.q.push(.{ .models = .{
                        .models = &.{},
                        .error_msg = try std.fmt.allocPrint(alloc, "could not load provider models: {s}", .{@errorName(e)}),
                    } });
                    continue;
                };
                app.q.push(.{ .models = catalog });
            },
            .set_model => |model| {
                app.cfg.model = model;
                app.q.push(.{ .model_changed = model });
            },
            .prompt => |p| {
                main.ag.interrupted.store(false, .release);
                app.ag_ptr.store(&main.ag, .release);
                const outcome = main.ag.runTask(p) catch |e| {
                    const msg = try std.fmt.allocPrint(alloc, "{s}", .{@errorName(e)});
                    app.q.push(.{ .failed = msg });
                    continue;
                };
                app.q.push(.{ .done = outcome });
                // self-check: a verified failure becomes an rsi proposal.
                // Cheap second pass — skipped when interrupted/empty.
                if (app.cfg.rsi_verify and outcome.text.len > 0 and
                    !main.ag.interrupted.load(.acquire))
                {
                    const v = rsi.selfCheck(alloc, app.cfg, app.hclient, p, outcome.text) catch |e| {
                        app.q.push(.{ .status = try std.fmt.allocPrint(alloc,
                            "self-check unavailable ({s})", .{@errorName(e)}) });
                        continue;
                    };
                    if (main.ag.traces) |ts|
                        ts.verdict(alloc, main.ag.sid, main.ag.turn_no, v.ok, v.gap);
                    if (!v.ok) {
                        const prop = rsi.buildProposal(alloc, p, v.gap) catch continue;
                        app.q.push(.{ .rsi_proposal = prop });
                    }
                }
            },
            .rsi_run => |prop| {
                const sink = rsi.Sink{
                    .ctx = @ptrCast(app),
                    .status = App.pushStatus,
                    .report = App.pushText,
                    .train = App.pushTraining,
                };
                rsi.runLoop(alloc, app.cfg, app.hclient, app.h, app.cwd, prop, &sink);
                if (main.ag.traces) |ts|
                    ts.rsiEvent(alloc, app.session_id, "loop", prop.focus);
                app.q.push(.rsi_done);
            },
        }
    }
}

// ---------- text helpers ----------

fn eqAny(s: []const u8, list: []const []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, s, x)) return true;
    return false;
}

fn spaces(fa: std.mem.Allocator, n: usize) ![]const u8 {
    const b = try fa.alloc(u8, n);
    @memset(b, ' ');
    return b;
}

fn rep(fa: std.mem.Allocator, s: []const u8, n: usize) ![]const u8 {
    var b = std.ArrayList(u8).init(fa);
    var i: usize = 0;
    while (i < n) : (i += 1) try b.appendSlice(s);
    return b.items;
}

fn spacesPad(fa: std.mem.Allocator, s: []const u8, n: usize) ![]const u8 {
    _ = s;
    return spaces(fa, n);
}

fn clip(fa: std.mem.Allocator, s: []const u8, max: usize) ![]const u8 {
    var out = std.ArrayList(u8).init(fa);
    var w: usize = 0;
    var i: usize = 0;
    while (i < s.len and w < max) {
        const c = s[i];
        const len = if (c < 0x80) 1 else (std.unicode.utf8ByteSequenceLength(c) catch 1);
        try out.appendSlice(s[i .. i + @min(len, s.len - i)]);
        w += 1;
        i += len;
    }
    if (i < s.len) try out.appendSlice("…");
    return out.items;
}

/// markdown-lite → styled logical lines (still to be wrapped)
fn mdLines(fa: std.mem.Allocator, text: []const u8) !std.ArrayList([]const u8) {
    var out = std.ArrayList([]const u8).init(fa);
    var in_code = false;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |ln0| {
        const ln = std.mem.trimRight(u8, ln0, "\r");
        if (std.mem.startsWith(u8, ln, "```")) {
            in_code = !in_code;
            continue;
        }
        if (in_code) {
            try out.append(try std.fmt.allocPrint(fa, "{s}  {s}{s}", .{ sty.gray, ln, sty.reset }));
            continue;
        }
        if (std.mem.startsWith(u8, ln, "### ")) {
            try out.append(try std.fmt.allocPrint(fa, "{s}{s}{s}", .{ sty.bold, try inlineMd(fa, ln[4..]), sty.reset }));
        } else if (std.mem.startsWith(u8, ln, "## ")) {
            try out.append(try std.fmt.allocPrint(fa, "{s}{s}{s}", .{ sty.bold, try inlineMd(fa, ln[3..]), sty.reset }));
        } else if (std.mem.startsWith(u8, ln, "# ")) {
            try out.append(try std.fmt.allocPrint(fa, "{s}{s}{s}", .{ sty.bold ++ sty.bright_cyan, try inlineMd(fa, ln[2..]), sty.reset }));
        } else if (std.mem.startsWith(u8, ln, "> ")) {
            try out.append(try std.fmt.allocPrint(fa, "{s}│{s} {s}{s}{s}", .{ sty.gray, sty.reset, sty.dim, try inlineMd(fa, ln[2..]), sty.reset }));
        } else if (std.mem.eql(u8, std.mem.trim(u8, ln, " "), "---") or std.mem.eql(u8, std.mem.trim(u8, ln, " "), "***")) {
            try out.append(try std.fmt.allocPrint(fa, "{s}──────────{s}", .{ sty.gray, sty.reset }));
        } else {
            try out.append(try inlineMd(fa, ln));
        }
    }
    return out;
}

fn inlineMd(fa: std.mem.Allocator, s: []const u8) ![]const u8 {
    var out = std.ArrayList(u8).init(fa);
    var code = false;
    var bold = false;
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == '`') {
            code = !code;
            if (code) {
                try out.appendSlice(sty.bright_cyan);
            } else {
                try out.appendSlice(if (bold) sty.reset ++ sty.bold else sty.reset);
            }
            i += 1;
            continue;
        }
        if (i + 1 < s.len and s[i] == '*' and s[i + 1] == '*') {
            bold = !bold;
            try out.appendSlice(if (bold) sty.bold else (if (code) sty.reset ++ sty.bright_cyan else sty.reset));
            i += 2;
            continue;
        }
        try out.append(s[i]);
        i += 1;
    }
    if (code or bold) try out.appendSlice(sty.reset);
    return out.items;
}
