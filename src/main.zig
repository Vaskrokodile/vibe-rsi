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
const improve = @import("improve.zig");
const repl = @import("repl.zig");
const tui_app = @import("tui_app.zig");
const memory_mod = @import("memory.zig");
const traces_mod = @import("traces.zig");
const refine_mod = @import("refine.zig");

const version = "0.1.0";

const usage_text =
    \\xera-rsi — an agentic coding harness in Zig, with the Harness²
    \\recursive self-improvement loop built in.
    \\
    \\usage:
    \\  xera-rsi [flags]                 interactive session
    \\  xera-rsi exec [flags] <prompt>   one-shot run, then exit
    \\  xera-rsi improve [flags] <task>  Harness² propose-probe-compose on a task
    \\  xera-rsi refine <instructions>   refine the harness from the latest session
    \\  xera-rsi harness [show|path]     inspect the editable harness
    \\  xera-rsi sessions                list saved sessions
    \\  xera-rsi resume <id>             resume a session interactively
    \\  xera-rsi doctor                  config + connectivity check
    \\  xera-rsi version
    \\
    \\flags:
    \\  -m, --model <name>       model id (default from config/env)
    \\      --base-url <url>     OpenAI-compatible base url
    \\      --api-key <key>      api key (or set XERA_API_KEY / OPENAI_API_KEY)
    \\  -C, --cwd <dir>          working directory
    \\  -y, --full-auto          never ask before writes/shell
    \\      --no-stream          disable token streaming
    \\      --classic            use the plain line-mode UI instead of the TUI
    \\  -k <n>                   improve: candidates per round (default 2)
    \\      --mode <m>           improve: parallel|sequential (default parallel)
    \\      --dry-run            improve: propose only, no probes/apply
    \\
    \\config: ~/.xera-rsi/config.json
    \\env:    XERA_MODEL, XERA_BASE_URL, XERA_API_KEY, XERA_API_KEY_ENV,
    \\        XERA_SHELL, XERA_FULL_AUTO, XERA_NO_STREAM
    \\
    \\harness:  <cwd>/.xera-rsi/harness (or ~/.xera-rsi/harness)
    \\          system_prompt.md guardrails.md tool_guidance.md memory.md
    \\          lessons.md subagents.md skills/ scripts/
    \\
;

const Parsed = struct {
    cmd: []const u8 = "chat",
    positional: std.ArrayList([]const u8),
    cli: config.Cli = .{},
    k: u8 = 2,
    mode: improve.Mode = .parallel,
    dry_run: bool = false,
    classic: bool = false,
    help: bool = false,
};

fn parseArgs(alloc: std.mem.Allocator, argv: []const []const u8) !Parsed {
    var p = Parsed{ .positional = std.ArrayList([]const u8).init(alloc) };
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            p.help = true;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            p.cli.model = try need(argv, i, arg);
        } else if (std.mem.eql(u8, arg, "--base-url")) {
            i += 1;
            p.cli.base_url = try need(argv, i, arg);
        } else if (std.mem.eql(u8, arg, "--api-key")) {
            i += 1;
            p.cli.api_key = try need(argv, i, arg);
        } else if (std.mem.eql(u8, arg, "-C") or std.mem.eql(u8, arg, "--cwd")) {
            i += 1;
            p.cli.cwd = try need(argv, i, arg);
        } else if (std.mem.eql(u8, arg, "-y") or std.mem.eql(u8, arg, "--full-auto")) {
            p.cli.approval = .full_auto;
        } else if (std.mem.eql(u8, arg, "--ask")) {
            p.cli.approval = .ask;
        } else if (std.mem.eql(u8, arg, "--no-stream")) {
            p.cli.stream = false;
        } else if (std.mem.eql(u8, arg, "--stream")) {
            p.cli.stream = true;
        } else if (std.mem.eql(u8, arg, "-k")) {
            i += 1;
            const v = try need(argv, i, arg);
            p.k = std.fmt.parseInt(u8, v, 10) catch 2;
            if (p.k == 0) p.k = 1;
        } else if (std.mem.eql(u8, arg, "--mode")) {
            i += 1;
            const v = try need(argv, i, arg);
            if (std.mem.eql(u8, v, "sequential") or std.mem.eql(u8, v, "seq")) {
                p.mode = .sequential;
            } else {
                p.mode = .parallel;
            }
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            p.dry_run = true;
        } else if (std.mem.eql(u8, arg, "--classic") or std.mem.eql(u8, arg, "--no-tui")) {
            p.classic = true;
        } else if (arg.len > 1 and arg[0] == '-' and p.positional.items.len == 0) {
            std.debug.print("unknown flag: {s}\n", .{arg});
            return error.UnknownFlag;
        } else {
            try p.positional.append(arg);
        }
    }
    return p;
}

fn need(argv: []const []const u8, i: usize, flag: []const u8) ![]const u8 {
    if (i >= argv.len) {
        std.debug.print("flag {s} needs a value\n", .{flag});
        return error.MissingValue;
    }
    return argv[i];
}

pub fn main() !u8 {
    util.setupConsole();
    var arena_inst = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_inst.deinit();
    // Sockets, agent workers and the UI all allocate concurrently — the
    // arena itself isn't thread-safe, so serialize access behind a mutex.
    var ts_alloc: std.heap.ThreadSafeAllocator = .{ .child_allocator = arena_inst.allocator() };
    const alloc = ts_alloc.allocator();

    const argv = try std.process.argsAlloc(alloc);
    const args = if (argv.len > 1) argv[1..] else &[_][]const u8{};

    var parsed = parseArgs(alloc, args) catch |e| {
        std.debug.print("could not parse arguments ({s})\n{s}", .{ @errorName(e), usage_text });
        return 2;
    };
    if (parsed.help) {
        std.debug.print("{s}", .{usage_text});
        return 0;
    }

    // first positional is the subcommand when it matches a known one
    const subcommands = [_][]const u8{ "chat", "exec", "improve", "refine", "harness", "sessions", "resume", "doctor", "config", "version", "help" };
    if (parsed.positional.items.len > 0) {
        const first = parsed.positional.items[0];
        for (subcommands) |s| {
            if (std.mem.eql(u8, first, s)) {
                parsed.cmd = s;
                _ = parsed.positional.orderedRemove(0);
                break;
            }
        }
    }
    if (std.mem.eql(u8, parsed.cmd, "help")) {
        std.debug.print("{s}", .{usage_text});
        return 0;
    }
    if (std.mem.eql(u8, parsed.cmd, "version")) {
        std.debug.print("xera-rsi {s}\n", .{version});
        return 0;
    }

    repl.initTerm();

    var cfg = try config.load(alloc, parsed.cli);

    const cwd = if (parsed.cli.cwd) |c|
        try util.resolvePath(alloc, ".", c)
    else
        try std.fs.cwd().realpathAlloc(alloc, ".");

    var h = try harness.Harness.init(alloc, &cfg, cwd);
    try h.ensure();

    var hclient = http.HttpClient.init(alloc);
    defer hclient.deinit();

    const cmd = parsed.cmd;
    const rest = parsed.positional.items;

    if (std.mem.eql(u8, cmd, "doctor")) {
        return cmdDoctor(alloc, &cfg, &hclient);
    }
    if (std.mem.eql(u8, cmd, "config")) {
        return cmdConfig(&cfg, &h);
    }
    if (std.mem.eql(u8, cmd, "harness")) {
        return cmdHarness(alloc, &h, rest);
    }
    if (std.mem.eql(u8, cmd, "sessions")) {
        return cmdSessions(alloc, &cfg);
    }
    if (std.mem.eql(u8, cmd, "resume")) {
        if (rest.len == 0) {
            std.debug.print("usage: xera-rsi resume <session-id>\n", .{});
            return 2;
        }
        const msgs = session.loadMessages(alloc, &cfg, rest[0]) catch |e| {
            std.debug.print("cannot load session {s}: {s}\n", .{ rest[0], @errorName(e) });
            return 1;
        };
        if (!parsed.classic and util.isTty(std.io.getStdIn())) {
            try tui_app.App.run(alloc, &cfg, &hclient, &h, cwd, msgs);
        } else {
            try repl.replLoop(alloc, &cfg, &hclient, &h, cwd, msgs);
        }
        return 0;
    }
    if (std.mem.eql(u8, cmd, "refine")) {
        return cmdRefine(alloc, &cfg, &hclient, &h, cwd, rest);
    }
    if (std.mem.eql(u8, cmd, "exec")) {
        if (rest.len == 0) {
            std.debug.print("usage: xera-rsi exec <prompt>\n", .{});
            return 2;
        }
        return cmdExec(alloc, &cfg, &hclient, &h, cwd, rest);
    }
    if (std.mem.eql(u8, cmd, "improve")) {
        if (rest.len == 0) {
            std.debug.print("usage: xera-rsi improve <task>\n", .{});
            return 2;
        }
        const task = try std.mem.join(alloc, " ", rest);
        var q = repl.quietUi();
        improve.run(alloc, &cfg, &hclient, &h, cwd, task, .{
            .k = parsed.k,
            .mode = parsed.mode,
            .dry_run = parsed.dry_run,
        }, &q) catch |e| {
            std.debug.print("improve failed: {s}\n", .{@errorName(e)});
            return 1;
        };
        return 0;
    }
    if (std.mem.eql(u8, cmd, "chat")) {
        if (cfg.api_key == null or cfg.api_key.?.len == 0) {
            std.debug.print(
                "warning: no API key configured. Set XERA_API_KEY (or {s}) or\n" ++
                    "api_key in ~/.xera-rsi/config.json. Continuing anyway.\n\n",
                .{cfg.api_key_env},
            );
        }
        if (!parsed.classic and util.isTty(std.io.getStdIn())) {
            try tui_app.App.run(alloc, &cfg, &hclient, &h, cwd, null);
        } else {
            try repl.replLoop(alloc, &cfg, &hclient, &h, cwd, null);
        }
        return 0;
    }

    std.debug.print("unknown command {s}\n{s}", .{ cmd, usage_text });
    return 2;
}

fn cmdExec(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    h: *harness.Harness,
    cwd: []const u8,
    rest: []const []const u8,
) !u8 {
    const prompt = try std.mem.join(alloc, " ", rest);
    var log = try session.create(alloc, cfg, cwd);
    var ag = try agent.Agent.init(alloc, cfg, hclient, h, cwd, cwd, 0, "");
    var q = repl.quietUi();
    ag.ui = &q;
    ag.slog = &log;
    ag.sid = log.id;
    if (traces_mod.Store.create(alloc)) |ts| ag.traces = ts;
    if (live.Live.create(alloc, log.id) catch null) |lv| {
        ag.livelog = lv;
        lv.sessionStart(cwd, cfg.model, h.dir, "exec");
    }
    // long-term memory (supermemory) — same recall + capture as the TUI
    if (memory_mod.Memory.create(alloc, hclient, cfg.memory_key, cfg.memory_base,
        if (cfg.memory_namespace.len > 0) cfg.memory_namespace
        else memory_mod.Memory.defaultNs(alloc, cwd))) |mem|
    {
        ag.memory = mem;
        ag.mem_auto = cfg.memory_auto;
        ag.mem_doc_id = try std.fmt.allocPrint(alloc, "xera-{s}-exec", .{log.id});
    }
    // exec is non-interactive: approver null -> auto-approve
    const outcome = ag.runTask(prompt) catch |e| {
        std.debug.print("\nexec failed: {s}\n", .{@errorName(e)});
        return 1;
    };
    if (!cfg.stream and outcome.text.len > 0) {
        std.debug.print("{s}\n", .{outcome.text});
    } else {
        std.debug.print("\n", .{});
    }
    std.debug.print("(session {s})\n", .{log.id});
    return 0;
}

/// `xera-rsi refine [--global] <instructions>` — run the continual-
/// harness refinement pass against the latest session transcript.
/// Local scope (workspace harness) by default; --global targets
/// ~/.xera-rsi/harness.
fn cmdRefine(
    alloc: std.mem.Allocator,
    cfg: *const config.Config,
    hclient: *http.HttpClient,
    h: *harness.Harness,
    cwd: []const u8,
    rest: []const []const u8,
) !u8 {
    _ = cwd;
    var glob = false;
    var words = std.ArrayList([]const u8).init(alloc);
    for (rest) |w| {
        if (std.mem.eql(u8, w, "--global") or std.mem.eql(u8, w, "-g"))
            glob = true
        else
            try words.append(w);
    }
    const instructions = try std.mem.join(alloc, " ", words.items);

    // pick the newest session — it's the trajectory being refined
    const infos = try session.list(alloc, cfg);
    if (infos.len == 0) {
        std.debug.print("no sessions yet — run the agent first\n", .{});
        return 1;
    }
    const msgs = session.loadMessages(alloc, cfg, infos[0].id) catch |e| {
        std.debug.print("cannot load session {s}: {s}\n", .{ infos[0].id, @errorName(e) });
        return 1;
    };
    std.debug.print("refining harness from session {s} ({d} turns){s}…\n", .{
        infos[0].id, infos[0].turns, if (glob) " [global]" else "" });

    const summary = refine_mod.run(alloc, cfg, hclient, h, msgs.items, "", .{
        .instructions = instructions,
        .scope_global = glob,
    }) catch |e| {
        std.debug.print("refine failed: {s}\n", .{@errorName(e)});
        return 1;
    };
    std.debug.print("{s}\n", .{summary});
    return 0;
}

fn cmdDoctor(alloc: std.mem.Allocator, cfg: *const config.Config, hclient: *http.HttpClient) !u8 {
    std.debug.print("xera-rsi doctor\n\n", .{});
    std.debug.print("  base_url   : {s}\n", .{cfg.base_url});
    std.debug.print("  model      : {s}\n", .{cfg.model});
    std.debug.print("  api key    : …{s}\n", .{config.maskedKey(cfg)});
    std.debug.print("  approval   : {s}\n", .{@tagName(cfg.approval)});
    std.debug.print("  streaming  : {s}\n\n", .{if (cfg.stream) "on" else "off"});

    const url = try std.fmt.allocPrint(alloc, "{s}/models", .{cfg.base_url});
    var headers = std.ArrayList(http.Header).init(alloc);
    if (cfg.api_key) |k| {
        const v = try std.fmt.allocPrint(alloc, "Bearer {s}", .{k});
        try headers.append(.{ .name = "authorization", .value = v });
    }
    const res = hclient.fetch(alloc, .GET, url, headers.items, null) catch |e| {
        std.debug.print("  GET /models : FAILED ({s})\n", .{@errorName(e)});
        std.debug.print("  → check base_url, network, and proxy env vars\n", .{});
        return 1;
    };
    std.debug.print("  GET /models : {s}\n", .{@tagName(res.status)});
    if (res.status != .ok) {
        const body = if (res.body.len > 500) res.body[0..500] else res.body;
        std.debug.print("  {s}\n", .{body});
        return 1;
    }
    // count models
    const root = util.jparse(alloc, res.body) catch {
        std.debug.print("  (response not JSON — endpoint reachable)\n", .{});
        return 0;
    };
    if (util.jget(root, "data")) |d| {
        if (d == .array) {
            std.debug.print("  {d} model(s) listed", .{d.array.items.len});
            var found = false;
            for (d.array.items) |m| {
                if (util.jstr(m, "id")) |id| {
                    if (std.mem.eql(u8, id, cfg.model)) found = true;
                }
            }
            if (found) {
                std.debug.print("  ✓ configured model present\n", .{});
            } else {
                std.debug.print("  ⚠ configured model '{s}' not in list\n", .{cfg.model});
            }
        }
    }

    // memory layer (supermemory) — configured? reachable?
    std.debug.print("\n", .{});
    const ns = if (cfg.memory_namespace.len > 0) cfg.memory_namespace else "xera-<workspace>";
    if (memory_mod.Memory.create(alloc, hclient, cfg.memory_key, cfg.memory_base, ns)) |mem| {
        std.debug.print("  memory     : {s} ns={s} auto={s}\n", .{
            cfg.memory_base, mem.ns, if (cfg.memory_auto) "on" else "off" });
        const prof = mem.profile(alloc) catch |e| blk: {
            break :blk std.fmt.allocPrint(alloc, "error: {s}", .{@errorName(e)}) catch "error";
        };
        if (std.mem.startsWith(u8, prof, "error:")) {
            std.debug.print("  profile    : FAILED — {s}\n", .{prof});
        } else {
            std.debug.print("  profile    : ok\n", .{});
        }
    } else {
        std.debug.print("  memory     : not configured (set SUPERMEMORY_API_KEY or memory_key in config.json)\n", .{});
    }
    return 0;
}

fn cmdConfig(cfg: *const config.Config, h: *harness.Harness) u8 {
    std.debug.print("base_url   : {s}\n", .{cfg.base_url});
    std.debug.print("model      : {s}\n", .{cfg.model});
    std.debug.print("api key    : …{s}\n", .{config.maskedKey(cfg)});
    std.debug.print("approval   : {s}\n", .{@tagName(cfg.approval)});
    std.debug.print("stream     : {}\n", .{cfg.stream});
    std.debug.print("max_tokens : {d}\n", .{cfg.max_tokens});
    std.debug.print("harness    : {s}\n", .{h.dir});
    return 0;
}

fn cmdHarness(alloc: std.mem.Allocator, h: *harness.Harness, rest: []const []const u8) !u8 {
    const sub = if (rest.len > 0) rest[0] else "show";
    if (std.mem.eql(u8, sub, "path")) {
        std.debug.print("{s}\n", .{h.dir});
        return 0;
    }
    std.debug.print("harness dir: {s} ({s})\n\n", .{ h.dir, if (h.global) "global" else "project" });
    for (harness.component_files) |f| {
        const body = h.read(f);
        std.debug.print("  {s: <18} {d: >6} bytes\n", .{ f, body.len });
    }
    const skills = h.listSkills() catch &[_]harness.NamedItem{};
    std.debug.print("\nskills: {d}\n", .{skills.len});
    for (skills) |s| std.debug.print("  - {s}  {s}\n", .{ s.name, s.desc });
    const agents = h.listAgents() catch &[_]harness.NamedItem{};
    std.debug.print("sub_agents: {d}\n", .{agents.len});
    for (agents) |a| std.debug.print("  - {s}  {s}\n", .{ a.name, a.desc });
    const scripts = h.listScripts() catch &[_][]const u8{};
    std.debug.print("scripts: {d}\n", .{scripts.len});
    for (scripts) |s| std.debug.print("  - {s}\n", .{s});
    if (h.hooksPath() != null) std.debug.print("hooks: hooks.json present\n", .{});
    const problems = h.validate(alloc) catch &[_][]const u8{};
    for (problems) |p| std.debug.print("  ! {s}\n", .{p});
    std.debug.print("\nedit these files directly, or let `xera-rsi improve` do it.\n", .{});
    return 0;
}

fn cmdSessions(alloc: std.mem.Allocator, cfg: *const config.Config) !u8 {
    const infos = try session.list(alloc, cfg);
    if (infos.len == 0) {
        std.debug.print("(no sessions yet)\n", .{});
        return 0;
    }
    for (infos) |info| {
        std.debug.print("{s}  {d} turn(s)  {s}\n  {s}\n", .{ info.id, info.turns, info.model, info.cwd });
    }
    return 0;
}
