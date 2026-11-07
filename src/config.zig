const std = @import("std");
const util = @import("util.zig");

pub const Approval = enum {
    /// Ask before writes/shell in interactive mode.
    ask,
    /// Auto-approve everything (except writes outside workspace still prompt in interactive).
    full_auto,

    pub fn parse(s: []const u8) ?Approval {
        if (std.ascii.eqlIgnoreCase(s, "ask") or std.ascii.eqlIgnoreCase(s, "suggest")) return .ask;
        if (std.ascii.eqlIgnoreCase(s, "full-auto") or std.ascii.eqlIgnoreCase(s, "auto") or
            std.ascii.eqlIgnoreCase(s, "yolo")) return .full_auto;
        return null;
    }
};

pub const Config = struct {
    /// e.g. https://api.openai.com/v1  (no trailing slash)
    base_url: []const u8 = "https://api.openai.com/v1",
    api_key: ?[]const u8 = null,
    /// env var to read the key from, when api_key isn't set directly
    api_key_env: []const u8 = "OPENAI_API_KEY",
    model: []const u8 = "gpt-5",
    /// max output tokens per completion
    max_tokens: u32 = 16384,
    temperature: ?f64 = null,
    approval: Approval = .ask,
    /// stream token deltas to the terminal
    stream: bool = true,
    /// shell used for run_shell on posix; windows always uses cmd /c unless overridden
    shell: ?[]const u8 = null,
    /// global harness directory (default ~/.xera-rsi/harness)
    harness_dir: ?[]const u8 = null,
    /// session transcripts directory (default ~/.xera-rsi/sessions)
    sessions_dir: ?[]const u8 = null,
    /// max agent iterations per turn before forced stop
    max_iters: u32 = 60,
    /// rough char budget for history before compaction kicks in
    history_budget: usize = 200_000,
    /// supermemory api key (SUPERMEMORY_API_KEY env or memory_key in config)
    memory_key: ?[]const u8 = null,
    /// supermemory base url — point at a self-hosted supermemory-local to
    /// keep memory off the cloud entirely
    memory_base: []const u8 = "https://api.supermemory.ai",
    /// namespace override; empty = "xera-<workspace-name>"
    memory_namespace: []const u8 = "",
    /// auto-recall prompts + auto-capture turns into memory
    memory_auto: bool = true,
    /// post-turn verification: a failed answer becomes an rsi proposal
    rsi_verify: bool = true,
};

const FileSchema = struct {
    base_url: ?[]const u8 = null,
    api_key: ?[]const u8 = null,
    api_key_env: ?[]const u8 = null,
    model: ?[]const u8 = null,
    max_tokens: ?u32 = null,
    temperature: ?f64 = null,
    approval: ?[]const u8 = null,
    stream: ?bool = null,
    shell: ?[]const u8 = null,
    max_iters: ?u32 = null,
    memory_key: ?[]const u8 = null,
    memory_base: ?[]const u8 = null,
    memory_namespace: ?[]const u8 = null,
    memory_auto: ?bool = null,
    rsi_verify: ?bool = null,
};

pub const Cli = struct {
    model: ?[]const u8 = null,
    base_url: ?[]const u8 = null,
    api_key: ?[]const u8 = null,
    approval: ?Approval = null,
    stream: ?bool = null,
    cwd: ?[]const u8 = null,
};

pub fn configPath(alloc: std.mem.Allocator) ![]u8 {
    const home = try util.homeDir(alloc);
    return std.fs.path.join(alloc, &.{ home, ".xera-rsi", "config.json" });
}

/// Load config: defaults -> ~/.xera-rsi/config.json -> env (XERA_*) -> cli flags.
pub fn load(alloc: std.mem.Allocator, cli: Cli) !Config {
    var cfg = Config{};

    const path = try configPath(alloc);
    if (std.fs.cwd().readFileAlloc(alloc, path, 1 << 20)) |text| {
        const parsed = std.json.parseFromSliceLeaky(FileSchema, alloc, text, .{
            .ignore_unknown_fields = true,
        }) catch |e| blk: {
            std.debug.print("warning: could not parse {s}: {s}\n", .{ path, @errorName(e) });
            break :blk FileSchema{};
        };
        if (parsed.base_url) |v| cfg.base_url = trimSlash(v);
        if (parsed.api_key) |v| cfg.api_key = v;
        if (parsed.api_key_env) |v| cfg.api_key_env = v;
        if (parsed.model) |v| cfg.model = v;
        if (parsed.max_tokens) |v| cfg.max_tokens = v;
        if (parsed.temperature) |v| cfg.temperature = v;
        if (parsed.stream) |v| cfg.stream = v;
        if (parsed.shell) |v| cfg.shell = v;
        if (parsed.max_iters) |v| cfg.max_iters = v;
        if (parsed.approval) |v| {
            if (Approval.parse(v)) |a| cfg.approval = a;
        }
        if (parsed.memory_key) |v| cfg.memory_key = v;
        if (parsed.memory_base) |v| cfg.memory_base = trimSlash(v);
        if (parsed.memory_namespace) |v| cfg.memory_namespace = v;
        if (parsed.memory_auto) |v| cfg.memory_auto = v;
        if (parsed.rsi_verify) |v| cfg.rsi_verify = v;
    } else |_| {}

    // env overrides
    if (util.getenv(alloc, "XERA_BASE_URL")) |v| cfg.base_url = trimSlash(v);
    if (util.getenv(alloc, "XERA_API_KEY")) |v| cfg.api_key = v;
    if (util.getenv(alloc, "XERA_API_KEY_ENV")) |v| cfg.api_key_env = v;
    if (util.getenv(alloc, "XERA_MODEL")) |v| cfg.model = v;
    if (util.getenv(alloc, "XERA_SHELL")) |v| cfg.shell = v;
    if (util.getenvBool(alloc, "XERA_NO_STREAM")) |b| {
        if (b) cfg.stream = false;
    }
    if (util.getenvBool(alloc, "XERA_FULL_AUTO")) |b| {
        if (b) cfg.approval = .full_auto;
    }
    if (util.getenv(alloc, "SUPERMEMORY_API_KEY")) |v| cfg.memory_key = v;
    if (util.getenv(alloc, "XERA_MEMORY_BASE")) |v| cfg.memory_base = trimSlash(v);
    if (util.getenv(alloc, "XERA_MEMORY_NS")) |v| cfg.memory_namespace = v;
    if (util.getenvBool(alloc, "XERA_NO_MEMORY")) |b| {
        if (b) cfg.memory_auto = false;
    }
    if (util.getenvBool(alloc, "XERA_NO_VERIFY")) |b| {
        if (b) cfg.rsi_verify = false;
    }

    // api key via api_key_env when not set directly
    if (cfg.api_key == null) {
        if (util.getenv(alloc, cfg.api_key_env)) |k| cfg.api_key = k;
    }

    // cli overrides
    if (cli.base_url) |v| cfg.base_url = trimSlash(v);
    if (cli.api_key) |v| cfg.api_key = v;
    if (cli.model) |v| cfg.model = v;
    if (cli.approval) |v| cfg.approval = v;
    if (cli.stream) |v| cfg.stream = v;

    return cfg;
}

fn trimSlash(s: []const u8) []const u8 {
    var out = s;
    while (out.len > 0 and out[out.len - 1] == '/') out = out[0 .. out.len - 1];
    return out;
}

pub fn maskedKey(cfg: *const Config) []const u8 {
    const k = cfg.api_key orelse return "(none)";
    if (k.len <= 8) return "********";
    return k[k.len - 4 ..];
}
