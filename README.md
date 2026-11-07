# xera-rsi

An agentic coding harness in **Zig** — modeled after Codex CLI / Claude Code —
with the **Harness²** recursive self-improvement loop built in.

Single static binary (~2 MB, no runtime deps). Speaks the OpenAI-compatible
chat-completions API, so it works with OpenAI, LiteLLM, Ollama, LM Studio,
DeepSeek, Groq, and any compatible endpoint.

```
xera-rsi                 interactive TUI (chat, tools, approvals)
xera-rsi exec "task"     one-shot run, then exit
xera-rsi improve "task"  Harness² propose-probe-compose on a task
xera-rsi harness         inspect the editable harness
xera-rsi sessions        list saved sessions
xera-rsi resume <id>     resume a session
xera-rsi doctor          config + connectivity check
```

## Interactive UI

Running with no subcommand opens a full-screen Codex-style TUI
(alternate screen; falls back to plain line mode when stdin isn't a TTY,
or with `--classic`):

- **banner** — extruded 3D `XERA-RSI` logo (cyan→blue→white truecolor)
  revealed by a matrix-scramble animation on startup
- **scrollback** — user turns, streamed assistant text (markdown-lite:
  **bold**, `code`, headers, quotes, code fences), tool cells
  (`⚙ tool args` → result preview → `✓`/`✗`), status notes
- **composer** — bordered input box, UTF-8 aware cursor movement,
  up/down history recall, paste
- **command picker** — type `/` and a live dropdown opens, narrowing as
  you type; `↑`/`↓` pick, `tab` completes, `enter` runs, `esc` dismisses
- **status bar** — model · workspace · session id · approval mode
- **spinner** — `⠙ working Ns · M tool call(s)` while the agent runs
- **approvals** — `y`/`n`/`a` on `⚠ allow?` rows, or `esc` to deny
- **keys** — `enter` send · `esc` interrupt (busy) / clear (idle) ·
  `pgup`/`pgdn` scroll · `↑`/`↓` history · `ctrl+l` redraw · `ctrl+c` quit

Slash commands: `/help` `/exit` `/clear` `/auto` `/model` `/usage`
(token counters) `/summarize` (compact now) `/skills` `/agents`
`/harness` (components + validation problems) `/library` `/sessions`
`/resume <id>` `/status` `/improve` `/rsi-loop [focus]`. `/model` opens a
searchable model picker for the currently configured provider; navigate with
arrows/Enter or hover and click a model. Use config (`base_url`, API key) to
change providers.

### Agent coordination

Every agent the app runs — the main agent plus `task` subagents — shares
one in-process **coordination fabric** so they don't trip over each other.
It's structured signals, not chat, so it costs almost no tokens:

- **claims** — an agent calls `coord{op:"claim",pat:"src/*.zig"}` before
  writing. The file tools *hard-enforce* claims: writes into paths held
  by another agent are refused (works even if a model never calls coord).
  Claims persist until `coord{op:"rel"}` or ~10min expiry.
- **feed** — successful writes/shell runs emit compact events
  (`seq|kind|from|data`, ~40 bytes) that are pushed to siblings as
  `<sync>` messages between model calls — awareness is free.
- **inbox** — `coord{op:"send",to:"aid"|"*",m:"..."}` for terse
  agent→agent negotiation; `coord{op:"status"}` shows the roster with
  notes and held claims.
- A subagent shares its parent's lineage (`main/s` inherits `main`), so
  delegated work is never blocked by the delegator's claims.

## Setup

```jsonc
// ~/.xera-rsi/config.json
{
  "base_url": "https://api.openai.com/v1",
  "api_key_env": "OPENAI_API_KEY",   // or "api_key": "sk-..."
  "model": "gpt-5",
  "max_tokens": 16384,
  "approval": "ask",                  // ask | full-auto
  "stream": true,

  // memory layer (supermemory) — all optional
  "memory_key": "sm_...",             // or SUPERMEMORY_API_KEY env
  "memory_base": "https://api.supermemory.ai", // or a self-hosted supermemory-local
  "memory_namespace": "",             // default: "xera-<workspace-name>"
  "memory_auto": true,                // auto-recall prompts + auto-capture turns

  // rsi loop — post-turn verification → improvement proposals
  "rsi_verify": true                  // failed turns offer an rsi loop
}
```

Environment overrides: `XERA_MODEL`, `XERA_BASE_URL`, `XERA_API_KEY`,
`XERA_API_KEY_ENV`, `XERA_SHELL`, `XERA_FULL_AUTO`, `XERA_NO_STREAM`,
`XERA_NO_VERIFY`, `SUPERMEMORY_API_KEY`, `XERA_MEMORY_BASE`, `XERA_MEMORY_NS`,
`XERA_NO_MEMORY`.

Common flags: `-m/--model`, `--base-url`, `--api-key`, `-C/--cwd`,
`-y/--full-auto`, `--no-stream`.

### ULT (GA106 megakernel engine)

`tools/ult-serve/serve.py` bridges the harness to the
[ULT](https://github.com/Vaskrokodile/ult) inference engine — handwritten
CUDA megakernels for consumer GPUs (RTX 3060-class), ~250 tok/s on
1.5–1.7B Q4 models:

```
vibe-ult.cmd                          # one-shot: server (all models) + TUI
python tools/ult-serve/serve.py --model all --port 8321   # both models,
                                        # lazy-loaded; /model switches live
```

then (if launching the harness manually):

```powershell
$env:XERA_BASE_URL="http://127.0.0.1:8321/v1"
$env:XERA_MODEL="ult-qwen-1.5b"
$env:XERA_API_KEY="ult"          # any value; the shim ignores auth
```

The adapter injects tool schemas into the prompt and parses
`<tool_call>{...}</tool_call>` back into OpenAI `tool_calls` (best-effort
on small models), maps R1/LFM3 `<think>` into `reasoning_content`, and
streams real SSE from the kernel's mapped token counter. Flags/env:
`GA106_HOME` (engine source tree, default `E:\GA106`),
`GA106_LFM3_VARIANT`, `--tools-limit` (cap injected schemas, default 8 —
small models can't juggle 20). The engine's `MAXPOS` window is 8192;
oversized requests clamp generation then truncate mid-prompt.

## Tools the agent can call

`read_file` `write_file` `edit_file` (exact-match replace) `list_dir`
`glob` `grep` `delete` (files or trees, workspace-confined) `make_dir`
`run_shell` (foreground, or `background:true` → handle + [shell-done])
`web_fetch` (URL → text, HTML stripped) `web_search` (DuckDuckGo lite)
`memory_append` `memory` (supermemory: remember/recall/profile/forget)
`coord` (inter-agent claims, feed, inbox) `skill_load`
`task` (subagent: `run` blocking, or `spawn`/`collect`/`list` async)
`goal` (persistent objective, re-surfaced until completed)
`refine` (schedule evidence-backed harness refinement at turn end)
`traces` (telemetry: `stats` `failures` `space` `dataset`)
`train` (training pipeline: `dataset` `check` `clean` `hf` `pull`
`presets` `launch` `status` — one call each, no code)
`harness` (direct harness edits: `list` `read` `append` `write` `delete`
`skill_add` `agent_add` `history` `snapshots` `rollback` — every write
auto-snapshots)

In interactive mode, write/shell tools ask for approval (`y`/`n`/`a`lways).
`exec` and `improve` run unattended (auto-approve). Writes are confined to the
workspace root. `esc` interrupts a running turn.

## Memory layer (supermemory)

With `SUPERMEMORY_API_KEY` set, every agent gets durable cross-session
memory via [supermemory](https://supermemory.ai) — their learner model
extracts, links, updates and forgets facts; we just ship context in and
pull it back out:

- **auto-recall** — each prompt runs a hybrid search; relevant hits are
  injected as a `<memory>` block before the turn (id + similarity shown,
  so `forget` is one call away). Fails silently — memory can never break
  a turn.
- **auto-capture** — each finished turn is appended to the agent's
  session document (`xera-<session>-<aid>`); supermemory's pipeline does
  the fact extraction. Subagents share the namespace but don't capture.
- **`memory` tool** — `remember` (instant, durable), `recall`, `profile`,
  `forget`/`forget_q` (dry-run first, `apply:true` confirms).
- **one namespace per workspace** (`xera-<dir>`), shared by the main agent
  and all subagents — they literally share a brain. `memory_append` still
  writes the local `harness/memory.md`; the two layers coexist.
- point `memory_base` at a [self-hosted](https://supermemory.ai/docs/self-hosting/overview)
  supermemory-local to keep memory fully offline. `doctor` shows the
  configured namespace and pings the profile endpoint.

## Long-running-work features

Ideas ported from prime-agent's RLM/continual-harness design, adapted to
this codebase's tool-loop architecture:

- **background shells** — `run_shell{background:true}` returns a `sh-N`
  handle immediately and keeps streaming output under a lock; the
  `[shell-done sh-N exit N]` notice lands in the agent's `<sync>` block.
  `{id, wait_ms}` blocks for finish, `{id}` snapshots, `{id, kill:true}`
  kills. Servers, watchers and long builds never hold a turn hostage.
- **async subagents** — `task{op:"spawn"}` launches a real subagent on a
  detached thread and returns the handle (`<aid>/sub-N`) instantly; the
  report arrives in the parent's inbox on completion — spawn is
  admission, not result. `op:"collect"` snapshots or blocks for results,
  `op:"list"` shows status. Spawned subs join the coordinator normally,
  so their claims participate in conflict enforcement.
- **persistent goals** — `goal{op:"set"}` pins an objective that the
  `<sync>` block re-surfaces every turn until `goal complete`; goals are
  mirrored into `coord status` so sibling agents see what everyone is
  driving at.
- **off-critical-path network** — memory capture ships turns to
  supermemory on a detached thread; nothing user-visible awaits HTTP.
- **continual refinement** — the `refine` tool (or
  `xera-rsi refine [--global] <instructions>`) runs a separate model
  pass over the trajectory and applies *precise* append/replace edits to
  harness files (`memory.md`, `lessons.md`, `guardrails.md`,
  `tool_guidance.md`, `skills/<name>.md`, `sub_agents/<name>/AGENT.md`)
  with per-edit evidence. Pre-edit snapshots land in
  `harness/snapshots/<ts>/`; every run appends to
  `harness/refinement_history.jsonl`. Local (workspace) scope by
  default; `--global` targets `~/.xera-rsi/harness`.

## The harness (Harness²)

The editable system around the model lives in
`<workspace>/.xera-rsi/harness` (falls back to `~/.xera-rsi/harness`).
The full harness2 c1..c8 surface — canonical names accepted everywhere:

```
c1  system_prompt.md | systemprompt.md | AGENTS.md      always-on prompt
c2  guardrails.md                                      hard constraints
c3  memory.md | LongTermMEMORY.md                      durable memory
c4  tool_guidance.md | tool_descriptions/tool_guidance.md
c5  subagents.md + sub_agents/<name>/AGENT.md          named personas
c6  skills/<name>.md | skills/<name>/SKILL.md          frontmatter'd skills
c7  scripts/<name>.<ext>                               helper scripts
c8  hooks.json | .codex/hooks.json                     lifecycle hooks
+   lessons.md                                         adopted lessons
```

- **skills** support frontmatter (`name:`, `description:`); descriptions are
  advertised in the system prompt and bodies load on demand via `skill_load`.
- **sub_agents** are named personas; the model spawns one with
  `task(prompt, agent="<name>")`.
- **hooks** fire inside the tool loop (the model can't skip them):
  `SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`,
  `SubagentStart`/`SubagentStop`, `PreCompact`/`PostCompact`, `Stop`.
  A hook command gets a JSON event on stdin; stdout
  `{"decision":"block","reason":"..."}` blocks the tool call and feeds the
  reason back to the model — see `hooks.example.json`.
- `xera-rsi harness` and `doctor` validate the surface (frontmatter
  problems, always-on cap, untriggered skills).

All of it is injected into the system prompt and all of it is editable — by
you, or by the `improve` loop.

## `xera-rsi improve`

Recursive task-level + harnessing-level recursion:

1. **Base** — run the task with the current harness, record the trajectory.
2. **Propose** — the harness editor proposes K candidate harness edits
   (file writes/appends/deletes inside the harness dir).
3. **Probe** — each candidate is materialized into an isolated harness copy and
   executed on the same task.
4. **Compose** — the composer reads the behavioral contrasts between the base
   and probe executions, adopts or merges the best ideas, applies the final
   edits, and appends an evidence-supported **lesson** to `lessons.md` (which
   persists into every future session — harnessing-level recursion).

```
xera-rsi improve "task" -k 2 --mode parallel      # default
xera-rsi improve "task" -k 3 --mode sequential    # refine step by step
xera-rsi improve "task" --dry-run                 # propose only
```

Run artifacts land in `.xera-rsi/runs/improve-<ts>/` in the harness2
evidence layout: `context/task.md`, `context/harness/` (base copy),
`context/rollouts/`, `context/deliverables/`,
`context/pool/<cand>/{harness/,harness.diff,rollout.md,produced.md}`,
`context/decisions/`. Candidate harnesses that fail surface validation are
recorded as rejected probes instead of being executed. Adopted harnesses are
snapshotted into `~/.xera-rsi/library/<id>/` (cross-task evidence).

## RSI loop (recursive self-improvement)

On top of Harness², xera-rsi keeps a persistent telemetry store and drives
an improvement loop off it:

- **traces** — every turn is appended to `~/.xera-rsi/traces/all.jsonl`:
  prompt, answer, tool calls, skills loaded, token usage, iterations,
  stop reason, duration, and the self-check verdict. Subagents record
  under `<session>/<aid>`. The model can query it itself with the
  `traces` tool (`stats`, `failures`, `space`, `dataset`).
- **self-check** — after each interactive answer, a cheap verification
  pass (no tools, temperature 0) judges whether the request was fully
  accomplished. The verdict lands in the trace store; a failure surfaces
  an **rsi proposal** panel (`a`/`enter` accept, `r`/`esc` reject — or
  click the buttons). Disable with `"rsi_verify": false` or
  `XERA_NO_VERIFY=1`.
- **`/rsi-loop [focus]`** — propose a loop manually; focus defaults to
  the last prompt.
- **the loop** (on accept) runs three phases on the worker:
  1. *analyze* — the model reads aggregate trace stats, recent verified
     failures and a 3D projection of every turn (x=tool calls, y=tokens
     out, z=skills) and returns a diagnosis + plan
  2. *harness* — evidence-backed edits applied through the refinement
     subsystem (same snapshots + history as `refine`)
  3. *train* — when the diagnosis says the gap is capability, not
     context: builds an SFT dataset from good traces
     (`~/.xera-rsi/train/data/`), optionally pulls a HuggingFace dataset
     the analyzer picked, and launches ETM (evict-train-merge) as a
     supervised background process with a log under
     `~/.xera-rsi/train/logs/` — the UI shows a "model is improving
     itself" indicator while it runs.

Nothing retrains or edits the harness without an explicit accept; a
rejected proposal just records itself and the session continues.

### RSI skills + no-code tools

Five skills ship seeded into every harness (`skills/<name>/SKILL.md` —
editable/removable like anything else in the harness), advertised in the
system prompt and loaded on demand with `skill_load`:

- **trace-mining** — read `traces{stats|failures|space}` for failure
  clusters, then decide harness-fix vs training
- **dataset-building** — build SFT data from verified traces, HF pulls,
  or hand synthesis; sizing and merge rules
- **data-quality** — audit with `train{op:"check"}` (exact/prompt dupes,
  echo/leak rows, degenerate rows, contamination vs a `ref` eval set),
  fix with `train{op:"clean"}`; what a good row looks like
- **training-setup** — launch an ETM run in one call:
  `train{op:"launch",preset:"standard",focus:"..."}` (auto-builds the
  dataset if `path` is omitted; `quick`/`standard`/`deep` presets,
  `train{op:"status"}` to watch)
- **harness-tuning** — improve the harness without touching code:
  `harness{op:"append",file:"tool_guidance.md",content:"..."}` etc.,
  snapshots + `harness{op:"rollback"}` for undo

The `train` tool drives the whole training pipeline
(dataset → check/clean → hf/pull → presets → launch → status) and the
`harness` tool gives direct, auto-snapshotted edits to every harness
file — both designed so the model never has to write scripts to improve
itself.

## Sessions

Every turn is appended as JSONL to `~/.xera-rsi/sessions/<id>.jsonl`.
`xera-rsi sessions` lists them; `xera-rsi resume <id>` or `/resume <id>`
loads one into context.

## Live observability

In parallel with session transcripts, every event is streamed to
`~/.xera-rsi/live/<session-id>.jsonl`: prompts, assistant text,
**reasoning traces**, tool calls + results, skills loaded
(`skill_load`), subagent spawns (`task`), per-call token usage, and turn ends.
Harness² runs append to `~/.xera-rsi/live/improvements.jsonl`
(`start` / `compose` / `dry_run` phases, with the adopted lesson).

A zero-dependency site renders all of it live in 3D — each session becomes a
helix of nodes colored by event type, like watching the agent's neural activity:

```
python3 tools/livesite/serve.py          # → http://localhost:8377
```

- **neural map** — live 3D graph (Three.js), nodes grow in as events stream;
  click a node for the full text/args; gold cluster at the center = harness²
  self-improvements.
- **statistics** — avg skills / tool calls / tokens per session, tool +
  skill histograms, per-session table, and the improvement feed with lessons.

`--live-dir` / `XERA_LIVE_DIR` overrides the source dir. The server is
read-only over the logs; agents never depend on it running.

## Build

Zig **0.14.1**:

```
zig build                    # debug
zig build -Doptimize=ReleaseSafe
```

Output: `zig-out/bin/xera-rsi(.exe)`.

## Layout

```
src/main.zig      CLI dispatch (chat/exec/improve/harness/sessions/resume/doctor)
src/config.zig    config.json + env + flag resolution
src/http.zig      HTTPS client + SSE reader
src/llm.zig       OpenAI-compatible client, streaming, tool-call assembly
src/tools.zig     tool registry + implementations + approval plumbing
src/agent.zig     the agent loop: turn, tool dispatch, subagents, compaction
src/harness.zig   Harness² components: load, assemble, edit, snapshot
src/improve.zig   Propose–Probe–Compose recursion
src/session.zig   JSONL transcripts, list, resume
src/live.zig      live event stream (~/.xera-rsi/live/*.jsonl)
src/coord.zig     inter-agent coordination fabric: claims, feed, inbox
src/memory.zig    supermemory v5 client: document add, hybrid search,
                  profile, forget (hosted or self-hosted)
src/refine.zig    continual-harness refinement: trajectory → typed
                  edits, snapshots, refinement_history.jsonl
src/traces.zig    telemetry store: turn/tool/skill/verdict JSONL, stats,
                  3D space projection, dataset export
src/rsi.zig       rsi loop: self-check verdicts, proposals, the
                  analyze→harness→train cycle
src/train.zig     dataset building, HF search/download, ETM repo +
                  supervised launch
tools/livesite/   zero-dep 3D observability site (serve.py + Three.js)
src/tui.zig       terminal layer: raw mode, alt screen, key input, styles
src/tui_app.zig   Codex-style TUI: scrollback blocks, composer, spinner,
                  approvals, worker thread + event queue, diffed rendering
src/repl.zig      plain line-mode fallback (--classic / non-TTY)
mock_llm.py       local OpenAI-compatible mock for offline testing
pty_test.py       ConPTY driver that screen-captures the TUI (dev)
```

## Known limits

- OpenAI-compatible endpoints only (Anthropic-native API not yet implemented)
- Shell sandboxing = approval gating only, no OS-level sandbox
- Improve probes run sequentially; parallel probes possible but not wired
- TUI is Windows + POSIX tested paths; `--classic` available everywhere
