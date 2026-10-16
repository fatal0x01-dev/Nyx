# Nyx

Nyx is a local, terminal-driven AI agent for doing system work from a single high-level goal. It does not try to be a chatbot sitting in a web UI. It is designed to plan work, queue tasks, then operate inside a real terminal session and observe the actual output of commands before deciding what to do next.

The project is intentionally opinionated:

- a planner decides what should happen next
- an executor runs one shell command at a time in a real tmux session
- the executor uses only command output from the current task, never prior terminal scrollback
- task state is persisted in SQLite so work can resume across runs
- completion is re-checked by the planner before the goal is treated as done

This is best understood as a small autonomous shell agent for local systems administration and development workflows.

## What it actually does

Given a goal like:

- "check whether the system is healthy"
- "fix the docker-compose setup for nginx"
- "run an Apache container on port 8080"
- "verify the internet connection"
- "update the config file and confirm it is valid"

Nyx will:

1. ask a planning model to break the goal into tasks
2. enqueue those tasks in a priority queue
3. pick the highest-priority pending task
4. hand that task to an executor model
5. run shell commands in a live tmux terminal, using only current-task command output as context
6. inspect command output, decide whether the task is complete or failed
7. retry or re-plan when needed
8. only mark a goal as finished when the planner confirms the task queue is empty and the outcome is valid

## Docker setup

For Docker application/service setup goals, Nyx creates the requested project
files and a Dockerfile through the verified `read_file`/`write_file` workflow,
builds a named local image, runs a named container, and verifies that
container. A request to run an existing image can use `docker run` directly.
Nyx uses Docker Compose only when explicitly requested. On a published-port
conflict, it retries with a different host port instead of changing host
Apache or other host services. Container cleanup targets only the named
container selected from `docker ps -a`; Nyx will not stop every container.

## Core design

The runtime is split into a few distinct layers:

- Planner: decides next steps and validates whether a task result is strong enough
- Executor: chooses the next shell command and drives a real terminal session
- TaskManager: priority-ordered queue backed by SQLite
- History: keeps recent execution state for planning context
- Memory: task store and retrieval layer, still lightweight compared to a full long-term memory system
- TmuxClient: uses tmux to attach to or create a live terminal session

This is not a browser automation project, a web app, or a generic agent framework. It is a local command-execution agent that works against the terminal and filesystem you already use.

## How it runs

The project is started from the repository root with:

```bash
bash run_nyx.sh
```

The script creates a Python virtual environment if needed, installs the required Python packages, then launches:

```bash
python orchestrator.py
```

If you run Nyx inside tmux, it will attach to the current tmux session. If not, it creates or uses a detached tmux session named `nyx` and tells you to attach to it with:

```bash
tmux attach -t nyx
```

## Requirements

- Python 3.10+
- tmux installed and available on PATH
- a local OpenAI-compatible model runner such as LM Studio or Ollama
- a model name that matches the configured model in `config/settings.py`

The default config is intentionally simple and local:

- planner and executor can use the same model in `SINGLE_MODEL_MODE = True`
- API base URL defaults to `http://localhost:11434/v1`
- the configured model is `qwen2.5-coder:14b`

You can change these in `config/settings.py`.

## Configuration

The main runtime settings live in:

- `config/settings.py`

Important knobs include:

- `SINGLE_MODEL_MODE`
- `PLANNER_MODEL`
- `EXECUTOR_MODEL`
- `MAX_ITERATIONS`
- `MAX_TASK_RETRIES`
- `TMUX_SESSION_NAME`
- `TASK_DB_PATH`

### Optional SSH execution

SSH is opt-in. Use `--ssh` to open a persistent remote shell in a new window
of the current tmux session before the planner starts; Nyx stays running in
its original window and executes tasks in the SSH window. Without that flag,
Nyx stays on its normal terminal.
The default target and compatibility options match:

```bash
ssh -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@172.16.42.1
```

For example:

```bash
bash run_nyx.sh --ssh --ip 172.16.42.1 "check the remote host"
bash run_nyx.sh --ssh --ip 192.0.2.10 --ssh-user admin --ssh-port 2222 "check the remote host"
```

`--ip` is required whenever `--ssh` is supplied. Use `--ssh-identity-file PATH` to select a key. Repeat `--ssh-option VALUE` to
specify custom OpenSSH `-o` options; providing one or more custom options
replaces the default legacy-RSA compatibility options. SSH authentication and
host-key confirmation prompts are handled in the terminal during connection.
After connection, planning and task validation are told which remote host is
active, and executor commands run in that remote shell.

## Project structure

```text
nyx/
├── config/              # runtime settings and model config
├── data/                # SQLite DB and logs
├── executor/            # shell-driving executor and prompts
├── frontend.py          # separate-terminal Textual run controls
├── memory/              # persistent task/memory store code
├── my-executor/         # separate executor prototype / compose setup
├── orchestrator.py      # main orchestration loop
├── planner/             # planning model + prompts
├── tasks/               # task definitions and queue management
├── tests/               # task validation and orchestration checks
├── utils/               # tmux, HTTP, helpers, logging
├── LICENSE
├── Pipfile
├── README.md
├── orchestrator.py
├── reset_tasks.py
├── run_nyx.sh
└── test_api.ps1
```

## Example workflow

A typical run looks like this:

1. You start Nyx with `bash run_nyx.sh`
2. You enter a goal such as: `check whether nginx is running and fix the docker-compose file if needed`
3. The planner creates tasks like:
   - inspect the compose file
   - validate the current config
   - update the nginx service if it is malformed
   - verify the service is running
4. The executor chooses shell commands one at a time, using output from the current task as evidence
5. If a command fails, the planner may decide to retry differently or change the plan
6. Once the queue is empty and the planner accepts the evidence, the orchestration loop finishes

## Operational notes

This project is experimental and intentionally narrow in scope:

- it is meant to operate inside a local terminal environment
- it depends on real shell output rather than a synthetic sandbox
- it assumes you are comfortable with the model making filesystem and shell changes in your own environment
- it is best suited for tasks where terminal context matters more than pure API/tool orchestration

The queue and run state are persisted in SQLite, so a previous session can be resumed if that behavior is enabled in configuration. By default, stale pending tasks are cleared unless explicitly configured otherwise.

## Live control dashboard

Keep the orchestrator's detailed debug output in its terminal. In a second
terminal, from the project directory, run:

```bash
./run_nyx.sh --frontend
```

The dashboard attaches to the latest Nyx run and shows its goal, phase,
iteration, and current task. Use **Pause run** to pause at the next safe
checkpoint (the current model request or shell command is allowed to finish),
and **Resume run** to continue. Enter a suggestion and press Enter or
**Send guidance** to include it in subsequent planner, executor, and
validation prompts for the current run. Suggestions do not rewrite the goal
and do not override Nyx's safety rules.

The dashboard and orchestrator communicate through
`data/orchestrator_control.db`; this is local control state, separate from the
task-history database and the debug logs. Close the dashboard with Ctrl+C; this
does not stop the orchestrator.

### Live HTML run report

Nyx also maintains a standalone HTML report at
`data/nyx_report.html`. Open that file in a browser while a run is active; it
refreshes automatically and shows the overall goal outcome, objective counts,
each task's status and result, and captured terminal evidence. The report is
updated throughout execution and remains available after Nyx exits. Run
`python reset_tasks.py` to clear task history, reset the control dashboard, and
restore the HTML report to its idle state.

## Safety and expectations

Nyx refuses recognized partition-table edits, filesystem formatting, raw-device
writes, and recursive deletion of system directories. If it encounters an
unsafe disk confirmation such as `Ignore/Cancel?`, it sends Ctrl+C rather than
choosing an option; if it cannot confirm the shell recovered, it stops the
orchestrator so no further commands are sent.
Any command refused by the high-risk-operation gate also ends the run without
automatic retries or alternate-command attempts.

These checks are defense-in-depth, not a security sandbox: arbitrary shell
commands and scripts can still make system changes. Review Nyx's goal and
suggestions, monitor its terminal, and do not delegate operations that must be
guaranteed harmless.

For explicit file-edit tasks, Nyx reads the complete UTF-8 text file, asks the
executor to produce the complete replacement contents, writes through an
atomic temporary-file replacement, and reads the file back to verify an exact
match. Editing is limited to files up to 48 KiB; symlinks and non-regular
targets are refused. Shell-based edits such as `sed -i` and redirection are
blocked for these tasks.

Use it for tasks you are comfortable delegating to a local shell agent. It is not a general-purpose, permissioned cloud automation platform.

## Status

This project is early-stage autonomy software for local terminal work. It is functional as a focused shell-operating planner/executor system, but it is still evolving and should be treated as experimental tooling rather than a polished end-user product.
