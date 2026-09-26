# Nyx

Nyx is a local, terminal-driven AI agent for doing system work from a single high-level goal. It does not try to be a chatbot sitting in a web UI. It is designed to plan work, queue tasks, then operate inside a real terminal session and observe the actual output of commands before deciding what to do next.

The project is intentionally opinionated:

- a planner decides what should happen next
- an executor runs one shell command at a time in a real tmux session
- task state is persisted in SQLite so work can resume across runs
- completion is re-checked by the planner before the goal is treated as done

This is best understood as a small autonomous shell agent for local systems administration and development workflows.

## What it actually does

Given a goal like:

- "check whether the system is healthy"
- "fix the docker-compose setup for nginx"
- "verify the internet connection"
- "update the config file and confirm it is valid"

Nyx will:

1. ask a planning model to break the goal into tasks
2. enqueue those tasks in a priority queue
3. pick the highest-priority pending task
4. hand that task to an executor model
5. run shell commands in a live tmux terminal, using terminal scrollback as context
6. inspect command output, decide whether the task is complete or failed
7. retry or re-plan when needed
8. only mark a goal as finished when the planner confirms the task queue is empty and the outcome is valid

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

## Project structure

```text
nyx/
├── config/              # runtime settings and model config
├── data/                # SQLite DB and logs
├── executor/            # shell-driving executor and prompts
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
4. The executor chooses shell commands one at a time, reading output and terminal history as evidence
5. If a command fails, the planner may decide to retry differently or change the plan
6. Once the queue is empty and the planner accepts the evidence, the orchestration loop finishes

## Operational notes

This project is experimental and intentionally narrow in scope:

- it is meant to operate inside a local terminal environment
- it depends on real shell output rather than a synthetic sandbox
- it assumes you are comfortable with the model making filesystem and shell changes in your own environment
- it is best suited for tasks where terminal context matters more than pure API/tool orchestration

The queue and run state are persisted in SQLite, so a previous session can be resumed if that behavior is enabled in configuration. By default, stale pending tasks are cleared unless explicitly configured otherwise.

## Safety and expectations

Nyx is not a protected sandboxed agent. It can run commands in your active terminal, and it makes real changes to the environment if the task requires it. That is the intended design.

Use it for tasks you are comfortable delegating to a local shell agent. It is not a general-purpose, permissioned cloud automation platform.

## Status

This project is early-stage autonomy software for local terminal work. It is functional as a focused shell-operating planner/executor system, but it is still evolving and should be treated as experimental tooling rather than a polished end-user product.
