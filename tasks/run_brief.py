"""
Run brief — the single, shared picture of a run that every Nyx role receives.

Problem this solves
-------------------
The planner, executor and validator used to be mutually blind:

* the executor saw only its one-line task, so it could not know that another
  task was going to write ``index.html`` and invented ``COPY ./html`` instead;
* the validator judged each task against the *whole* goal, so it rejected
  "write the Dockerfile" for not also running the container;
* once a host port changed (8080 -> 8082 after a conflict) the pending tasks
  still carried the stale port.

The brief fixes this without breaking Nyx's "no terminal scrollback" rule. It
contains no raw terminal output. It carries only:

1. the overall goal (context, not this task's job),
2. the full ordered plan with the CURRENT task marked,
3. ESTABLISHED FACTS: deterministic observations extracted from real command
   results (files written + verified, image built, container + published
   port ...). Facts are computed by code, never by a model, so they cannot be
   hallucinated, and they override stale wording in task descriptions.
4. a note about the previous attempt when the current task is being retried.

Facts are stored in ``task.metadata["facts"]`` when a task completes, so they
persist with the task and survive a resume.
"""

from __future__ import annotations

import re
import shlex

from tasks.task import Task, TaskStatus

# Budgets keep the brief small enough for local models with modest contexts.
OUTCOME_CHARS = 140
FILE_EXCERPT_CHARS = 300
MAX_FILE_EXCERPTS = 4
RETRY_NOTE_CHARS = 400
MAX_BRIEF_CHARS = 5000

_DOCKER_RUN_VALUE_FLAGS = {
    "-p", "--publish", "--name", "-v", "--volume", "-e", "--env", "--network",
    "--net", "--restart", "-w", "--workdir", "-u", "--user", "-h", "--hostname",
    "--label", "-l", "--env-file", "--entrypoint", "-m", "--memory", "--cpus",
    "--mount", "--add-host", "--dns", "--platform", "--pull", "--cap-add",
    "--cap-drop", "--device", "--expose", "--ip", "--log-driver",
}


# ── Fact extraction (deterministic) ──────────────────────────────────────────


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _docker_args(command: str) -> tuple[str, list[str]] | None:
    """Return (subcommand, args after it) for a docker command, else None."""
    tokens = _tokens(command)
    for index, token in enumerate(tokens):
        if token == "docker" and index + 1 < len(tokens):
            return tokens[index + 1], tokens[index + 2:]
        if token not in {"sudo", "-E", "env"} and "=" not in token:
            # Something other than sudo/env prefixes precedes any "docker".
            break
    return None


def _flag_value(args: list[str], *names: str) -> str | None:
    for index, arg in enumerate(args):
        for name in names:
            if arg == name and index + 1 < len(args):
                return args[index + 1]
            if arg.startswith(f"{name}=") and len(name) > 2:
                return arg.split("=", 1)[1]
            if len(name) == 2 and arg.startswith(name) and len(arg) > 2 and not arg.startswith("--"):
                # e.g. -t name (handled above) / -tname (rare)
                return arg[2:]
    return None


def _parse_docker_run(args: list[str]) -> dict:
    name = _flag_value(args, "--name")
    publishes: list[str] = []
    image = None
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in {"-p", "--publish"} and index + 1 < len(args):
            publishes.append(args[index + 1])
            skip_next = True
            continue
        if arg.startswith("--publish="):
            publishes.append(arg.split("=", 1)[1])
            continue
        if arg.startswith("-"):
            if arg in _DOCKER_RUN_VALUE_FLAGS:
                skip_next = True
            continue
        image = arg
        break
    return {"name": name, "image": image, "publishes": publishes}


def _parse_publish(spec: str) -> tuple[str, str] | None:
    """'8082:80' / '127.0.0.1:8082:80/tcp' -> ('8082', '80')."""
    spec = spec.split("/", 1)[0]
    parts = spec.split(":")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        return parts[-2], parts[-1]
    return None


def _first_path_arg(args: list[str]) -> str | None:
    for arg in args:
        if not arg.startswith("-"):
            return arg
    return None


def extract_facts(step_log: list[dict]) -> list[dict]:
    """Derive authoritative facts from the successful steps of a task.

    Returns ``[{"key": ..., "text": ...}]``. A later fact with the same key
    replaces an earlier one (e.g. a container re-run on a different port).
    """
    facts: dict[str, str] = {}
    excerpts: dict[str, str] = {}

    for step in step_log or []:
        if not isinstance(step, dict) or step.get("exit_code") != 0:
            continue
        command = str(step.get("command", "")).strip()

        # Files written through the verified write_file workflow.
        if step.get("file_operation") == "verify" and step.get("file_path"):
            path = str(step["file_path"])
            facts[f"file:{path}"] = f"file written and verified: {path}"
            content = step.get("verified_content")
            if isinstance(content, str) and content.strip():
                excerpt = content.strip()[:FILE_EXCERPT_CHARS]
                if len(content.strip()) > FILE_EXCERPT_CHARS:
                    excerpt += " ..."
                excerpts[f"file:{path}"] = excerpt
            continue

        if command.startswith("<"):
            continue  # bookkeeping steps such as <read_file>, <human_input>

        tokens = _tokens(command)
        stripped = [t for t in tokens if t != "sudo"]
        if stripped[:1] == ["mkdir"]:
            target = _first_path_arg(stripped[1:])
            if target:
                facts[f"dir:{target}"] = f"directory exists: {target}"
            continue

        docker = _docker_args(command)
        if docker is None:
            continue
        sub, args = docker

        if sub == "build":
            tag = _flag_value(args, "-t", "--tag")
            context = args[-1] if args and not args[-1].startswith("-") else None
            if tag:
                where = f" (build context: {context})" if context else ""
                facts[f"image:{tag}"] = f"docker image built: {tag}{where}"
        elif sub == "run":
            info = _parse_docker_run(args)
            name = info["name"]
            if name:
                ports = [p for p in map(_parse_publish, info["publishes"]) if p]
                text = f"container {name} is running"
                if info["image"]:
                    text += f" from image {info['image']}"
                if ports:
                    host, container = ports[0]
                    text += (
                        f"; host port {host} -> container port {container}"
                        f" (reach it at http://localhost:{host}). Use host port {host}"
                        " for any check of this container; a different port in"
                        " task wording is outdated"
                    )
                facts[f"container:{name}"] = text
        elif sub in {"rm", "stop"}:
            for name in [a for a in args if not a.startswith("-")]:
                if f"container:{name}" in facts or sub == "rm":
                    facts[f"container:{name}"] = (
                        f"container {name} was {'removed' if sub == 'rm' else 'stopped'}"
                    )

    # Attach short content excerpts (most recent few) so later tasks can see
    # contracts such as a Dockerfile's COPY lines.
    result = []
    excerpt_keys = list(excerpts)[-MAX_FILE_EXCERPTS:]
    for key, text in facts.items():
        if key in excerpt_keys:
            indented = "\n".join(f"      {line}" for line in excerpts[key].splitlines())
            text = f"{text}\n{indented}"
        result.append({"key": key, "text": text})
    return result


def collect_facts(tasks: list[Task]) -> list[str]:
    """Merge facts from completed tasks in execution order; later keys win."""
    ordered = sorted(
        (t for t in tasks if t.status == TaskStatus.COMPLETED),
        key=lambda t: (t.completed_at or "", t.created_at or ""),
    )
    merged: dict[str, str] = {}
    for task in ordered:
        for fact in (task.metadata or {}).get("facts", []) or []:
            if isinstance(fact, dict) and fact.get("key") and fact.get("text"):
                merged.pop(fact["key"], None)  # re-insert so recency is preserved
                merged[fact["key"]] = fact["text"]
    return list(merged.values())


# ── Brief rendering ──────────────────────────────────────────────────────────


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def facts_section(tasks: list[Task]) -> str:
    facts = collect_facts(tasks)
    if not facts:
        return ""
    body = "\n".join(f"  - {fact}" for fact in facts)
    return (
        "ESTABLISHED FACTS (observed by Nyx from real command results; they take "
        "precedence over older wording in task descriptions):\n" + body
    )


def build_run_brief(
    goal: str,
    tasks: list[Task],
    current_task: Task | None = None,
    include_outcomes: bool = True,
    max_chars: int = MAX_BRIEF_CHARS,
) -> str:
    """Render the shared brief for one role.

    ``include_outcomes=False`` is used for the validator: it still learns the
    scope of the other tasks, but never sees sibling outcomes as evidence.
    """
    live = [t for t in tasks if t.status != TaskStatus.CANCELLED]
    ordered = sorted(live, key=lambda t: (t.priority, t.created_at or ""))

    plan_lines = []
    for index, task in enumerate(ordered, start=1):
        is_current = current_task is not None and task.id == current_task.id
        if is_current:
            label = "CURRENT TASK"
        elif task.status == TaskStatus.COMPLETED:
            label = "done"
        elif task.status == TaskStatus.FAILED:
            label = "failed"
        else:
            label = "pending"
        line = f"  {index}. [{label}] {_clip(task.description, 220)}"
        if include_outcomes and task.status == TaskStatus.COMPLETED and task.result and not is_current:
            line += f"\n       outcome: {_clip(task.result, OUTCOME_CHARS)}"
        plan_lines.append(line)

    sections = [
        "OVERALL GOAL (context only; you are responsible for the CURRENT TASK, "
        "other plan items belong to other tasks):\n" + _clip(goal, 700),
        "FULL PLAN (execution order):\n" + ("\n".join(plan_lines) if plan_lines else "  (no tasks yet)"),
    ]

    facts = facts_section(tasks)
    if facts:
        sections.append(facts)

    if current_task is not None and current_task.retries > 0 and current_task.error:
        sections.append(
            f"PREVIOUS ATTEMPT OF THIS TASK (attempt {current_task.retries} did not "
            f"succeed): {_clip(current_task.error, RETRY_NOTE_CHARS)}\n"
            "Inspect the current state first. Work from that attempt may already "
            "exist and be correct; verify it instead of redoing or deleting it."
        )

    brief = "\n\n".join(sections)
    if len(brief) > max_chars:
        # Facts and the retry note are the most valuable parts; trim the plan.
        keep = "\n\n".join(sections[2:])
        budget = max(600, max_chars - len(keep) - 4)
        head = "\n\n".join(sections[:2])
        brief = head[:budget].rstrip() + "\n  ...(plan truncated)\n\n" + keep
    return brief


# ── Dependency ordering ──────────────────────────────────────────────────────

_AUTHOR_VERB = re.compile(r"^\s*(write|create|author|generate|add|make|draft)\b", re.IGNORECASE)
_DOCKERFILE = re.compile(r"\bdockerfile\b", re.IGNORECASE)
_FILE_HINT = re.compile(
    r"\bfile\b|\.(?:html?|css|js|jsx|ts|tsx|py|php|conf|json|ya?ml|txt|sh)\b",
    re.IGNORECASE,
)
_DIR_ONLY = re.compile(r"\b(?:directory|folder)\b", re.IGNORECASE)


def _is_dockerfile_task(description: str) -> bool:
    return bool(_AUTHOR_VERB.match(description) and _DOCKERFILE.search(description))


def _is_app_file_task(description: str) -> bool:
    if not _AUTHOR_VERB.match(description) or _DOCKERFILE.search(description):
        return False
    if not _FILE_HINT.search(description):
        return False
    if _DIR_ONLY.search(description) and not re.search(r"\bfile\b", description, re.IGNORECASE):
        return False
    return True


def order_app_files_before_dockerfile(raw_tasks: list[dict]) -> list[dict]:
    """Make sure files a Dockerfile will COPY are written before the Dockerfile.

    The executor can only write a correct ``COPY`` line for files that exist
    (or that it was told exist), so within one planned batch any
    application-file task scheduled after a Dockerfile task is moved in front
    of it. The batch keeps the same multiset of priorities; only the
    assignment is permuted. Returns the original list unchanged when no
    reordering is needed.
    """
    indexed = sorted(
        enumerate(raw_tasks),
        key=lambda pair: (int(pair[1].get("priority", 1)), pair[0]),
    )
    planned = [task for _, task in indexed]
    original = list(planned)

    for task in list(planned):
        if not _is_dockerfile_task(str(task.get("description", ""))):
            continue
        current_pos = planned.index(task)
        later_app_files = [
            t for t in planned[current_pos + 1:]
            if _is_app_file_task(str(t.get("description", "")))
        ]
        if not later_app_files:
            continue
        remaining = [t for t in planned if t not in later_app_files]
        insert_at = remaining.index(task)
        planned = remaining[:insert_at] + later_app_files + remaining[insert_at:]

    if planned == original:
        return raw_tasks

    pool = sorted(int(t.get("priority", 1)) for t in planned)
    reordered = []
    for priority, task in zip(pool, planned):
        updated = dict(task)
        updated["priority"] = priority
        reordered.append(updated)
    return reordered
