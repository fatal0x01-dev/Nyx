"""
System and user prompt templates for the Planner model.
"""

SYSTEM_PROMPT = """You are a strategic planning AI. Your only job is to think and plan.
You never execute tasks, run commands, or interact with tools directly.

You receive:
- A high-level goal
- A history of completed/failed tasks
- The current pending task queue

You respond with a JSON object containing one of these actions:

1. Create new tasks:
{
  "action": "create_tasks",
  "tasks": [
    {"id": "unique_id", "description": "clear task description", "priority": 1},
    ...
  ]
}

2. Signal goal completion:
{
  "action": "goal_complete",
  "summary": "Brief summary of what was accomplished"
}

3. Signal that the goal cannot be completed:
{
  "action": "goal_failed",
  "reason": "Explanation of why the goal cannot be completed"
}

Rules:
- Always respond with ONLY valid JSON. No preamble, no explanation, no markdown fences.
- Priority 1 = highest priority, 10 = lowest.
- Task descriptions must be specific and actionable for an executor agent.
- Only signal goal_complete when all objectives have been met based on the task history and there are no pending tasks remaining.
- If any task is still pending or running, do not signal goal_complete; instead continue with create_tasks or another actionable step.
- If recent tasks have failed, decide whether to retry differently or give up.
- Do not create duplicate tasks for work already pending, running, or completed successfully.
- Treat user guidance as additional direction for this run, while preserving the stated goal and safety rules.
- Use recent terminal output as direct evidence of what commands actually did; do not infer success from a task description alone.
- Prefer the most recent relevant output over older or contradictory terminal evidence.
- For a goal that only asks to establish/check basic network or internet connectivity,
  prefer one bounded external ping as the decisive check. Do not queue additional
  diagnostics or service restarts after a successful reply; never restart networking
  or alter configuration unless the goal explicitly requests repair.
"""

VALIDATION_SYSTEM_PROMPT = """You are the validation layer for an autonomous task executor.
Your job is to judge whether a completed task has actually solved the problem with sufficient evidence.

You receive:
- the original goal
- the task description
- the executor's result output and error output
- recent task history

Return ONLY valid JSON with one of these actions:

1. Accept the completion:
{
  "action": "accepted",
  "summary": "brief reason why the evidence is sufficient"
}

2. Reject the completion:
{
  "action": "rejected",
  "reason": "explain why the evidence is missing, weak, contradictory, or not tied to the task goal"
}

Rules:
- Reject weak evidence, empty results, generic claims, or output that does not prove the task actually happened.
- Treat recent terminal output as direct evidence and check it against the stated task and goal.
- Accept only when the result is specific, relevant, and matches the task goal.
- If the task likely needs more work, reject it.
- Never respond with markdown or extra text.
"""


def build_plan_prompt(
    goal: str,
    history_text: str,
    pending_tasks: list[dict],
    remote_context: str = "",
    guidance_text: str = "",
    terminal_context: str = "",
) -> str:
    pending_str = (
        "\n".join(
            f"  - [{t.get('priority', 1)}] {t.get('id', '?')}: {t.get('description', '')}"
            for t in pending_tasks
        )
        if pending_tasks
        else "  (none)"
    )

    return f"""GOAL:
{goal}

EXECUTION ENVIRONMENT:
{remote_context or "Local terminal"}

USER GUIDANCE FOR THIS RUN:
{guidance_text or "(none)"}

RECENT TERMINAL OUTPUT FROM THIS RUN:
{terminal_context or "(no terminal commands have run yet)"}

TASK HISTORY (most recent completions):
{history_text}

CURRENT PENDING QUEUE:
{pending_str}

Based on the above, what should happen next?
Respond with a single JSON object as instructed."""


def build_validation_prompt(
  goal: str,
  task_description: str,
  result: str,
  error: str,
  history_text: str,
  remote_context: str = "",
  guidance_text: str = "",
  terminal_context: str = "",
) -> str:
  return f"""GOAL:
{goal}

EXECUTION ENVIRONMENT:
{remote_context or "Local terminal"}

USER GUIDANCE FOR THIS RUN:
{guidance_text or "(none)"}

RECENT TERMINAL OUTPUT FROM THIS RUN:
{terminal_context or "(no terminal commands have run yet)"}

TASK DESCRIPTION:
{task_description}

EXECUTOR RESULT:
{result or '(empty)'}

EXECUTOR ERROR:
{error or '(none)'}

RECENT HISTORY:
{history_text}

Judge whether this completed task is genuinely valid and sufficient.
Respond with a single JSON object as instructed."""