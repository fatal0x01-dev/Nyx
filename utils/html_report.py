"""Render the current Nyx run as a refreshable, standalone HTML report."""

from __future__ import annotations

import html
import os
import tempfile
from collections import Counter
from typing import Any

from tasks.task import Task


def _escape(value: Any) -> str:
    return html.escape(str("" if value is None else value), quote=True)


def _task_card(task: Task) -> str:
    status = task.status.value
    status_label = "Queued" if status == "pending" else status.replace("_", " ").title()
    details: list[str] = []
    result = task.result if status == "completed" else task.error or task.result
    if result:
        details.append(f'<div class="result">{_escape(result)}</div>')

    terminal_evidence = task.metadata.get("terminal_evidence", "")
    if terminal_evidence:
        details.append(
            '<details><summary>Terminal evidence</summary>'
            f'<pre>{_escape(terminal_evidence)}</pre></details>'
        )

    attempts = f'<span class="attempts">Retries: {task.retries}</span>' if task.retries else ""
    return (
        f'<article class="objective {status}">'
        '<div class="objective-heading">'
        f'<span class="badge {status}">{_escape(status_label)}</span>'
        f'<h3>{_escape(task.description)}</h3>{attempts}'
        '</div>'
        f'{"".join(details)}'
        '</article>'
    )


def render_html_report(state: dict[str, Any] | None, tasks: list[Task]) -> str:
    """Create a self-contained HTML dashboard from persisted run/task state."""
    counts = Counter(task.status.value for task in tasks)
    completed = counts["completed"]
    failed = counts["failed"]
    active_count = counts["running"] + counts["pending"]
    run_status = str(state.get("status", "idle")) if state else "idle"

    if state is None:
        outcome = "Ready for a run"
        outcome_detail = "Start Nyx to track objectives and results here."
    elif run_status == "completed":
        outcome = "Goal achieved"
        outcome_detail = state.get("phase", "Run completed successfully.")
    elif run_status == "failed":
        outcome = "Goal not achieved"
        outcome_detail = state.get("phase", "Run ended without meeting the goal.")
    elif run_status == "interrupted":
        outcome = "Run interrupted"
        outcome_detail = "The run stopped before its outcome was confirmed."
    elif run_status == "paused":
        outcome = "Run paused"
        outcome_detail = state.get("phase", "Waiting at a safe checkpoint.")
    else:
        outcome = "Run in progress"
        outcome_detail = state.get("phase", "Nyx is working on the goal.")

    objective_cards = "".join(_task_card(task) for task in tasks)
    if not objective_cards:
        objective_cards = '<p class="empty">No objectives have been queued yet.</p>'

    goal = state.get("goal", "") if state else ""
    run_id = state.get("run_id", "") if state else ""
    current_task = state.get("current_task", "") if state else ""
    created_at = state.get("created_at", "") if state else ""
    updated_at = state.get("updated_at", "") if state else ""
    progress = round(completed / len(tasks) * 100) if tasks else 0
    if run_status == "completed":
        progress = 100

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="2">
  <title>Nyx Run Report</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #0b1120; color: #e5eaf4; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 36px 22px 64px; }}
    .topline {{ color: #8ea0bb; font-size: .82rem; letter-spacing: .14em; text-transform: uppercase; }}
    h1 {{ margin: 8px 0 0; font-size: clamp(2rem, 6vw, 3.4rem); letter-spacing: -.04em; }}
    h2 {{ margin: 0; font-size: 1.15rem; }}
    .muted {{ color: #97a6bd; }}
    .hero, .panel {{ background: #111a2e; border: 1px solid #25334d; border-radius: 18px; }}
    .hero {{ margin-top: 24px; padding: 26px; }}
    .outcome {{ display: inline-flex; padding: 7px 11px; border-radius: 99px; background: #183a32; color: #86efac; font-weight: 700; }}
    .outcome.failed {{ background: #44242d; color: #fda4af; }}
    .outcome.running, .outcome.paused, .outcome.interrupted {{ background: #3b321d; color: #fcd34d; }}
    .goal {{ font-size: 1.25rem; line-height: 1.55; margin: 16px 0 8px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }}
    .stat {{ background: #0b1120; border: 1px solid #25334d; border-radius: 12px; padding: 14px; }}
    .stat strong {{ display: block; font-size: 1.6rem; }}
    .stat span {{ color: #97a6bd; font-size: .86rem; }}
    .progress {{ height: 8px; background: #26344c; border-radius: 99px; margin-top: 20px; overflow: hidden; }}
    .progress span {{ display: block; height: 100%; background: #34d399; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px 20px; margin-top: 14px; font-size: .85rem; }}
    .panel {{ margin-top: 20px; padding: 22px; }}
    .panel-heading {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 16px; }}
    .objective {{ background: #0b1120; border: 1px solid #26344c; border-left: 4px solid #64748b; border-radius: 12px; padding: 15px; margin: 10px 0; }}
    .objective.completed {{ border-left-color: #34d399; }}
    .objective.failed {{ border-left-color: #fb7185; }}
    .objective.running {{ border-left-color: #60a5fa; }}
    .objective.pending {{ border-left-color: #fbbf24; }}
    .objective.cancelled {{ border-left-color: #94a3b8; }}
    .objective-heading {{ display: flex; align-items: center; gap: 10px; }}
    .objective h3 {{ margin: 0; flex: 1; font-size: 1rem; line-height: 1.5; }}
    .badge {{ white-space: nowrap; border-radius: 99px; padding: 4px 9px; background: #26344c; color: #cbd5e1; font-size: .75rem; font-weight: 700; }}
    .badge.completed {{ color: #86efac; background: #183a32; }}
    .badge.failed {{ color: #fda4af; background: #44242d; }}
    .badge.running {{ color: #93c5fd; background: #1e3554; }}
    .badge.pending {{ color: #fcd34d; background: #3b321d; }}
    .result {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 12px 0 0 2px; color: #cbd5e1; line-height: 1.55; }}
    details {{ margin: 12px 0 0 2px; color: #93c5fd; }}
    summary {{ cursor: pointer; }}
    pre {{ overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; padding: 13px; border-radius: 9px; background: #070b14; color: #cbd5e1; line-height: 1.5; }}
    .empty {{ color: #97a6bd; }}
    footer {{ margin-top: 20px; color: #64748b; font-size: .8rem; text-align: center; }}
    @media (max-width: 650px) {{ .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .objective-heading {{ align-items: flex-start; flex-wrap: wrap; }} }}
  </style>
</head>
<body>
  <main>
    <div class="topline">Nyx · Run report</div>
    <h1>Mission status</h1>
    <section class="hero">
      <span class="outcome {_escape(run_status)}">{_escape(outcome)}</span>
      <p class="goal">{_escape(goal or "No active goal")}</p>
      <p class="muted">{_escape(outcome_detail)}</p>
      <div class="stats">
        <div class="stat"><strong>{completed}</strong><span>Objectives met</span></div>
        <div class="stat"><strong>{failed}</strong><span>Failed objectives</span></div>
        <div class="stat"><strong>{active_count}</strong><span>In progress / queued</span></div>
        <div class="stat"><strong>{len(tasks)}</strong><span>Total objectives</span></div>
      </div>
      <div class="progress" aria-label="Completed objective progress"><span style="width: {progress}%"></span></div>
      <div class="meta muted">
        <span>Run: {_escape(run_id or "—")}</span>
        <span>Iteration: {_escape(state.get("iteration", 0) if state else 0)}</span>
        <span>Current task: {_escape(current_task or "None")}</span>
        <span>Started: {_escape(created_at or "—")}</span>
        <span>Last updated: {_escape(updated_at or "—")}</span>
      </div>
    </section>
    <section class="panel">
      <div class="panel-heading"><h2>Objectives and results</h2><span class="muted">Automatically refreshes every 2 seconds</span></div>
      {objective_cards}
    </section>
    <footer>Generated by Nyx · This report is a local static HTML file.</footer>
  </main>
</body>
</html>
"""


def write_html_report(path: str, state: dict[str, Any] | None, tasks: list[Task]) -> None:
    """Atomically replace the report so a browser never reads a partial document."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    contents = render_html_report(state, tasks)
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=".nyx-report-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(contents)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
