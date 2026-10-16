from tasks.task import Task, TaskStatus
from utils.html_report import render_html_report, write_html_report


def test_report_shows_outcome_objectives_and_terminal_evidence(tmp_path):
    task = Task(
        id="task-1",
        description='Check <host> & "network"',
        status=TaskStatus.COMPLETED,
        result="Connectivity confirmed.",
        metadata={
            "terminal_evidence": "$ ping -c 1 8.8.8.8\n64 bytes from 8.8.8.8"
        },
    )
    state = {
        "run_id": "run-1",
        "goal": "Establish connectivity",
        "status": "completed",
        "phase": "Complete",
        "iteration": 2,
        "current_task": "",
        "created_at": "2026-10-16T00:00:00+00:00",
        "updated_at": "2026-10-16T00:01:00+00:00",
    }
    report_path = tmp_path / "nyx_report.html"

    write_html_report(str(report_path), state, [task])
    report = report_path.read_text(encoding="utf-8")

    assert "Goal achieved" in report
    assert "Objectives met" in report
    assert ">1</strong><span>Objectives met" in report
    assert "Connectivity confirmed." in report
    assert "ping -c 1 8.8.8.8" in report
    assert "Check &lt;host&gt; &amp; &quot;network&quot;" in report
    assert '<meta http-equiv="refresh" content="2">' in report


def test_idle_report_is_clear_and_has_zero_objectives():
    report = render_html_report(None, [])

    assert "Ready for a run" in report
    assert "No active goal" in report
    assert ">0</strong><span>Objectives met" in report
    assert "No objectives have been queued yet." in report


def test_report_shows_failed_goal_and_noncompleted_objectives():
    state = {
        "run_id": "run-2",
        "goal": "Configure a service",
        "status": "failed",
        "phase": "Failed",
        "iteration": 3,
        "current_task": "",
    }
    tasks = [
        Task(id="failed", description="Restart service", status=TaskStatus.FAILED, error="Service unavailable"),
        Task(
            id="pending",
            description="Inspect service",
            status=TaskStatus.PENDING,
            result="Stale result from earlier attempt",
            error="Latest retry needs more evidence",
        ),
    ]

    report = render_html_report(state, tasks)

    assert "Goal not achieved" in report
    assert "Failed objective" in report
    assert "Service unavailable" in report
    assert "Latest retry needs more evidence" in report
    assert "Stale result from earlier attempt" not in report
    assert "Queued" in report
