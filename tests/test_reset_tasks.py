import sqlite3

import reset_tasks
from utils.control_plane import ControlPlane


def test_reset_tasks_clears_task_history_and_dashboard_state(tmp_path, monkeypatch, capsys):
    task_db = tmp_path / "task_history.db"
    control_db = tmp_path / "orchestrator_control.db"
    report_path = tmp_path / "nyx_report.html"
    monkeypatch.setattr(reset_tasks, "DB_PATH", str(task_db))
    monkeypatch.setattr(reset_tasks.cfg, "CONTROL_DB_PATH", str(control_db))
    monkeypatch.setattr(reset_tasks.cfg, "HTML_REPORT_PATH", str(report_path))

    with sqlite3.connect(task_db) as connection:
        connection.executescript(
            """
            CREATE TABLE tasks (id TEXT);
            CREATE TABLE logs (id TEXT);
            INSERT INTO tasks VALUES ('task-1');
            INSERT INTO logs VALUES ('event-1');
            """
        )

    control = ControlPlane(str(control_db))
    control.start_run("run-1", "Check the host")
    control.add_guidance("run-1", "Use read-only checks.")
    assert control.snapshot() is not None

    assert reset_tasks.main() == 0

    with sqlite3.connect(task_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM logs").fetchone()[0] == 0
    assert control.snapshot() is None
    assert "Reset dashboard state" in capsys.readouterr().out
    assert "Ready for a run" in report_path.read_text()
    assert "No objectives have been queued yet." in report_path.read_text()


def test_reset_tasks_clears_dashboard_even_without_task_database(tmp_path, monkeypatch, capsys):
    task_db = tmp_path / "missing_task_history.db"
    control_db = tmp_path / "orchestrator_control.db"
    report_path = tmp_path / "nyx_report.html"
    monkeypatch.setattr(reset_tasks, "DB_PATH", str(task_db))
    monkeypatch.setattr(reset_tasks.cfg, "CONTROL_DB_PATH", str(control_db))
    monkeypatch.setattr(reset_tasks.cfg, "HTML_REPORT_PATH", str(report_path))

    control = ControlPlane(str(control_db))
    control.start_run("run-2", "Check the host")

    assert reset_tasks.main() == 0

    assert control.snapshot() is None
    assert "No task DB found" in capsys.readouterr().out
    assert "Ready for a run" in report_path.read_text()
