from executor.executor import Executor
from orchestrator import Orchestrator
from tasks.task import Task
from tasks.task_manager import TaskManager
from memory.task_store import TaskStore
from config import settings
from utils.helpers import normalize_multiline_write_command


def testmarks_completed_only_when_terminal_context_is_confident():
    assert Executor.validate_completion_status(
        status="completed",
        result="service is running and responding on port 3000",
        error="",
        terminal_history="Listening on 0.0.0.0:3000",
        last_exit_code=0,
    ) is True

    assert Executor.validate_completion_status(
        status="completed",
        result="",
        error="",
        terminal_history="Traceback (most recent call last):\nModuleNotFoundError",
        last_exit_code=1,
    ) is False

    assert Executor.validate_completion_status(
        status="completed",
        result="",
        error="",
        terminal_history="This will overwrite existing files. Continue? [Y/n]",
        last_exit_code=None,
    ) is False


def test_orchestrator_rejects_goal_complete_when_pending_tasks_remain():
    assert Orchestrator._completion_is_valid(
        {"action": "goal_complete", "summary": "done"},
        [Task(id="t1", description="run docker-compose up")],
    ) is False

    assert Orchestrator._completion_is_valid(
        {"action": "goal_complete", "summary": "done"},
        [],
    ) is True


def test_retried_pending_tasks_are_not_treated_as_live_work_after_goal_confirmation():
    orchestrator = Orchestrator()
    stale_retry = Task(id="retry", description="verify internet connectivity", retries=1)
    fresh_task = Task(id="fresh", description="inspect unrelated service")
    orchestrator._manager.add_task(stale_retry)

    orchestrator._retire_stale_retries([stale_retry])

    assert stale_retry.status.value == "cancelled"
    assert orchestrator._manager.next_task() is None
    assert fresh_task.retries == 0
    assert Orchestrator._completion_is_valid(
        {"action": "goal_complete", "summary": "internet connectivity verified"},
        [fresh_task],
    ) is False


def test_retry_demotes_a_failed_task_so_other_tasks_can_run():
    store = TaskStore(settings.TASK_DB_PATH)
    store.clear_all_tasks()
    manager = TaskManager(store)

    first = Task(id="t1", description="inspect docker-compose", priority=1)
    second = Task(id="t2", description="update docker-compose", priority=3)
    manager.add_task(first)
    manager.add_task(second)

    manager.mark_retry(first, "still failing")

    assert first.priority > 1
    assert manager.next_task().id == second.id


def test_completion_validation_ignores_stale_scrollback_from_earlier_errors():
    assert Executor.validate_completion_status(
        status="completed",
        result="nginx is running on port 80",
        error="",
        terminal_history="docker-compose: command not found\nthis line is older history",
        last_exit_code=0,
        step_log=[
            {"command": "docker-compose config --dry-run", "output": "nginx is running on port 80", "exit_code": 0, "timed_out": False},
        ],
    ) is True


def test_executor_stops_on_repeated_no_progress():
    step_log = [
        {"command": "docker-compose config", "output": "go-yaml load error in scanner at L1.C47", "exit_code": 1, "timed_out": False},
        {"command": "docker-compose config", "output": "go-yaml load error in scanner at L1.C47", "exit_code": 1, "timed_out": False},
    ]
    assert Executor._should_stop_on_no_progress(step_log) is True


def test_no_progress_guard_handles_missing_output_without_crashing():
    assert Executor._should_stop_on_no_progress([
        {"command": "ping -c 1 google.com", "output": ""},
        {"command": "ping -c 1 google.com", "output": ""},
    ]) is True


def test_executor_normalizes_shell_action_aliases_and_network_ping_count():
    normalized = Executor._normalize_action(
        {"action": "run", "command": "ping -c 4 8.8.8.8"},
        "verify internet connectivity",
    )
    assert normalized["action"] == "run_command"
    assert normalized["command"] == "ping -c 1 8.8.8.8"

    direct_command = Executor._normalize_action(
        {"action": "ping -c 4 google.com"},
        "verify internet connectivity",
    )
    assert direct_command["action"] == "run_command"
    assert direct_command["command"] == "ping -c 1 google.com"


def test_file_edit_tasks_refuse_non_write_actions():
    task = "Fix docker-compose.yml by writing the correct nginx service"
    assert Executor._requires_file_write(task) is True
    assert Executor._is_file_write_command("cat > docker-compose.yml <<'EOF'\nversion: '3.8'\nEOF") is True
    assert Executor._is_file_write_command("docker-compose up -d") is False
    assert Executor._needs_write_before_retry(
        task,
        [{"command": "docker-compose config", "output": "name: ubuntu-docker\nservices:\n  ubuntu: ..."}],
    ) is True


def test_compose_task_requires_matching_service_name():
    task = "configure nginx service in /home/fatal0x01/ubuntu-docker"
    assert Executor._task_matches_compose_requirement(task, "services:\n  nginx:\n    image: nginx:latest") is True
    assert Executor._task_matches_compose_requirement(task, "services:\n  ubuntu:\n    image: ubuntu:latest") is False


def test_completion_rejects_ubuntu_when_task_requires_nginx():
    assert Executor.validate_completion_status(
        status="completed",
        result="docker-compose config succeeded",
        error="",
        terminal_history="name: ubuntu-docker\nservices:\n  nginx:\n    image: nginx\n  ubuntu:\n    image: ubuntu",
        last_exit_code=0,
        step_log=[
            {"command": "docker-compose config", "output": "name: ubuntu-docker\nservices:\n  nginx:\n    image: nginx\n  ubuntu:\n    image: ubuntu", "exit_code": 0, "timed_out": False},
        ],
        task_description="configure nginx service and run it",
    ) is False


def test_validation_tasks_require_a_file_read_before_completion():
    assert Executor._is_validation_task("Validate the docker-compose.yml file for syntax errors") is True
    assert Executor._has_file_read_step([
        {"command": "docker-compose config", "output": "name: ubuntu-docker", "exit_code": 0, "timed_out": False},
    ]) is False
    assert Executor._has_file_read_step([
        {"command": "cat docker-compose.yml", "output": "services:\n  nginx:\n    image: nginx", "exit_code": 0, "timed_out": False},
    ]) is True


def test_network_check_success_is_recognized_as_done():
    assert Executor._is_network_check_task("Check current network connection status") is True
    assert Executor.validate_completion_status(
        status="completed",
        result="Network connection to google.com is active with 0% packet loss.",
        error="",
        terminal_history="PING google.com (142.250.190.46) 56(84) bytes of data.\n64 bytes from ... time=12.3 ms\n--- google.com ping statistics ---\n1 packets transmitted, 1 received, 0% packet loss",
        last_exit_code=0,
        step_log=[
            {"command": "ping -c 1 google.com", "output": "64 bytes from ... time=12.3 ms\n0% packet loss", "exit_code": 0, "timed_out": False},
        ],
        task_description="Check current network connection status",
    ) is True


def test_orchestrator_requires_planner_validation_after_task_completion(monkeypatch):
    orchestrator = Orchestrator()
    task = Task(id="t1", description="check docker-compose.yml")
    result = {"status": "completed", "result": "docker-compose config succeeded"}

    def fake_validate_task_result(**kwargs):
        return {"action": "rejected", "reason": "The task result is insufficient; no direct validation was performed."}

    monkeypatch.setattr(orchestrator._planner, "validate_task_result", fake_validate_task_result)

    assert orchestrator._validate_task_completion(task, result) is False


def test_normalizes_escaped_newline_yaml_writes():
    cmd = "echo 'version: \"3.8\"\\nservices:\\n  nginx:\\n    image: nginx:latest\\n    ports:\\n      - \"80:80\"' > docker-compose.yml"
    normalized = normalize_multiline_write_command(cmd)
    assert "cat > docker-compose.yml <<'EOF'" in normalized
    assert "version: \"3.8\"" in normalized
    assert "\\n" not in normalized
