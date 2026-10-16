from types import SimpleNamespace

import re
import threading
import time

import pytest

from executor.executor import Executor
from orchestrator import Orchestrator, parse_cli_args
from tasks.task import Task
from tasks.task_manager import TaskManager
from memory.task_store import TaskStore
from config import settings
from utils.helpers import normalize_multiline_write_command
from utils.tmux_client import TmuxClient
from utils.ssh import SSHConfig, build_ssh_command
from utils.tmux_client import TmuxError
from utils.control_plane import ControlPlane
from executor.shell_prompts import build_step_prompt
from planner.prompts import build_plan_prompt


def test_marks_completed_only_when_current_task_evidence_is_confident():
    assert Executor.validate_completion_status(
        status="completed",
        result="service is running and responding on port 3000",
        error="",
        last_exit_code=0,
        step_log=[{"output": "Listening on 0.0.0.0:3000"}],
    ) is True

    assert Executor.validate_completion_status(
        status="completed",
        result="",
        error="",
        last_exit_code=1,
        step_log=[{"output": "Traceback (most recent call last):\nModuleNotFoundError"}],
    ) is False

    assert Executor.validate_completion_status(
        status="completed",
        result="",
        error="",
        last_exit_code=None,
        step_log=[{"output": "This will overwrite existing files. Continue? [Y/n]"}],
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


def test_completion_validation_uses_only_current_task_output():
    assert Executor.validate_completion_status(
        status="completed",
        result="nginx is running on port 80",
        error="",
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


def test_successful_ping_satisfies_basic_connectivity_goal_only():
    successful_ping = [
        {"command": "ping -c 1 8.8.8.8", "output": "64 bytes from 8.8.8.8", "exit_code": 0},
    ]
    failed_ping = [
        {"command": "ping -c 1 8.8.8.8", "output": "100% packet loss", "exit_code": 1},
    ]

    assert Orchestrator._connectivity_goal_satisfied(
        "establish network connectivity",
        successful_ping,
    ) is True
    assert Orchestrator._connectivity_goal_satisfied(
        "diagnose network connectivity",
        successful_ping,
    ) is False
    assert Orchestrator._connectivity_goal_satisfied(
        "establish network connectivity",
        failed_ping,
    ) is False
    assert Orchestrator._connectivity_goal_satisfied(
        "check service status",
        successful_ping,
    ) is False
    assert Executor._is_basic_connectivity_task(
        "Check current network connectivity on the remote host"
    ) is True
    assert Executor._is_basic_connectivity_task(
        "Diagnose connectivity and repair the firewall"
    ) is False


def test_executor_finishes_basic_connectivity_task_after_one_successful_ping(monkeypatch):
    command_calls = []

    class FakeTmux:
        @staticmethod
        def get_target():
            return "nyx:1.0", True

        @staticmethod
        def run_command(target, command, timeout):
            command_calls.append(command)
            return {
                "output": "1 packets transmitted, 1 received, 0% packet loss",
                "exit_code": 0,
                "timed_out": False,
            }

    executor = Executor.__new__(Executor)
    executor._tmux = FakeTmux()
    executor._last_step_log = []
    executor._ssh_target = None
    executor._remote_context = ""
    executor._control_plane = None
    executor._control_run_id = None
    monkeypatch.setattr(TmuxClient, "is_installed", staticmethod(lambda: True))
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda task, step_log, step: {
            "action": "run_command",
            "command": "ping -c 1 8.8.8.8",
        },
    )

    result = executor.execute(Task(id="connectivity", description="Check network connectivity"))

    assert result == {
        "status": "completed",
        "result": "Network connectivity confirmed by a successful ping.",
    }
    assert command_calls == ["ping -c 1 8.8.8.8"]


def test_successful_connectivity_probe_cancels_remaining_planned_tasks(tmp_path, monkeypatch):
    store = TaskStore(str(tmp_path / "tasks.db"))
    monkeypatch.setattr(settings, "HTML_REPORT_PATH", str(tmp_path / "nyx_report.html"))
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._goal = "establish network connectivity"
    orchestrator._iteration = 0
    orchestrator._remote_context = ""
    orchestrator._recent_terminal_context = []
    orchestrator._start_time = 0.0
    orchestrator._run_id = "connectivity-run"
    orchestrator._control = ControlPlane(str(tmp_path / "control.db"))
    orchestrator._control.start_run(orchestrator._run_id, orchestrator._goal)
    orchestrator._manager = TaskManager(store)
    orchestrator._history = SimpleNamespace(as_prompt_text=lambda limit: "")
    orchestrator._store = store
    orchestrator._executor = SimpleNamespace(
        _last_step_log=[
            {
                "command": "ping -c 1 8.8.8.8",
                "output": "64 bytes from 8.8.8.8: icmp_seq=1 ttl=117 time=12 ms",
                "exit_code": 0,
            }
        ]
    )
    orchestrator._planner = SimpleNamespace(
        plan=lambda **kwargs: {"action": "create_tasks", "tasks": []}
    )
    ping_task = Task(id="ping", description="Ping an external address")
    restart_task = Task(id="restart", description="Restart networking service", priority=2)
    orchestrator._manager.add_task(ping_task)
    orchestrator._manager.add_task(restart_task)
    finishes = []
    monkeypatch.setattr(orchestrator, "_finish", lambda **kwargs: finishes.append(kwargs))

    def complete_ping(task):
        orchestrator._manager.mark_completed(task, "Connectivity confirmed.")

    monkeypatch.setattr(orchestrator, "_run_task", complete_ping)

    orchestrator._loop()

    assert ping_task.status.value == "completed"
    assert restart_task.status.value == "cancelled"
    assert len(finishes) == 1
    assert finishes[0]["success"] is True


def test_control_plane_shares_pause_status_and_guidance(tmp_path):
    control = ControlPlane(str(tmp_path / "control.db"))
    control.start_run("run-1", "check remote host")

    control.set_paused("run-1", True)
    control.add_guidance("run-1", "Prefer read-only checks.")
    snapshot = control.snapshot("run-1")

    assert snapshot["status"] == "paused"
    assert snapshot["paused"] is True
    assert snapshot["guidance"][0]["content"] == "Prefer read-only checks."
    assert control.guidance_text("run-1") == "- Prefer read-only checks."

    control.set_paused("run-1", False)
    assert control.snapshot("run-1")["status"] == "running"


def test_control_plane_pause_waits_until_resume(tmp_path):
    control = ControlPlane(str(tmp_path / "control.db"))
    control.start_run("run-2", "check remote host")
    control.set_paused("run-2", True)
    resumed = threading.Event()

    def wait_for_resume():
        control.wait_if_paused("run-2", poll_interval=0.01)
        resumed.set()

    worker = threading.Thread(target=wait_for_resume)
    worker.start()
    time.sleep(0.04)
    assert not resumed.is_set()

    control.set_paused("run-2", False)
    assert resumed.wait(timeout=1)
    worker.join(timeout=1)
    assert not worker.is_alive()


def test_control_plane_rejects_guidance_after_run_finishes(tmp_path):
    control = ControlPlane(str(tmp_path / "control.db"))
    control.start_run("run-3", "check remote host")
    control.update_run("run-3", status="completed")

    with pytest.raises(ValueError, match="no longer active"):
        control.add_guidance("run-3", "Run another check.")
    with pytest.raises(ValueError, match="no longer active"):
        control.set_paused("run-3", True)


def test_user_guidance_is_included_in_planner_and_executor_prompts():
    guidance = "- Prefer read-only checks.\n- Do not restart network services."
    plan_prompt = build_plan_prompt(
        "Check connectivity",
        "No completed tasks.",
        [],
        guidance_text=guidance,
    )
    executor_prompt = build_step_prompt(
        task_id="task-1",
        task_description="Check connectivity",
        step_log=[],
        steps_taken=0,
        max_steps=12,
        guidance_text=guidance,
    )

    assert guidance in plan_prompt
    assert guidance in executor_prompt


def test_terminal_context_is_included_in_planner_prompt():
    terminal_context = "$ ping -c 1 8.8.8.8\n64 bytes from 8.8.8.8\nexit_code=0"
    prompt = build_plan_prompt(
        "establish network connectivity",
        "No completed tasks.",
        [],
        terminal_context=terminal_context,
    )

    assert terminal_context in prompt


def test_orchestrator_formats_terminal_command_evidence():
    evidence = Orchestrator._format_terminal_evidence(
        Task(id="task-1", description="Check connectivity"),
        [{
            "command": "ping -c 1 8.8.8.8",
            "output": "64 bytes from 8.8.8.8",
            "exit_code": 0,
            "timed_out": False,
        }],
    )

    assert "TASK task-1: Check connectivity" in evidence
    assert "$ ping -c 1 8.8.8.8" in evidence
    assert "exit_code=0 timed_out=False" in evidence
    assert "64 bytes from 8.8.8.8" in evidence


def test_tmux_command_result_extracts_only_current_command_output():
    lines = [
        "old command: failed",
        "__NYX_START_token__",
        "current command succeeded",
        "__NYX_DONE_token___0",
    ]
    assert TmuxClient._command_result(lines, "__NYX_START_token__", "__NYX_DONE_token__") == (
        "current command succeeded",
        0,
    )


def test_tmux_command_result_works_when_start_marker_scrolled_off_screen():
    lines = ["current command output", "__NYX_DONE_token___1"]
    assert TmuxClient._command_result(lines, "__NYX_START_token__", "__NYX_DONE_token__") == (
        "current command output",
        1,
    )


def test_tmux_detects_common_pager_screens():
    assert TmuxClient._is_pager_screen("lines 1-20/60 (END)") is True
    assert TmuxClient._is_pager_screen("--More--") is True
    assert TmuxClient._is_pager_screen("network status is healthy") is False


def test_tmux_exits_detected_pager_and_collects_command_result(monkeypatch):
    client = TmuxClient()
    commands = []
    screen_calls = 0
    start_marker = ""
    done_marker = ""

    def fake_run(args):
        nonlocal start_marker, done_marker
        commands.append(args)
        if args[0] == "send-keys" and "Enter" in args:
            command = args[3]
            start_marker = re.search(r"__NYX_START_[a-f0-9]+__", command).group(0)
            done_marker = re.search(r"__NYX_DONE_[a-f0-9]+__", command).group(0)
        return ""

    def fake_capture_screen(target):
        nonlocal screen_calls
        screen_calls += 1
        if screen_calls == 1:
            return f"{start_marker}\nlines 1-20/60 (END)"
        return f"{start_marker}\nstatus output\n{done_marker}_0"

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(client, "capture_screen", fake_capture_screen)
    monkeypatch.setattr("utils.tmux_client.time.sleep", lambda _: None)

    result = client.run_command("nyx:0.0", "systemctl status networking", poll_interval=0)

    assert result["output"] == "status output"
    assert result["exit_code"] == 0
    assert ["send-keys", "-t", "nyx:0.0", "q"] in commands


def test_tmux_interrupts_timed_out_command_and_waits_for_sentinel(monkeypatch):
    client = TmuxClient()
    commands = []
    marker_values = {}
    capture_count = 0

    def fake_run(args):
        commands.append(args)
        if args[0] == "send-keys" and "Enter" in args:
            wrapped_command = args[3]
            marker_values["start"] = re.search(
                r"__NYX_START_[a-f0-9]+__", wrapped_command
            ).group(0)
            marker_values["done"] = re.search(
                r"__NYX_DONE_[a-f0-9]+__", wrapped_command
            ).group(0)
        return ""

    def fake_capture_screen(target):
        nonlocal capture_count
        capture_count += 1
        start = marker_values["start"]
        if capture_count == 1:
            return f"{start}\npartial output"
        return f"{start}\npartial output\n{marker_values['done']}_130"

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(client, "capture_screen", fake_capture_screen)
    monkeypatch.setattr("utils.tmux_client.time.sleep", lambda _: None)

    result = client.run_command("nyx:0.0", "top", timeout=0, poll_interval=0)

    assert result["timed_out"] is True
    assert result["output"] == "partial output"
    assert result["exit_code"] == 130
    assert [cmd for cmd in commands if cmd[0] == "send-keys"][-1][3] == "C-c"


def test_tmux_cancels_ignore_cancel_prompt_without_sending_wrapper_text(monkeypatch):
    client = TmuxClient()
    commands = []
    markers = {}
    capture_count = 0

    def fake_run(args):
        commands.append(args)
        if args[0] == "send-keys" and "Enter" in args:
            wrapped_command = args[3]
            assert "\n" not in wrapped_command
            markers["start"] = re.search(
                r"__NYX_START_[a-f0-9]+__", wrapped_command
            ).group(0)
            markers["done"] = re.search(
                r"__NYX_DONE_[a-f0-9]+__", wrapped_command
            ).group(0)
        return ""

    def fake_capture_screen(target):
        nonlocal capture_count
        capture_count += 1
        if capture_count == 1:
            return (
                f"{markers['start']}\nError: Can't have a partition outside the disk!\n"
                "Ignore/Cancel?"
            )
        return (
            f"{markers['start']}\nError: Can't have a partition outside the disk!\n"
            f"Ignore/Cancel? ^C\n{markers['done']}_130"
        )

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(client, "capture_screen", fake_capture_screen)
    monkeypatch.setattr("utils.tmux_client.time.sleep", lambda _: None)

    result = client.run_command("nyx:0.0", "sudo parted -l", timeout=2, poll_interval=0)

    sent_keys = [command[3] for command in commands if command[0] == "send-keys"]
    assert sent_keys[-1] == "C-c"
    assert "__nyx_status_" not in sent_keys[-1]
    assert result["safety_interrupted"] is True
    assert result["human_input_needed"] is False
    assert result["exit_code"] == 130
    assert "canceled an unsafe interactive confirmation" in result["output"]


def test_tmux_halts_if_unsafe_prompt_cannot_be_interrupted(monkeypatch):
    client = TmuxClient()
    markers = {}

    def fake_run(args):
        if args[0] == "send-keys" and "Enter" in args:
            wrapped_command = args[3]
            assert "\n" not in wrapped_command
            markers["start"] = re.search(
                r"__NYX_START_[a-f0-9]+__", wrapped_command
            ).group(0)
            markers["done"] = re.search(
                r"__NYX_DONE_[a-f0-9]+__", wrapped_command
            ).group(0)
            return ""
        raise TmuxError("unable to send Ctrl-C")

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(
        client,
        "capture_screen",
        lambda target: f"{markers['start']}\nIgnore/Cancel?",
    )

    result = client.run_command("nyx:0.0", "sudo parted -l", timeout=2, poll_interval=0)

    assert result["safety_interrupted"] is True
    assert result["safety_halt"] is True
    assert result["timed_out"] is True


def test_systemctl_status_commands_disable_the_pager():
    assert Executor._disable_systemctl_pager(
        "sudo systemctl status networking"
    ) == "sudo systemctl --no-pager status networking"
    assert Executor._disable_systemctl_pager(
        "/usr/bin/systemctl --no-pager status networking"
    ) == "/usr/bin/systemctl --no-pager status networking"


def test_ssh_config_builds_requested_rsa_compatible_remote_shell_command():
    config = SSHConfig(ip="172.16.42.1")

    command = build_ssh_command(config, "start-marker", "ready-marker")

    assert "ssh -tt -o HostKeyAlgorithms=+ssh-rsa" in command
    assert "-o PubkeyAcceptedAlgorithms=+ssh-rsa root@172.16.42.1" in command
    assert "ready-marker" in command


def test_ssh_config_accepts_cli_overrides():
    goal, config = parse_cli_args([
        "--ssh",
        "--ip",
        "192.0.2.10",
        "--ssh-user",
        "admin",
        "--ssh-port",
        "2222",
        "--ssh-identity-file",
        "/tmp/ssh key",
        "--ssh-option",
        "ConnectTimeout=10",
        "check remote status",
    ])

    assert goal == "check remote status"
    assert config == SSHConfig(
        ip="192.0.2.10",
        username="admin",
        port=2222,
        identity_file="/tmp/ssh key",
        options=("ConnectTimeout=10",),
    )


def test_tmux_opens_ssh_and_waits_for_remote_ready_marker(monkeypatch):
    client = TmuxClient()
    sent_commands = []
    screen_reads = []

    def fake_run(args):
        sent_commands.append(args)
        ready_marker = re.search(r"__NYX_SSH_READY_[a-f0-9]+__", args[3]).group(0)
        screen_reads.append(ready_marker)
        return ""

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(client, "capture_screen", lambda target: screen_reads[0])
    monkeypatch.setattr("utils.tmux_client.time.sleep", lambda _: None)

    client.start_ssh("nyx:0.0", SSHConfig(ip="172.16.42.1"), timeout=1, poll_interval=0)

    assert sent_commands[0][:3] == ["send-keys", "-t", "nyx:0.0"]
    assert "root@172.16.42.1" in sent_commands[0][3]


def test_tmux_creates_ssh_window_in_pane_session(monkeypatch):
    client = TmuxClient()
    captured_args = []

    def fake_run(args):
        captured_args.extend(args)
        return "nyx:1.0\n"

    monkeypatch.setattr(client, "_run", fake_run)

    assert client.new_window("nyx:0.0") == "nyx:1.0"
    assert captured_args == [
        "new-window",
        "-P",
        "-F",
        "#{session_name}:#{window_index}.#{pane_index}",
        "-t",
        "nyx:",
    ]


def test_tmux_reports_ssh_connection_errors(monkeypatch):
    client = TmuxClient()
    sent_commands = []

    def fake_run(args):
        sent_commands.append(args)
        if args[0] == "send-keys":
            return ""
        return ""

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(
        client,
        "capture_screen",
        lambda target: "ssh: connect to host 172.16.42.1 port 22: Connection refused",
    )
    monkeypatch.setattr("utils.tmux_client.time.sleep", lambda _: None)

    with pytest.raises(TmuxError, match="SSH connection failed"):
        client.start_ssh("nyx:0.0", SSHConfig(ip="172.16.42.1"), timeout=1, poll_interval=0)


def test_ssh_is_opt_in_and_uses_configured_defaults():
    goal, config = parse_cli_args(["check local status"])
    assert goal == "check local status"
    assert config is None

    with pytest.raises(SystemExit):
        parse_cli_args(["--ssh", "check remote status"])

    goal, config = parse_cli_args(["--ssh", "--ip", "172.16.42.1", "check remote status"])
    assert goal == "check remote status"
    assert config == SSHConfig(ip="172.16.42.1")


def test_ssh_requires_valid_ip_address():
    with pytest.raises(SystemExit):
        parse_cli_args(["--ssh", "--ip", "not-an-ip", "check remote status"])

    with pytest.raises(SystemExit):
        parse_cli_args(["--ip", "172.16.42.1", "check local status"])


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


def test_executor_normalizes_top_and_rejects_interactive_commands():
    assert Executor._normalize_noninteractive_command("top") == ("top -b -n 1", None)
    assert Executor._normalize_noninteractive_command("sudo top") == (
        "sudo top -b -n 1",
        None,
    )
    command, error = Executor._normalize_noninteractive_command("htop")
    assert command == "htop"
    assert error and "top -b -n 1" in error
    assert Executor._normalize_noninteractive_command("tail -f /var/log/syslog")[1] is not None
    assert Executor._normalize_noninteractive_command("journalctl -f")[1] is not None
    assert Executor._normalize_noninteractive_command("ping 8.8.8.8") == (
        "ping -c 1 8.8.8.8",
        None,
    )


def test_high_risk_command_gate_blocks_disk_mutations_but_allows_read_only_listing():
    assert Executor._high_risk_command_reason("sudo parted -l") is None
    assert Executor._high_risk_command_reason("sudo fdisk -l /dev/vda") is None
    assert Executor._high_risk_command_reason("lsblk -o NAME,SIZE,FSTYPE") is None

    dangerous_commands = [
        "sudo parted /dev/vda mklabel gpt",
        "sudo -n parted -s /dev/vda mklabel gpt",
        "mkfs.ext4 /dev/vda1",
        "wipefs -a /dev/vda",
        "dd if=image.iso of=/dev/vda",
        "sudo rm -rf /etc",
        "sudo rm -rf /",
        "sudo chmod -R 777 /",
        "sudo find /etc -delete",
        "echo inspect; sudo sgdisk --zap-all /dev/vda",
        "bash -c 'sudo parted /dev/vda mklabel gpt'",
    ]
    for command in dangerous_commands:
        assert Executor._high_risk_command_reason(command) is not None, command


def test_executor_never_sends_blocked_partition_command_to_tmux(monkeypatch):
    class FakeTmux:
        @staticmethod
        def get_target():
            return "nyx:0.0", True

        @staticmethod
        def run_command(*args, **kwargs):
            pytest.fail("high-risk command must be blocked before reaching tmux")

    executor = Executor.__new__(Executor)
    executor._tmux = FakeTmux()
    executor._last_step_log = []
    executor._ssh_target = None
    executor._remote_context = ""
    executor._control_plane = None
    executor._control_run_id = None
    monkeypatch.setattr(TmuxClient, "is_installed", staticmethod(lambda: True))
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda task, step_log, step: {
            "action": "run_command",
            "command": "sudo parted /dev/vda mklabel gpt",
        },
    )

    result = executor.execute(Task(id="disk-edit", description="Inspect disk layout"))

    assert result["status"] == "failed"
    assert result["safety_blocked"] is True
    assert "blocks partition-editing commands" in result["error"]
    assert executor._last_step_log[0]["safety_blocked"] is True


def _make_executor_for_actions(actions, tmux):
    executor = Executor.__new__(Executor)
    executor._tmux = tmux
    executor._last_step_log = []
    executor._ssh_target = None
    executor._remote_context = ""
    executor._control_plane = None
    executor._control_run_id = None
    action_iter = iter(actions)
    executor._decide_next_action = lambda task, step_log, step: next(action_iter)
    return executor


def test_model_approval_is_not_sent_as_terminal_keystrokes(monkeypatch):
    class FakeTmux:
        def __init__(self):
            self.sent_keys = []

        @staticmethod
        def get_target():
            return "nyx:0.0", True

        def send_raw_keys(self, target, keys):
            self.sent_keys.append((target, keys))

    tmux = FakeTmux()
    executor = _make_executor_for_actions(
        [
            {
                "action": "human_input",
                "kind": "approval",
                "prompt": "Do you want to run fdisk with sudo?",
            },
            {"action": "done", "status": "completed", "result": "Finished safely."},
        ],
        tmux,
    )
    monkeypatch.setattr(TmuxClient, "is_installed", staticmethod(lambda: True))
    monkeypatch.setattr(
        "executor.executor.request_human_input",
        lambda kind, prompt: "yes",
    )

    result = executor.execute(Task(id="approval", description="Inspect disk layout"))

    assert result["status"] == "completed"
    assert tmux.sent_keys == []
    assert executor._last_step_log == [{
        "command": "<human_input approval>",
        "output": "User approved: Do you want to run fdisk with sudo?",
        "exit_code": None,
        "timed_out": False,
    }]


def test_repeated_model_approval_request_stops_without_reprompting(monkeypatch):
    class FakeTmux:
        @staticmethod
        def get_target():
            return "nyx:0.0", True

        @staticmethod
        def send_raw_keys(target, keys):
            pytest.fail("model-requested approval must not be sent to the terminal")

    prompt_calls = []
    monkeypatch.setattr(TmuxClient, "is_installed", staticmethod(lambda: True))
    monkeypatch.setattr(
        "executor.executor.request_human_input",
        lambda kind, prompt: prompt_calls.append(prompt) or "yes",
    )
    executor = _make_executor_for_actions(
        [
            {"action": "human_input", "kind": "approval", "prompt": "Run with sudo?"},
            {"action": "human_input", "kind": "approval", "prompt": "Run with sudo?"},
        ],
        FakeTmux(),
    )

    result = executor.execute(Task(id="repeat-approval", description="Inspect disk"))

    assert result["status"] == "failed"
    assert "requested approval more than once" in result["error"]
    assert prompt_calls == ["Run with sudo?"]


def test_model_requested_credential_is_not_typed_into_terminal(monkeypatch):
    class FakeTmux:
        @staticmethod
        def get_target():
            return "nyx:0.0", True

        @staticmethod
        def send_raw_keys(target, keys):
            pytest.fail("model-requested credentials must not be typed into the terminal")

    monkeypatch.setattr(TmuxClient, "is_installed", staticmethod(lambda: True))
    executor = _make_executor_for_actions(
        [{"action": "human_input", "kind": "credential", "prompt": "Enter password"}],
        FakeTmux(),
    )

    result = executor.execute(Task(id="credential", description="Inspect service"))

    assert result["status"] == "failed"
    assert "will not type model-requested credentials" in result["error"]


def test_orchestrator_stops_run_after_high_risk_command_is_blocked(tmp_path, monkeypatch):
    store = TaskStore(str(tmp_path / "tasks.db"))
    control = ControlPlane(str(tmp_path / "control.db"))
    control.start_run("safety-run", "Inspect disk layout")
    monkeypatch.setattr(settings, "HTML_REPORT_PATH", str(tmp_path / "report.html"))

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._goal = "Inspect disk layout"
    orchestrator._iteration = 0
    orchestrator._remote_context = ""
    orchestrator._recent_terminal_context = []
    orchestrator._safety_halt_reason = ""
    orchestrator._start_time = 0.0
    orchestrator._run_id = "safety-run"
    orchestrator._control = control
    orchestrator._manager = TaskManager(store)
    orchestrator._store = store
    orchestrator._history = SimpleNamespace(as_prompt_text=lambda limit: "")
    orchestrator._executor = SimpleNamespace(
        execute=lambda task: {
            "status": "failed",
            "result": "",
            "error": "Partition editing is blocked.",
            "safety_blocked": True,
        },
        _last_step_log=[{
            "command": "sudo parted /dev/vda mklabel gpt",
            "output": "Partition editing is blocked.",
            "exit_code": None,
            "timed_out": False,
        }],
    )
    plan_calls = []

    def create_dangerous_task(**kwargs):
        plan_calls.append(kwargs)
        return {
            "action": "create_tasks",
            "tasks": [{"id": "danger", "description": "Inspect disk layout"}],
        }

    orchestrator._planner = SimpleNamespace(plan=create_dangerous_task)
    finishes = []
    monkeypatch.setattr(orchestrator, "_finish", lambda **kwargs: finishes.append(kwargs))

    orchestrator._loop()

    task = store.get_task("danger")
    assert task is not None
    assert task.status.value == "failed"
    assert task.retries == 0
    assert len(plan_calls) == 1
    assert finishes == [{
        "success": False,
        "message": (
            "Nyx blocked or canceled an unsafe operation and stopped the run without retrying it. "
            "Review the goal and terminal before starting another run."
        ),
    }]


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
