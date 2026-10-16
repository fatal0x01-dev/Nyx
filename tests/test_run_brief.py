"""Regression tests for cross-role awareness (run brief) and the retry loop.

These encode the failure observed in run aa3430bc: a Docker + Apache login-page
goal where tasks that had actually succeeded were marked failed, the Dockerfile
was written before the file it copies, and a changed host port never reached
the tasks that depended on it.
"""

from types import SimpleNamespace

from config import settings
from executor.executor import Executor
from executor.shell_prompts import SYSTEM_PROMPT as EXECUTOR_SYSTEM_PROMPT, build_step_prompt
from memory.task_store import TaskStore
from orchestrator import Orchestrator
from planner.prompts import VALIDATION_SYSTEM_PROMPT, build_validation_prompt
from tasks.run_brief import (
    build_run_brief,
    collect_facts,
    extract_facts,
    order_app_files_before_dockerfile,
)
from tasks.task import Task, TaskStatus
from tasks.task_manager import TaskManager
from utils.control_plane import ControlPlane

GOAL = "run apache in docker and serve a static login form"
BUILD_TASK = "Build a Docker image named 'apache-web-server' from the Dockerfile in /root/web-server"
RUN_TASK = "Run a Docker container named 'apache-container' using the 'apache-web-server' image, mapping port 8080 to the container's port 80"

# Step logs transcribed from the real run's stored terminal evidence.
BUILD_STEPS = [
    {"command": "<read_file /root/web-server/Dockerfile>", "exit_code": 0,
     "output": "FROM httpd:latest\nCOPY ./html /usr/local/apache2/htdocs/\nEXPOSE 80"},
    {"command": "docker build -t apache-web-server /root/web-server", "exit_code": 1,
     "output": "=> ERROR [2/2] COPY ./html\nERROR: failed to build: \"/html\": not found"},
    {"command": "<human_input approval>", "exit_code": None,
     "output": "User approved: The Docker build failed because './html' does not exist"},
    {"command": "mkdir /root/web-server/html", "exit_code": 0, "output": ""},
    {"command": "docker build -t apache-web-server /root/web-server", "exit_code": 0,
     "output": "[+] Building 1.0s (7/7) FINISHED\n => naming to docker.io/library/apache-web-server"},
]
RUN_STEPS = [
    {"command": "docker run -d --name apache-container -p 8080:80 apache-web-server", "exit_code": 125,
     "output": "docker: Error response from daemon: Conflict. name already in use"},
    {"command": "docker rm apache-container", "exit_code": 0, "output": "apache-container"},
    {"command": "docker run -d --name apache-container -p 8080:80 apache-web-server", "exit_code": 125,
     "output": "docker: failed to set up container networking: address already in use"},
    {"command": "<human_input approval>", "exit_code": None, "output": "User approved: port 8080 is in use."},
    {"command": "docker run -d --name apache-container -p 8082:80 apache-web-server", "exit_code": 0,
     "output": "22bf1834b61ac165aeac889d1cda08df215d1329951b01822b593f6c7451273f"},
]


def _accepted(steps, description):
    return Executor.validate_completion_status(
        status="completed",
        result="Completed the requested task.",
        error="",
        last_exit_code=None,
        step_log=steps,
        task_description=description,
    )


# ── Evidence check: recovery is not failure ──────────────────────────────────


def test_recovered_build_failure_is_accepted():
    assert _accepted(BUILD_STEPS, BUILD_TASK) is True


def test_recovered_port_and_name_conflicts_are_accepted():
    assert _accepted(RUN_STEPS, RUN_TASK) is True


def test_unrecovered_failure_is_still_rejected_even_without_marker_words():
    steps = [{"command": "docker run -d --name c -p 8080:80 img", "exit_code": 125,
              "output": "docker: Error response from daemon: address already in use"}]
    assert _accepted(steps, RUN_TASK) is False


def test_failure_followed_by_unrelated_success_is_not_treated_as_recovered():
    steps = [
        {"command": "curl -sI http://localhost:8082", "exit_code": 7,
         "output": "curl: (7) Failed to connect to localhost port 8082"},
        {"command": "docker ps", "exit_code": 0, "output": "CONTAINER ID   IMAGE\nabc   apache-web-server"},
    ]
    assert _accepted(steps, "Verify the container serves the page") is False


def test_nonzero_exit_without_any_marker_is_rejected():
    steps = [{"command": "systemctl is-active foo", "exit_code": 3, "output": "inactive"}]
    assert _accepted(steps, "Check foo is active") is False


# ── Fact extraction ──────────────────────────────────────────────────────────


def test_facts_capture_the_port_the_container_actually_got():
    facts = {f["key"]: f["text"] for f in extract_facts(RUN_STEPS)}
    text = facts["container:apache-container"]
    assert "host port 8082 -> container port 80" in text
    assert "8080" not in text


def test_facts_capture_image_files_and_directories():
    steps = BUILD_STEPS + [
        {"command": "mkdir -p /root/web-server", "exit_code": 0, "output": ""},
        {"command": "<verify_file /root/web-server/Dockerfile>", "exit_code": 0,
         "file_operation": "verify", "file_path": "/root/web-server/Dockerfile",
         "output": "ok", "verified_content": "FROM httpd:latest\nCOPY index.html /x/\n"},
    ]
    facts = {f["key"]: f["text"] for f in extract_facts(steps)}
    assert "docker image built: apache-web-server (build context: /root/web-server)" == facts["image:apache-web-server"]
    assert "directory exists: /root/web-server" == facts["dir:/root/web-server"]
    assert "COPY index.html" in facts["file:/root/web-server/Dockerfile"]


def test_failed_steps_never_become_facts():
    steps = [{"command": "docker run -d --name c -p 8080:80 img", "exit_code": 125, "output": "boom"}]
    assert extract_facts(steps) == []


def test_later_fact_replaces_earlier_one_for_same_container():
    tasks = []
    for index, port in enumerate((8081, 8082)):
        task = Task(id=f"r{index}", description="run", status=TaskStatus.COMPLETED,
                    created_at=f"2026-01-01T00:00:0{index}", completed_at=f"2026-01-01T00:01:0{index}")
        task.metadata["facts"] = extract_facts([
            {"command": f"docker run -d --name web -p {port}:80 img", "exit_code": 0, "output": "id"}
        ])
        tasks.append(task)
    merged = collect_facts(tasks)
    assert len(merged) == 1 and "host port 8082" in merged[0]


# ── Dependency ordering ──────────────────────────────────────────────────────


def _planned(*descriptions):
    return [{"id": f"t{i}", "description": d, "priority": i + 1} for i, d in enumerate(descriptions)]


def test_html_is_moved_ahead_of_the_dockerfile_that_copies_it():
    raw = _planned(
        "Create a new directory for the web server project at /root/web-server",
        "Write a Dockerfile in /root/web-server to configure an Apache web server",
        "Write a static HTML file with a simple login form in /root/web-server",
        BUILD_TASK,
    )
    ordered = order_app_files_before_dockerfile(raw)
    assert [t["id"] for t in ordered] == ["t0", "t2", "t1", "t3"]
    assert [t["priority"] for t in ordered] == [1, 2, 3, 4]


def test_correct_order_is_left_untouched():
    raw = _planned(
        "Create the directory /srv/app",
        "Write a static HTML file in /srv/app",
        "Write a Dockerfile in /srv/app",
    )
    assert order_app_files_before_dockerfile(raw) is raw


def test_directory_and_build_tasks_are_not_misclassified():
    raw = _planned(
        "Write a Dockerfile in /srv/app",
        "Create a new directory for the project at /srv/app",
        BUILD_TASK,
    )
    assert order_app_files_before_dockerfile(raw) is raw


# ── Brief rendering ──────────────────────────────────────────────────────────


def _store_with_plan(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    manager = TaskManager(store)
    tasks = {}
    for index, (tid, desc) in enumerate(
        [("write_html", "Write index.html in /srv/app"), ("run", RUN_TASK), ("verify", "Verify the service on port 8080")]
    ):
        task = Task(id=tid, description=desc, priority=index + 1, created_at=f"2026-01-01T00:00:0{index}")
        manager.add_task(task)
        tasks[tid] = task
    return store, manager, tasks


def test_brief_marks_current_task_and_carries_facts_and_outcomes(tmp_path):
    store, manager, tasks = _store_with_plan(tmp_path)
    tasks["run"].metadata["facts"] = extract_facts(RUN_STEPS)
    manager.mark_completed(tasks["run"], "container started")
    manager.mark_running(tasks["verify"])

    brief = build_run_brief(GOAL, store.get_all_tasks(), current_task=tasks["verify"])

    assert "[CURRENT TASK] Verify the service on port 8080" in brief
    assert "[pending] Write index.html" in brief
    assert "outcome: container started" in brief
    assert "host port 8082" in brief


def test_validator_brief_hides_sibling_outcomes_but_keeps_facts(tmp_path):
    store, manager, tasks = _store_with_plan(tmp_path)
    tasks["run"].metadata["facts"] = extract_facts(RUN_STEPS)
    manager.mark_completed(tasks["run"], "SECRET-SIBLING-OUTCOME")
    brief = build_run_brief(GOAL, store.get_all_tasks(), current_task=tasks["verify"], include_outcomes=False)
    assert "SECRET-SIBLING-OUTCOME" not in brief
    assert "host port 8082" in brief


def test_retry_note_is_included_only_for_retried_tasks(tmp_path):
    store, manager, tasks = _store_with_plan(tmp_path)
    fresh = build_run_brief(GOAL, store.get_all_tasks(), current_task=tasks["run"])
    assert "PREVIOUS ATTEMPT" not in fresh
    manager.mark_retry(tasks["run"], "Planner rejected task completion: port mismatch")
    retried = build_run_brief(GOAL, store.get_all_tasks(), current_task=tasks["run"])
    assert "PREVIOUS ATTEMPT" in retried and "port mismatch" in retried


def test_brief_stays_within_budget_and_keeps_facts_when_plan_is_huge():
    tasks = []
    for index in range(80):
        tasks.append(Task(id=f"t{index}", description="do a thing " * 20, priority=index,
                          created_at=f"2026-01-01T00:00:{index:02d}"))
    done = Task(id="done", description="run", status=TaskStatus.COMPLETED, priority=999)
    done.metadata["facts"] = extract_facts(RUN_STEPS)
    brief = build_run_brief(GOAL, tasks + [done], current_task=tasks[0])
    assert len(brief) < 6000
    assert "host port 8082" in brief


# ── Prompts carry the brief ──────────────────────────────────────────────────


def test_executor_prompt_includes_brief_and_instructs_using_it():
    prompt = build_step_prompt(
        task_id="t", task_description="Write a Dockerfile", step_log=[], steps_taken=0,
        max_steps=12, run_brief="FULL PLAN: 2. write index.html",
    )
    assert "RUN BRIEF:\nFULL PLAN: 2. write index.html" in prompt
    assert "RUN BRIEF" in EXECUTOR_SYSTEM_PROMPT
    assert "ESTABLISHED FACTS are authoritative" in EXECUTOR_SYSTEM_PROMPT


def test_validation_prompt_scopes_the_task_and_accepts_recovery():
    prompt = build_validation_prompt(
        goal=GOAL, task_description="Write a Dockerfile", result="ok", error="",
        history_text="", run_brief="FULL PLAN ...",
    )
    assert "RUN BRIEF (scope only, not evidence):\nFULL PLAN ..." in prompt
    rules = " ".join(VALIDATION_SYSTEM_PROMPT.split())
    assert "is NOT missing from this task" in rules
    assert "later successful run of the same command" in rules


# ── Orchestrator wiring ──────────────────────────────────────────────────────


def _bare_orchestrator(tmp_path, monkeypatch, step_log):
    monkeypatch.setattr(settings, "HTML_REPORT_PATH", str(tmp_path / "report.html"))
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._goal = GOAL
    orchestrator._iteration = 1
    orchestrator._run_id = "run"
    orchestrator._recent_terminal_context = []
    orchestrator._safety_halt_reason = ""
    orchestrator._remote_context = ""
    orchestrator._control = ControlPlane(str(tmp_path / "control.db"))
    orchestrator._control.start_run("run", GOAL)
    orchestrator._store = TaskStore(str(tmp_path / "tasks.db"))
    orchestrator._manager = TaskManager(orchestrator._store)
    orchestrator._executor = SimpleNamespace(
        _last_step_log=step_log,
        execute=lambda task: {"status": "completed", "result": "Container started on 8082."},
        validate_completion_status=Executor.validate_completion_status,
    )
    seen = {}

    def validate(**kwargs):
        seen.update(kwargs)
        return {"action": "accepted", "summary": "ok"}

    orchestrator._planner = SimpleNamespace(validate_task_result=validate)
    monkeypatch.setattr(orchestrator, "_publish_report", lambda: None)
    return orchestrator, seen


def test_recovered_task_completes_on_first_attempt_and_persists_its_own_facts(tmp_path, monkeypatch):
    orchestrator, seen = _bare_orchestrator(tmp_path, monkeypatch, list(RUN_STEPS))
    task = Task(id="run", description=RUN_TASK, priority=1)
    orchestrator._manager.add_task(task)

    orchestrator._run_task(task)

    stored = orchestrator._store.get_task("run")
    assert stored.status == TaskStatus.COMPLETED
    assert stored.retries == 0  # previously: retried 3x, deleting its own container each time
    assert "host port 8082" in stored.metadata["facts"][0]["text"]
    # evidence stored with the task is THIS attempt's, not the previous one's
    assert "TASK OUTCOME run: status=completed" in stored.metadata["terminal_evidence"]
    # the validator was given the brief for scope, separate from terminal evidence
    assert "[CURRENT TASK]" in seen["run_brief"]
    assert "FULL PLAN" not in seen["terminal_context"]  # brief never leaks into evidence


def test_orchestrator_enqueues_html_before_dockerfile_and_brief_reaches_executor(tmp_path, monkeypatch):
    orchestrator, _ = _bare_orchestrator(tmp_path, monkeypatch, [])
    orchestrator._enqueue_planner_tasks(_planned(
        "Create the project directory /srv/app",
        "Write a Dockerfile in /srv/app for Apache",
        "Write a static HTML file with a login form in /srv/app",
    ))
    order = [t.description for t in orchestrator._manager.queue_snapshot()]
    assert order[1].startswith("Write a static HTML")
    assert order[2].startswith("Write a Dockerfile")

    first = orchestrator._manager.next_task()
    brief = orchestrator._run_brief_text(first)
    assert "[CURRENT TASK] Create the project directory" in brief
    assert "[pending] Write a static HTML file" in brief


def test_planner_context_contains_established_facts(tmp_path, monkeypatch):
    orchestrator, _ = _bare_orchestrator(tmp_path, monkeypatch, [])
    orchestrator._system_context = {"user": "root"}
    task = Task(id="run", description=RUN_TASK, priority=1)
    orchestrator._manager.add_task(task)
    task.metadata["facts"] = extract_facts(RUN_STEPS)
    orchestrator._manager.mark_completed(task, "started")
    orchestrator._recent_terminal_context = ["x" * 20000]

    text = orchestrator._terminal_context_text()

    assert "ESTABLISHED FACTS" in text and "host port 8082" in text  # survives trimming
    assert text.startswith("CURRENT SYSTEM CONTEXT:")
    assert len(text) <= 12000
