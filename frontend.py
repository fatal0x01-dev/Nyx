"""Separate-terminal Textual dashboard for controlling a Nyx run."""

from __future__ import annotations

import argparse
import sqlite3

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Static

import config.settings as cfg
from utils.control_plane import ControlPlane


class NyxDashboard(App[None]):
    TITLE = "NYX CONTROL"
    SUB_TITLE = "Orchestrator dashboard"
    CSS = """
    Screen {
        background: #0b1120;
        color: #e5eaf4;
    }
    #body {
        height: 1fr;
        padding: 0 1;
    }
    .panel {
        border: round #334155;
        background: #111a2e;
        padding: 0 1;
    }
    #overview {
        height: 7;
    }
    #guidance {
        height: 1fr;
        min-height: 4;
    }
    #suggestions {
        height: 4;
    }
    #controls {
        height: 3;
        align: left middle;
    }
    #controls Button {
        min-width: 12;
    }
    #suggestion-input {
        width: 1fr;
        margin-right: 1;
    }
    .heading {
        color: #60a5fa;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit dashboard"),
        ("ctrl+p", "toggle_pause", "Pause / resume"),
    ]

    def __init__(self, database_path: str = cfg.CONTROL_DB_PATH) -> None:
        super().__init__()
        self.control = ControlPlane(database_path)
        self.run_id: str | None = None
        self.run_active = False
        self.run_paused = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            with Horizontal(classes="panel", id="overview"):
                yield Static("Waiting for an orchestrator run…", id="run-status", markup=False)
            with Horizontal(classes="panel", id="guidance"):
                yield Static("", id="run-guidance", markup=False)
            with Horizontal(classes="panel", id="suggestions"):
                yield Static("", id="suggestion-history", markup=False)
            with Horizontal(id="controls"):
                yield Button("Pause run", id="pause", variant="warning", disabled=True)
                yield Button("Resume run", id="resume", variant="success", disabled=True)
                yield Input(
                    placeholder="Suggest guidance for the AI…",
                    id="suggestion-input",
                    disabled=True,
                )
                yield Button("Send guidance", id="send-guidance", variant="primary", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.5, self.refresh_status)
        self.refresh_status()

    def refresh_status(self) -> None:
        state = self.control.snapshot()
        self.run_id = state["run_id"] if state else None
        self.run_active = bool(state and state["status"] in {"running", "paused"})
        self.run_paused = bool(state and state["paused"])

        status_widget = self.query_one("#run-status", Static)
        guidance_widget = self.query_one("#run-guidance", Static)
        history_widget = self.query_one("#suggestion-history", Static)
        pause_button = self.query_one("#pause", Button)
        resume_button = self.query_one("#resume", Button)
        suggestion_input = self.query_one("#suggestion-input", Input)
        send_button = self.query_one("#send-guidance", Button)

        if state is None:
            status_widget.update("Waiting for an orchestrator run…\n\nStart Nyx in its usual terminal.")
            guidance_widget.update("GUIDANCE\n\nNo active run yet.")
            history_widget.update("RECENT SUGGESTIONS\n\nSuggestions you submit will appear here.")
        else:
            status_label = "PAUSED" if state["paused"] else state["status"].upper()
            status_widget.update(
                f"NYX RUN  {state['run_id']}\n"
                f"Status: {status_label}  •  Phase: {state['phase']}\n"
                f"Iteration: {state['iteration']}\n"
                f"Goal: {state['goal']}\n"
                f"Task: {state['current_task'] or '—'}"
            )
            guidance_widget.update(
                "AI GUIDANCE\n\n"
                + (
                    "\n".join(f"• {item['content']}" for item in state["guidance"])
                    if state["guidance"]
                    else "No suggestions added. Guidance applies to future AI decisions in this run."
                )
            )
            history_widget.update(
                "RECENT SUGGESTIONS\n\n"
                + (
                    "\n".join(f"• {item['content']}" for item in state["guidance"][-5:])
                    if state["guidance"]
                    else "None yet"
                )
            )

        pause_button.disabled = not self.run_active or self.run_paused
        resume_button.disabled = not self.run_active or not self.run_paused
        suggestion_input.disabled = not self.run_active
        send_button.disabled = not self.run_active

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self.run_active or self.run_id is None:
            return
        try:
            if event.button.id == "pause":
                self.control.set_paused(self.run_id, True)
                self.notify("Paused at the next safe checkpoint.", title="Nyx paused")
            elif event.button.id == "resume":
                self.control.set_paused(self.run_id, False)
                self.notify("Nyx will continue.", title="Nyx resumed")
            elif event.button.id == "send-guidance":
                self.submit_guidance()
        except ValueError as error:
            self.notify(str(error), title="Run is no longer active", severity="warning")
        self.refresh_status()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "suggestion-input":
            self.submit_guidance()

    def submit_guidance(self) -> None:
        if not self.run_active or self.run_id is None:
            self.notify("There is no active orchestrator run.", severity="error")
            return
        suggestion_input = self.query_one("#suggestion-input", Input)
        content = suggestion_input.value.strip()
        if not content:
            self.notify("Enter a suggestion before sending it.", severity="warning")
            return
        try:
            self.control.add_guidance(self.run_id, content)
        except (ValueError, sqlite3.Error) as error:
            self.notify(str(error), title="Could not send guidance", severity="error")
            return
        suggestion_input.value = ""
        self.notify("Added to future AI decisions.", title="Guidance sent")
        self.refresh_status()

    def action_toggle_pause(self) -> None:
        if not self.run_active or self.run_id is None:
            return
        try:
            self.control.set_paused(self.run_id, not self.run_paused)
        except ValueError as error:
            self.notify(str(error), title="Run is no longer active", severity="warning")
        self.refresh_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the Nyx orchestrator control dashboard.")
    parser.add_argument(
        "--control-db",
        default=cfg.CONTROL_DB_PATH,
        help="Path to the shared orchestrator control database.",
    )
    args = parser.parse_args()
    NyxDashboard(args.control_db).run()


if __name__ == "__main__":
    main()
