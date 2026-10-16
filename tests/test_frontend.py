import asyncio

from textual.widgets import Button, Input

from frontend import NyxDashboard
from utils.control_plane import ControlPlane


def test_dashboard_controls_pause_and_submits_guidance(tmp_path):
    database_path = str(tmp_path / "control.db")
    control = ControlPlane(database_path)
    control.start_run("dashboard-test", "Inspect the remote host")

    async def exercise_dashboard():
        app = NyxDashboard(database_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.click("#pause")
            assert control.snapshot("dashboard-test")["paused"] is True

            suggestion = app.query_one("#suggestion-input", Input)
            suggestion.value = "Prefer read-only checks."
            await pilot.click("#send-guidance")
            assert control.guidance_text("dashboard-test") == "- Prefer read-only checks."

            await pilot.click("#resume")
            assert control.snapshot("dashboard-test")["paused"] is False
            assert app.query_one("#resume", Button).disabled is True

    asyncio.run(exercise_dashboard())
