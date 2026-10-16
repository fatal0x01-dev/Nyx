"""
Planner — sends goal + history to the planning model on the laptop
and returns a structured action (create_tasks | goal_complete | goal_failed).
"""

from __future__ import annotations

from typing import Optional

import httpx

import config.settings as cfg
from planner.prompts import SYSTEM_PROMPT, VALIDATION_SYSTEM_PROMPT, build_plan_prompt, build_validation_prompt
from tasks.task import Task
from utils.logger import get_logger, get_reasoning_logger
from utils.helpers import parse_json_response, new_id, utcnow

log = get_logger(__name__)
reasoning = get_reasoning_logger(__name__)


class Planner:
    def __init__(self):
        self._base_url = cfg.PLANNER_API_URL.rstrip("/")
        self._model = cfg.PLANNER_MODEL
        log.info("Planner ready — %s  model=%s", self._base_url, self._model)

    # ── Public API ────────────────────────────────────────────────────────

    def plan(
        self,
        goal: str,
        history_text: str,
        pending_tasks: list[Task],
        remote_context: str = "",
        guidance_text: str = "",
        terminal_context: str = "",
    ) -> dict:
        """
        Ask the planner what to do next.

        Returns one of:
          {"action": "create_tasks",  "tasks": [...]}
          {"action": "goal_complete", "summary": "..."}
          {"action": "goal_failed",   "reason":  "..."}
        """
        pending_dicts = [
            {"id": t.id, "description": t.description, "priority": t.priority}
            for t in pending_tasks
        ]
        user_prompt = build_plan_prompt(
            goal,
            history_text,
            pending_dicts,
            remote_context,
            guidance_text,
            terminal_context,
        )

        log.debug("Planner prompt (user):\n%s", user_prompt[:600])

        raw = self._call_lmstudio(user_prompt)
        log.debug("Planner raw response:\n%s", raw[:600])

        result = parse_json_response(raw)
        if result is None:
            log.error("Planner returned unparseable response: %s", raw[:300])
            return {
                "action": "goal_failed",
                "reason": f"Planner returned non-JSON response: {raw[:200]}",
            }

        # Normalise tasks — ensure each has an id
        if result.get("action") == "create_tasks":
            for task in result.get("tasks", []):
                if not task.get("id"):
                    task["id"] = new_id()

        log.info("Planner action: %s", result.get("action"))
        reasoning.info("Planner action: %s", result.get("action"))
        if result.get("action") == "create_tasks":
            reasoning.info("Planned %d next task(s).", len(result.get("tasks", [])))
        elif result.get("action") == "goal_complete":
            reasoning.info("Planner believes the goal is complete.")
        elif result.get("action") == "goal_failed":
            reasoning.warning("Planner believes the goal is not achievable: %s", result.get("reason", "unknown reason"))
        return result

    def validate_task_result(
        self,
        goal: str,
        task_description: str,
        result: str,
        error: str,
        history_text: str,
        remote_context: str = "",
        guidance_text: str = "",
        terminal_context: str = "",
    ) -> dict:
        """Return {action: accepted|rejected, ...} after validating a task result."""
        user_prompt = build_validation_prompt(
            goal=goal,
            task_description=task_description,
            result=result,
            error=error,
            history_text=history_text,
            remote_context=remote_context,
            guidance_text=guidance_text,
            terminal_context=terminal_context,
        )
        raw = self._call_lmstudio_for_validation(user_prompt)
        result_json = parse_json_response(raw)
        if result_json is None or result_json.get("action") not in {"accepted", "rejected"}:
            log.warning(
                "Planner validation returned an invalid response; requesting one strict-format retry."
            )
            raw = self._call_lmstudio_for_validation(
                user_prompt
                + "\nYour previous response was invalid. Return ONLY one JSON object "
                'with action exactly "accepted" or "rejected".'
            )
            result_json = parse_json_response(raw)
        if result_json is None:
            return {
                "action": "rejected",
                "reason": f"Planner validation returned non-JSON output after retry: {raw[:200]}",
            }
        if result_json.get("action") not in {"accepted", "rejected"}:
            return {
                "action": "rejected",
                "reason": "Planner validation returned an invalid action after retry.",
            }
        return result_json

    # ── LM Studio client ──────────────────────────────────────────────────

    def _call_lmstudio(self, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": cfg.PLANNER_TEMPERATURE,
            "max_tokens": cfg.MAX_TOKENS,
        }

        try:
            with httpx.Client(timeout=cfg.REQUEST_TIMEOUT) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Planner request timed out after {cfg.REQUEST_TIMEOUT}s"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Planner HTTP error {e.response.status_code}: {e.response.text[:200]}"
            )
        except Exception as e:
            raise RuntimeError(f"Planner call failed: {e}") from e

    def _call_lmstudio_for_validation(self, user_prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": cfg.PLANNER_TEMPERATURE,
            "max_tokens": cfg.MAX_TOKENS,
        }

        try:
            with httpx.Client(timeout=cfg.REQUEST_TIMEOUT) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Planner validation request timed out after {cfg.REQUEST_TIMEOUT}s"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Planner validation HTTP error {e.response.status_code}: {e.response.text[:200]}"
            )
        except Exception as e:
            raise RuntimeError(f"Planner validation call failed: {e}") from e