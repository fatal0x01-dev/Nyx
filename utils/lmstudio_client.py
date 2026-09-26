"""
LMStudioClient — HTTP client for LM Studio.

Supports:

1. Standard OpenAI-compatible chat completions
   Used by the Planner.

2. MCP-enabled /api/v1/chat requests
   Used by the Executor when tools are required.
"""

from __future__ import annotations

import httpx

import config.settings as cfg
from utils.logger import get_logger

log = get_logger(__name__)


class LMStudioClient:
    def __init__(self, base_url: str, model: str, temperature: float):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._provider = self._detect_provider(self._base_url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_provider(base_url: str) -> str:
        lowered = base_url.lower()
        if "11434" in lowered or "ollama" in lowered:
            return "ollama"
        return "lmstudio"

    def _health_check_url(self) -> str:
        if self._provider == "ollama":
            return f"{self._base_url}/api/tags"
        return f"{self._base_url}/models"

    def _headers(self) -> dict:
        """
        Build request headers.

        LM Studio MCP API may require a bearer token depending on
        server settings.
        """

        headers = {
            "Content-Type": "application/json",
        }

        token = getattr(cfg, "LMSTUDIO_API_TOKEN", "")

        if token:
            headers["Authorization"] = f"Bearer {token}"

        return headers

    # ------------------------------------------------------------------
    # Standard chat completion (Planner)
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict]) -> str:
        """
        Standard OpenAI-compatible endpoint.

        Returns assistant text only.
        """

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": cfg.MAX_TOKENS,
        }

        log.debug(
            "LMStudio Chat -> model=%s messages=%d",
            self._model,
            len(messages),
        )

        try:
            with httpx.Client(timeout=cfg.REQUEST_TIMEOUT) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

                return data["choices"][0]["message"]["content"]

        except httpx.TimeoutException:
            raise RuntimeError(
                f"LMStudio request timed out after "
                f"{cfg.REQUEST_TIMEOUT}s"
            )

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"LMStudio HTTP {e.response.status_code}: "
                f"{e.response.text[:500]}"
            )

        except Exception as e:
            raise RuntimeError(
                f"LMStudio call failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # MCP-enabled chat (Executor)
    # ------------------------------------------------------------------

    def chat_mcp(
        self,
        prompt: str,
        integrations: list[dict],
    ) -> dict:
        """
        LM Studio MCP endpoint.

        Example:

            integrations=[
                {
                    "type": "plugin",
                    "id": "mcp/mcp-docker"
                }
            ]

        Returns the full LM Studio MCP response.
        """

        payload = {
            "model": self._model,
            "input": prompt,
            "integrations": integrations,
            "temperature": self._temperature,
            "context_length": 32768,
        }

        log.info(
            "LMStudio MCP request -> model=%s integrations=%s",
            self._model,
            integrations,
        )

        try:
            with httpx.Client(timeout=cfg.REQUEST_TIMEOUT) as client:
                response = client.post(
                    f"{self._base_url}/api/v1/chat",
                    headers=self._headers(),
                    json=payload,
                )

                response.raise_for_status()

                data = response.json()

                log.debug(
                    "LMStudio MCP response received"
                )

                return data

        except httpx.TimeoutException:
            raise RuntimeError(
                f"MCP request timed out after "
                f"{cfg.REQUEST_TIMEOUT}s"
            )

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"MCP HTTP {e.response.status_code}: "
                f"{e.response.text[:1000]}"
            )

        except Exception as e:
            raise RuntimeError(
                f"MCP call failed: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Return True if LM Studio is reachable.
        """

        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(
                    self._health_check_url(),
                    headers=self._headers(),
                )

                return response.status_code == 200

        except Exception:
            return False