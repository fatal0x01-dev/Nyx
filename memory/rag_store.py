"""
RAGStore — vector memory layer (Phase 3).

In Phase 1 this module is a stub that logs a warning if you try to use it.
The interface is defined now so the orchestrator can call rag_store.query(...)
without breaking when RAG is not configured.
"""

from __future__ import annotations

from utils.logger import get_logger

log = get_logger(__name__)


class RAGStore:
    def __init__(self, store_dir: str | None = None):
        self._enabled = False
        log.info("RAGStore: disabled (Phase 1 stub). Enable in Phase 3.")

    def add(self, text: str, metadata: dict | None = None):
        if not self._enabled:
            return
        raise NotImplementedError("RAG not yet implemented")

    def query(self, query: str, top_k: int = 5) -> list[str]:
        if not self._enabled:
            return []
        raise NotImplementedError("RAG not yet implemented")