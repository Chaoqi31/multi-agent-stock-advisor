"""Cross-run advisory memory.

Stores a short situation summary plus the recommendation for each completed analysis, and
retrieves the most similar past situations for the same ticker so later runs can be calibrated
against earlier calls and their realized outcomes.

Backed by a local ChromaDB collection with its default (on-device) embedding function, so it
runs on a laptop with no GPU. If ChromaDB is unavailable or disabled via ENABLE_MEMORY=false,
every method degrades to a no-op and the rest of the pipeline is unaffected.
"""

import os
from pathlib import Path
from typing import List, Optional

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# <agent-root>/.chroma  (gitignored)
_DEFAULT_DIR = Path(__file__).resolve().parents[2] / ".chroma"
_COLLECTION = "advisory_situations"


def _memory_enabled() -> bool:
    return os.getenv("ENABLE_MEMORY", "true").strip().lower() in ("1", "true", "yes")


class AdvisoryMemory:
    """Thin wrapper over a persistent ChromaDB collection of past analysis situations."""

    def __init__(self, persist_dir: Optional[str] = None, collection: str = _COLLECTION):
        self._collection = None
        if not _memory_enabled():
            logger.info("Advisory memory disabled (ENABLE_MEMORY=false).")
            return
        try:
            import chromadb

            persist_dir = persist_dir or os.getenv("MEMORY_DIR", str(_DEFAULT_DIR))
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=persist_dir)
            self._collection = client.get_or_create_collection(name=collection)
            logger.info(f"Advisory memory ready ({self._collection.count()} stored situations).")
        except Exception as exc:  # noqa: BLE001 - memory is best-effort
            logger.warning(f"Advisory memory unavailable, continuing without it: {exc}")
            self._collection = None

    @property
    def available(self) -> bool:
        return self._collection is not None

    def store(self, thread_id: str, ticker: str, situation: str, recommendation: str, date: str) -> None:
        """Persist (or overwrite) the situation for a run, keyed by thread_id."""
        if not self._collection or not thread_id:
            return
        try:
            document = situation.strip() or f"{ticker} analysis on {date}"
            self._collection.upsert(
                ids=[thread_id],
                documents=[document[:4000]],
                metadatas=[
                    {
                        "thread_id": thread_id,
                        "ticker": (ticker or "").upper(),
                        "date": date,
                        "recommendation": recommendation[:1000],
                        "outcome": "",
                        "return_metrics": "",
                    }
                ],
            )
            logger.info(f"Stored advisory memory for {ticker} ({thread_id}).")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to store advisory memory: {exc}")

    def retrieve(self, ticker: str, query: str, k: int = 3) -> str:
        """Return a formatted block of the k most similar past situations for this ticker.

        Empty string when memory is unavailable or there is no prior history (e.g. first run).
        """
        if not self._collection:
            return ""
        try:
            result = self._collection.query(
                query_texts=[query or ticker],
                n_results=max(1, k),
                where={"ticker": (ticker or "").upper()},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Advisory memory retrieval failed: {exc}")
            return ""

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        if not docs:
            return ""

        lines: List[str] = []
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            entry = f"- 日期 {meta.get('date', '未知')}：{(doc or '')[:300]}"
            rec = meta.get("recommendation")
            if rec:
                entry += f"\n  当时建议：{rec[:200]}"
            outcome = meta.get("outcome")
            if outcome:
                entry += f"\n  实际结果：{outcome[:200]}"
            lines.append(entry)
        return "\n".join(lines)

    def update_with_outcome(self, thread_id: str, outcome: str, return_metrics: str = "") -> bool:
        """Backfill the realized outcome for a stored situation. Returns True on success."""
        if not self._collection or not thread_id:
            return False
        try:
            existing = self._collection.get(ids=[thread_id])
            metas = existing.get("metadatas") or []
            if not metas:
                logger.warning(f"No stored situation for thread_id={thread_id}; cannot reflect.")
                return False
            meta = dict(metas[0] or {})
            meta["outcome"] = outcome[:1000]
            meta["return_metrics"] = return_metrics[:500]
            self._collection.update(ids=[thread_id], metadatas=[meta])
            logger.info(f"Updated advisory memory outcome for {thread_id}.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to update advisory memory outcome: {exc}")
            return False
