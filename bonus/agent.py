"""bonus/agent.py — HybridMemoryAgent: episodic memory (Qdrant) + user profile (Feast).

Deliverable for the Bonus Challenge — Day 19, Track 2.

Run:
    python bonus/demo.py          # 5-query demo
    python -c "from bonus.agent import HybridMemoryAgent; a = HybridMemoryAgent(); a.remember('test', 'u_001'); print(a.recall('test', 'u_001'))"
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# ── paths ────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

FEAST_REPO = _ROOT / "app" / "feast_repo"
COLLECTION  = "hybrid_memory"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM   = 384


# ── helpers ──────────────────────────────────────────────────────────────────

def _embed(embedder: TextEmbedding, text: str) -> list[float]:
    """Embed one text string → float list."""
    return np.asarray(next(embedder.embed([text])), dtype=np.float32).tolist()


def _load_feast_store():
    """Return a FeatureStore if Feast registry exists, else None."""
    try:
        from feast import FeatureStore
        registry = FEAST_REPO / "registry.db"
        if registry.exists():
            return FeatureStore(repo_path=str(FEAST_REPO))
    except Exception:  # noqa: BLE001
        pass
    return None


# ── main class ───────────────────────────────────────────────────────────────

@dataclass
class HybridMemoryAgent:
    """Minimal AI memory agent combining Vector Store (episodic) + Feature Store (stable profile).

    Architecture decisions (see ARCHITECTURE.md):
      - Chunking: per-message (each remember() call = 1 vector point)
      - Feature schema: tabular (topic_affinity, reading_speed_wpm, preferred_lang)
      - Freshness: sub-second for episodic (Qdrant upsert sync), daily for profile (Feast)
    """

    client: QdrantClient = field(default_factory=lambda: QdrantClient(":memory:"))
    embedder: TextEmbedding = field(default_factory=lambda: TextEmbedding(EMBED_MODEL))
    top_k: int = 3
    _next_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # Create collection if it does not exist yet.
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    # ── public API ───────────────────────────────────────────────────────────

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user.

        Architecture note (Decision 1): per-message chunking.
        Each call = 1 Qdrant point. Freshness = sub-second (Decision 3).

        Args:
            text:    The text to remember (a message, document excerpt, note).
            user_id: The user this memory belongs to (isolation via payload filter).
        """
        vector = _embed(self.embedder, text)
        point = PointStruct(
            id=self._next_id,
            vector=vector,
            payload={
                "user_id": user_id,
                "text": text[:500],          # cap at 500 chars to keep payload small
                "timestamp": int(time.time()),
                "type": "episodic",
            },
        )
        self.client.upsert(collection_name=COLLECTION, points=[point])
        self._next_id += 1

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features → assembled context string.

        Steps:
          1. Feast online lookup → user profile + recent activity (Decision 2: tabular)
          2. Qdrant hybrid search filtered by user_id (soft isolation — see limitations)
          3. Assemble context string for LLM consumption

        Args:
            query:   The question or topic to retrieve memories for.
            user_id: Which user's memories to search.

        Returns:
            A context string ready to be injected into an LLM prompt.
        """
        # Step 1 — Feature Store: stable profile + recent activity
        features = self._get_features(user_id)

        # Step 2 — Vector Store: episodic memories filtered by user
        memories = self._search_memories(query, user_id)

        # Step 3 — Assemble context
        return self._assemble_context(user_id, query, features, memories)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _get_features(self, user_id: str) -> dict[str, Any]:
        """Fetch online features from Feast. Returns empty dict if Feast not ready."""
        store = _load_feast_store()
        if store is None:
            return {}
        try:
            result = store.get_online_features(
                features=[
                    "user_profile_features:reading_speed_wpm",
                    "user_profile_features:preferred_language",
                    "user_profile_features:topic_affinity",
                    "query_velocity_features:queries_last_hour",
                    "query_velocity_features:distinct_topics_24h",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            return {k: v[0] for k, v in result.items() if k != "user_id"}
        except Exception:  # noqa: BLE001
            return {}

    def _search_memories(self, query: str, user_id: str) -> list[dict]:
        """Hybrid search Qdrant filtered by user_id.

        Architecture note (Decision 1): retrieval is per-message, so each result
        is a precise memory chunk, not a whole conversation.

        Soft isolation via payload filter — NOT cryptographic (see limitations).
        """
        q_vec = _embed(self.embedder, query)
        user_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            query_filter=user_filter,
            limit=self.top_k,
        ).points
        return [
            {"text": p.payload.get("text", ""), "score": p.score,
             "timestamp": p.payload.get("timestamp", 0)}
            for p in results
        ]

    def _assemble_context(
        self,
        user_id: str,
        query: str,
        features: dict,
        memories: list[dict],
    ) -> str:
        """Build a context string combining profile and episodic memories.

        Format optimised for Vietnamese-language LLM prompts.
        Code-switching note: labels in Vietnamese, values may be en/vi.
        """
        lines: list[str] = [f"=== Ngữ cảnh cho user {user_id!r} ==="]

        # Profile block (Decision 2: tabular features)
        if features:
            spd   = features.get("reading_speed_wpm", "?")
            lang  = features.get("preferred_language", "?")
            topic = features.get("topic_affinity", "?")
            qph   = features.get("queries_last_hour", "?")
            dt24  = features.get("distinct_topics_24h", "?")
            lines += [
                f"Hồ sơ người dùng:",
                f"  Tốc độ đọc     : {spd} wpm",
                f"  Ngôn ngữ ưu tiên: {lang}",
                f"  Chủ đề quan tâm : {topic}",
                f"Hoạt động gần đây:",
                f"  Query trong 1h  : {qph}",
                f"  Chủ đề trong 24h: {dt24}",
            ]
        else:
            lines.append("(Feast chưa sẵn sàng — không có hồ sơ người dùng)")

        # Episodic memories block
        if memories:
            lines.append(f"\nKý ức liên quan đến '{query}':")
            for i, m in enumerate(memories, 1):
                lines.append(f"  {i}. [score={m['score']:.3f}] {m['text'][:120]}")
        else:
            lines.append("\n(Chưa có ký ức nào — hãy dùng remember() để thêm)")

        return "\n".join(lines)
