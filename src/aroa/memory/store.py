"""Memory: three Qdrant collections searched with dense + BM25 vectors fused by Reciprocal Rank Fusion.

- facts:      one durable sentence each, deduplicated at write time by cosine similarity
- episodes:   one per turn (user + assistant), re-ranked by recency at recall time
- documents:  chunks of the files in the documents folder, indexed incrementally by file hash

Vectors are computed locally with fastembed; nothing leaves the machine to index.
"""

import math
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from ..config import Settings
from .chunking import Chunk
from .loaders import document_files, file_hash, load_file

DENSE = "dense"
SPARSE = "bm25"
FACTS, EPISODES, DOCUMENTS = "facts", "episodes", "documents"


def _e5_prefixes(model: str) -> tuple[str, str]:
    """E5 models are trained with "query: " / "passage: " prefixes; fastembed does not add them."""
    return ("query: ", "passage: ") if "e5" in model.lower() else ("", "")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _matter_filter(matter: str) -> models.Filter:
    """Memory of this matter plus general memory (facts stored outside any matter)."""
    return models.Filter(
        should=[
            models.FieldCondition(key="matter", match=models.MatchValue(value=matter)),
            models.FieldCondition(key="matter", match=models.MatchValue(value="")),
        ]
    )


def _age_days(ts: str) -> float:
    try:
        then = datetime.fromisoformat(ts)
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(UTC) - then).total_seconds() / 86400)


class MemoryStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))
        self.dense = TextEmbedding(settings.embedding_model)
        self.sparse = SparseTextEmbedding(settings.sparse_model, language=settings.sparse_language)
        self.query_prefix, self.passage_prefix = _e5_prefixes(settings.embedding_model)
        self._titles: list[str] | None = None

    # ── collections ────────────────────────────────────────────────────
    def _ensure(self, name: str, size: int) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name,
                vectors_config={DENSE: models.VectorParams(size=size, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )

    def _upsert(self, name: str, texts: list[str], payloads: list[dict], ids: list[str]) -> None:
        if not texts:
            return
        dense = list(self.dense.passage_embed([self.passage_prefix + t for t in texts]))
        sparse = list(self.sparse.passage_embed(texts))
        self._ensure(name, len(dense[0]))
        self.client.upsert(
            name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, pid)),
                    vector={
                        DENSE: d.tolist(),
                        SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                    },
                    payload=payload,
                )
                for pid, payload, d, s in zip(ids, payloads, dense, sparse, strict=True)
            ],
        )

    def _search(self, name: str, query: str, k: int, flt: models.Filter | None = None) -> list[models.ScoredPoint]:
        if not self.client.collection_exists(name):
            return []
        dense = next(iter(self.dense.query_embed(self.query_prefix + query)))
        sparse = next(iter(self.sparse.query_embed(query)))
        result = self.client.query_points(
            name,
            # The filter goes on each prefetch: a top-level filter alone does not narrow the candidates
            # that the fusion sees, and the wall between matters must hold at the first step.
            prefetch=[
                models.Prefetch(query=dense.tolist(), using=DENSE, limit=self.settings.prefetch_k, filter=flt),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
                    using=SPARSE,
                    limit=self.settings.prefetch_k,
                    filter=flt,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=flt,
            limit=k,
            with_payload=True,
        )
        return result.points

    def count(self, name: str) -> int:
        return self.client.count(name).count if self.client.collection_exists(name) else 0

    # ── facts ──────────────────────────────────────────────────────────
    def similar_fact(self, text: str) -> tuple[str, float] | None:
        """The most similar stored fact and its cosine score, if any."""
        if not self.client.collection_exists(FACTS):
            return None
        dense = next(iter(self.dense.query_embed(self.query_prefix + text)))
        result = self.client.query_points(FACTS, query=dense.tolist(), using=DENSE, limit=1, with_payload=True)
        if not result.points:
            return None
        point = result.points[0]
        return point.payload["text"], float(point.score)

    def add_facts(
        self, facts: list[str], *, thread_id: str, turn_id: str, matter: str = ""
    ) -> tuple[list[str], list[str]]:
        """Store the facts that are not near-duplicates. Returns (stored, skipped)."""
        stored, skipped = [], []
        for fact in (f.strip() for f in facts):
            if not fact:
                continue
            similar = self.similar_fact(fact)
            if similar and similar[1] >= self.settings.fact_duplicate_threshold:
                skipped.append(fact)
                continue
            self._upsert(
                FACTS,
                [fact],
                [{"text": fact, "ts": _now(), "thread_id": thread_id, "turn_id": turn_id, "matter": matter}],
                [f"fact:{turn_id}:{len(stored)}"],
            )
            stored.append(fact)
        return stored, skipped

    def list_facts(self, limit: int = 200) -> list[dict]:
        if not self.client.collection_exists(FACTS):
            return []
        points, _ = self.client.scroll(FACTS, limit=limit, with_payload=True, with_vectors=False)
        return sorted((p.payload for p in points), key=lambda p: p["ts"], reverse=True)

    def forget(self, text: str) -> int:
        """Delete facts whose text equals the given one. Returns how many were removed."""
        matches = [f for f in self.list_facts(limit=10000) if f["text"] == text]
        if not matches:
            return 0
        self.client.delete(
            FACTS,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="text", match=models.MatchValue(value=text))])
            ),
        )
        return len(matches)

    # ── episodes ───────────────────────────────────────────────────────
    def add_episode(self, user: str, assistant: str, *, thread_id: str, turn_id: str, matter: str = "") -> None:
        text = f"User: {user}\nAssistant: {assistant}"
        payload = {
            "text": text,
            "user": user,
            "assistant": assistant,
            "ts": _now(),
            "thread_id": thread_id,
            "matter": matter,
        }
        self._upsert(EPISODES, [text], [payload], [f"episode:{turn_id}"])

    def recall(self, query: str, *, exclude_thread: str | None = None, matter: str = "") -> dict:
        """Facts and past episodes relevant to the query, limited to this matter and to general memory.

        That limit is the ethical wall: what was said while working on one matter is never recalled in another.
        Episodes are re-ranked by recency.
        """
        wall = _matter_filter(matter)
        facts = [
            {"text": p.payload["text"], "ts": p.payload["ts"], "score": round(float(p.score), 3)}
            for p in self._search(FACTS, query, self.settings.recall_facts, wall)
        ]
        episodes = []
        for p in self._search(EPISODES, query, self.settings.recall_episodes * 3, wall):
            if exclude_thread and p.payload.get("thread_id") == exclude_thread:
                continue  # the current thread is already in the context window
            decay = math.exp(-_age_days(p.payload["ts"]) * math.log(2) / self.settings.episode_half_life_days)
            episodes.append(
                {
                    "user": p.payload["user"],
                    "assistant": p.payload["assistant"],
                    "ts": p.payload["ts"],
                    "score": round(float(p.score) * decay, 4),
                }
            )
        episodes.sort(key=lambda e: e["score"], reverse=True)
        return {"facts": facts, "episodes": episodes[: self.settings.recall_episodes]}

    # ── documents ──────────────────────────────────────────────────────
    def indexed_sources(self) -> dict[str, dict]:
        if not self.client.collection_exists(DOCUMENTS):
            return {}
        sources: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                DOCUMENTS, limit=512, offset=offset, with_payload=["source", "title", "file_hash"], with_vectors=False
            )
            for point in points:
                entry = sources.setdefault(
                    point.payload["source"],
                    {"title": point.payload["title"], "file_hash": point.payload.get("file_hash", ""), "chunks": 0},
                )
                entry["chunks"] += 1
            if offset is None:
                return sources

    def delete_source(self, source: str) -> None:
        self._titles = None
        if not self.client.collection_exists(DOCUMENTS):
            return
        self.client.delete(
            DOCUMENTS,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
            ),
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self._titles = None
        self._upsert(
            DOCUMENTS, [c.context_text for c in chunks], [c.to_payload() for c in chunks], [c.id for c in chunks]
        )

    def sync_documents(self, anonymize=None) -> dict:
        """Index new or changed files in docs_dir and drop removed ones. Copies the samples on first run.

        With `anonymize`, every chunk is pseudonymised before it is embedded or stored."""
        docs = self.settings.docs_dir
        if not docs.exists() and self.settings.samples_dir.exists():
            shutil.copytree(self.settings.samples_dir, docs)
        docs.mkdir(parents=True, exist_ok=True)
        indexed = self.indexed_sources()
        files = {p.name: p for p in document_files(docs)}
        added, removed = [], []
        for name, info in indexed.items():
            if name not in files or file_hash(files[name]) != info["file_hash"]:
                self.delete_source(name)
                removed.append(name)
        for name, path in files.items():
            if name not in indexed or name in removed:
                chunks = load_file(path, self.settings.chunk_max_chars)
                if anonymize:
                    for chunk in chunks:
                        chunk.text = anonymize(chunk.text)
                self.add_chunks(chunks)
                added.append(name)
        return {"added": added, "removed": [r for r in removed if r not in files], "total": len(files)}

    def document_titles(self) -> list[str]:
        if self._titles is None:
            self._titles = sorted({info["title"] for info in self.indexed_sources().values()})
        return self._titles

    def search_documents(self, query: str, k: int | None = None) -> list[Chunk]:
        return [Chunk(**p.payload) for p in self._search(DOCUMENTS, query, k or self.settings.top_k)]

    def stats(self) -> dict:
        return {name: self.count(name) for name in (FACTS, EPISODES, DOCUMENTS)}

    def wipe(self, collections: tuple[str, ...] = (FACTS, EPISODES)) -> None:
        for name in collections:
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
        self._titles = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
