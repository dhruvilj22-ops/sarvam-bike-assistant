"""
Dual index: Qdrant vector store (external or in-memory) + BM25 keyword index.
Namespace isolation is enforced via document_id payload filter, not separate collections.
Hybrid BM25 + dense retrieval: BM25 handles exact part codes/specs, semantic handles natural language.
"""
import os
import pickle
import urllib.parse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

_COLLECTION = "bike_manuals"
_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_INDEX_DIR = "/tmp/indexes" if os.getenv("VERCEL") else str(_ROOT / "data" / "indexes")
_INDEX_DIR = Path(os.getenv("INDEX_DIR", _DEFAULT_INDEX_DIR))

_qdrant_client: Optional[QdrantClient] = None
logger = logging.getLogger(__name__)


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        # Feature flag:
        # USE_CLOUD_QDRANT=true  -> attempt remote Cloud Qdrant
        # USE_CLOUD_QDRANT=false -> force in-memory mode
        raw_use_cloud = os.getenv("USE_CLOUD_QDRANT")
        url = os.getenv("QDRANT_URL", "").strip().rstrip("/")
        # Default to cloud when URL is present, unless explicitly disabled.
        use_cloud = bool(url) if raw_use_cloud is None else _is_truthy(raw_use_cloud)
        if use_cloud and url:
            try:
                logger.info("qdrant_init mode=remote url=%s", url)
                # Qdrant Cloud uses port 443 (HTTPS), not the self-hosted default 6333.
                # qdrant_client ignores the URL scheme and falls back to 6333 unless told explicitly.
                parsed = urllib.parse.urlparse(url)
                port = parsed.port or (443 if parsed.scheme == "https" else 6333)
                _qdrant_client = QdrantClient(
                    url=url,
                    port=port,
                    api_key=os.getenv("QDRANT_API_KEY") or None,
                )
                # Force a lightweight call so config/cert/path errors surface now.
                _qdrant_client.get_collections()
                logger.info("qdrant_init_success mode=remote")
            except Exception:
                logger.exception("qdrant_init_failed mode=remote, falling_back=memory")
                _qdrant_client = QdrantClient(":memory:")
                logger.info("qdrant_init_success mode=memory")
        else:
            reason = "feature_flag_disabled"
            if use_cloud and not url:
                reason = "no_qdrant_url"
            logger.info("qdrant_init mode=memory reason=%s", reason)
            _qdrant_client = QdrantClient(":memory:")
    return _qdrant_client


def reset_qdrant_client():
    """Force a new client on next call. Used in tests to get a fresh in-memory instance."""
    global _qdrant_client
    _qdrant_client = None


def _ensure_collection(client: QdrantClient, dim: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if _COLLECTION not in existing:
        client.create_collection(
            _COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def build_indexes(chunks: List[Dict], document_id: str) -> Dict:
    """
    Store chunks in Qdrant and build a BM25 index for the given document namespace.
    Returns {"qdrant_count": int, "bm25_path": str}.
    """
    if not chunks:
        return {"qdrant_count": 0, "bm25_path": ""}

    dim = len(chunks[0]["vector"])
    client = get_qdrant_client()
    _ensure_collection(client, dim)

    # Delete any existing chunks for this document_id before re-indexing
    try:
        from qdrant_client.models import FilterSelector
        client.delete(
            _COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
            ),
        )
    except Exception:
        pass

    points = []
    for i, chunk in enumerate(chunks):
        payload = {k: v for k, v in chunk.items() if k != "vector"}
        points.append(PointStruct(id=i, vector=chunk["vector"], payload=payload))

    client.upsert(_COLLECTION, points=points)

    # BM25 index — simple whitespace tokenization for keyword matching
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    chunk_ids = [c["chunk_id"] for c in chunks]

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    bm25_path = _INDEX_DIR / f"{document_id}_bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids, "chunks": chunks}, f)

    return {"qdrant_count": len(points), "bm25_path": str(bm25_path)}


def vector_search(
    query_vector: List[float],
    document_id: str,
    top_k: int = 7,
) -> List[Tuple[Dict, float]]:
    """Return top_k chunks for document_id with their cosine scores."""
    client = get_qdrant_client()
    try:
        result = client.query_points(
            collection_name=_COLLECTION,
            query=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            limit=top_k,
            with_payload=True,
        )
        return [(r.payload, r.score) for r in result.points]
    except (ValueError, Exception):
        return []


def bm25_search(
    query_tokens: List[str],
    document_id: str,
    top_k: int = 7,
) -> List[Tuple[Dict, float]]:
    """Return top_k chunks by BM25 score for document_id."""
    bm25_path = _INDEX_DIR / f"{document_id}_bm25.pkl"
    if not bm25_path.exists():
        return []
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    bm25: BM25Okapi = data["bm25"]
    chunks: List[Dict] = data["chunks"]
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices]
