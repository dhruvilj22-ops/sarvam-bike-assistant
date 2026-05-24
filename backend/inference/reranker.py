"""
Cohere Rerank cross-encoder: narrows top 5-7 chunks to top 3 high-quality results.
Uses requests (no cohere SDK needed) — REQUESTS_CA_BUNDLE env var handles SSL.
"""
import logging
import os
from typing import Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)

_COHERE_URL = "https://api.cohere.ai/v1/rerank"
_MOCK_SCORES = [0.9, 0.7, 0.5, 0.4, 0.3]


def rerank(
    query: str,
    chunks: List[Dict],
    top_n: int = 3,
    use_mocks: bool = False,
) -> List[Tuple[Dict, float]]:
    """Return top_n (chunk, relevance_score) pairs, sorted by relevance descending."""
    if not chunks:
        return []
    top_n = min(top_n, len(chunks))

    if use_mocks:
        return [
            (chunks[i], _MOCK_SCORES[i] if i < len(_MOCK_SCORES) else 0.4)
            for i in range(top_n)
        ]

    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('COHERE_API_KEY', '')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "rerank-english-v3.0",
            "query": query,
            "documents": [c["text"] for c in chunks],
            "top_n": top_n,
        }
        resp = requests.post(_COHERE_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        out = [
            (chunks[r["index"]], r["relevance_score"])
            for r in data["results"]
        ]
        logger.info("rerank_summary top_n=%s in_chunks=%s out_chunks=%s", top_n, len(chunks), len(out))
        for i, (chunk, score) in enumerate(out[:3], start=1):
            logger.info(
                "rerank_top rank=%s chunk_id=%s page=%s section=%s relevance=%.6f",
                i,
                chunk.get("chunk_id", ""),
                chunk.get("page_number", 0),
                chunk.get("section_number", ""),
                float(score),
            )
        return out
    except Exception:
        logger.warning("Cohere rerank failed — falling back to mock scores")
        return [
            (chunks[i], _MOCK_SCORES[i] if i < len(_MOCK_SCORES) else 0.4)
            for i in range(top_n)
        ]
