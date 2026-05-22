"""
Full inference pipeline: expand → retrieve → rerank → confidence gate → generate.
Entry point for both the API layer (Part 5) and CLI testing.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "backend"))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from inference.expander import expand_query
from inference.generator import generate
from inference.history import add_turn, get_context_history, reset_thread
from inference.reranker import rerank
from inference.retriever import retrieve

CONFIDENCE_THRESHOLD = 0.35


def run_query(
    query: str,
    document_id: str,
    thread_id: str = "default",
    use_mocks: Optional[bool] = None,
) -> Dict:
    """
    Run the full inference pipeline for a single query.
    Returns the structured response dict with all 6 output fields plus pipeline metadata.
    """
    if use_mocks is None:
        use_mocks = os.getenv("USE_MOCKS", "false").lower() == "true"

    # Step 1: Query expansion + intent classification
    expanded = expand_query(query, use_mocks=use_mocks)
    intent = expanded["intent"]

    # Step 2 & 3: Hybrid retrieval (embedding happens inside retriever)
    chunks_with_scores = retrieve(
        query=expanded["expanded"],
        document_id=document_id,
        intent=intent,
        top_k=7,
        use_mocks=use_mocks,
    )

    if not chunks_with_scores:
        result = {
            "answer_text": (
                "I couldn't find this in your manual. "
                "For this issue, I'd recommend visiting an authorised service center."
            ),
            "spoken_summary": "No relevant content was found in the manual.",
            "citations": [],
            "severity_label": "N/A",
            "confidence": "low",
            "suggested_followups": [],
            "intent": intent,
            "language": expanded["language"],
            "context_confidence": "low",
        }
        add_turn(thread_id, query, result["answer_text"])
        return result

    # Step 4: Rerank to top 5
    chunks_only = [c for c, _ in chunks_with_scores]
    reranked = rerank(expanded["expanded"], chunks_only, top_n=5, use_mocks=use_mocks)

    # Step 5: Confidence gate
    top_score = reranked[0][1] if reranked else 0.0
    context_confidence = "low" if top_score < CONFIDENCE_THRESHOLD else "high"

    # Step 6: History context for this thread
    history_context = get_context_history(thread_id, use_mocks=use_mocks)

    # Step 7: Generate response
    response = generate(
        query=query,
        chunks=reranked,
        context_confidence=context_confidence,
        history_context=history_context,
        use_mocks=use_mocks,
        language=expanded["language"],
    )

    # Attach pipeline metadata
    response["intent"] = intent
    response["language"] = expanded["language"]
    response["context_confidence"] = context_confidence

    # Step 8: Update history
    add_turn(thread_id, query, response.get("answer_text", ""))

    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Bike assistant inference pipeline")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--document-id", required=True, help="Document ID to query against")
    parser.add_argument("--thread-id", default="default", help="Conversation thread ID")
    parser.add_argument("--mock", action="store_true", help="Use mock APIs")
    args = parser.parse_args()

    result = run_query(
        query=args.query,
        document_id=args.document_id,
        thread_id=args.thread_id,
        use_mocks=args.mock,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
