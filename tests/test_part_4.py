"""
Part 4 tests: inference pipeline.
Run: pytest tests/test_part_4.py -v
All tests use use_mocks=True (no LLM/Cohere API cost).
Embeddings are mocked so retrieval results are positional, not semantic — structure is verified, not content.
"""
import os
import sys
import uuid
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from ingestion.classifier import ContentType
from ingestion.chunker import chunk_blocks
from ingestion.embedder import embed_chunks
from ingestion.indexer import build_indexes, reset_qdrant_client
from ingestion.parser import parse_pdf
from inference.expander import expand_query
from inference.reranker import rerank
from inference.generator import generate
from inference.history import add_turn, get_context_history, reset_thread, SUMMARY_THRESHOLD
from inference.pipeline import run_query, CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Session-scoped fixtures: build an indexed test document for pipeline tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_pdf(tmp_path_factory):
    import fitz

    tmp = tmp_path_factory.mktemp("p4_pdf")
    pdf_path = tmp / "test_manual.pdf"
    doc = fitz.open()

    font = fitz.Font("helv")

    p1 = doc.new_page()
    tw = fitz.TextWriter(p1.rect)
    tw.append((50, 60), "1 ENGINE SYSTEM", font=font, fontsize=14)
    tw.append((50, 90), "1.1 ENGINE OVERVIEW", font=font, fontsize=11)
    prose = (
        "The Royal Enfield Meteor 350 features a single-cylinder 349cc SOHC air-cooled engine. "
        "It produces 20.2 bhp at 6100 rpm and 27 Nm of torque at 4000 rpm. "
        "Engine oil capacity is 1.5 litres. Recommended oil grade is 15W-50 fully synthetic."
    )
    tw.append((50, 120), prose, font=font, fontsize=9)
    tw.write_text(p1)

    p2 = doc.new_page()
    tw2 = fitz.TextWriter(p2.rect)
    tw2.append((50, 50), "1.2 OIL CHANGE PROCEDURE", font=font, fontsize=11)
    warning = "WARNING: Hot oil can cause severe burns. Allow engine to cool 30 minutes before draining."
    tw2.append((50, 80), warning, font=font, fontsize=9)
    procedure = (
        "1. Place motorcycle on centre stand on level surface.\n"
        "2. Remove oil drain plug (10mm socket) and drain completely.\n"
        "3. Replace drain plug and torque to 24 Nm.\n"
        "4. Remove old oil filter and install a new filter hand-tight.\n"
        "5. Fill with 1.5 litres of 15W-50 engine oil.\n"
        "6. Start engine, check for leaks, verify oil pressure light extinguishes."
    )
    tw2.append((50, 120), procedure, font=font, fontsize=9)
    tw2.write_text(p2)

    p3 = doc.new_page()
    tw3 = fitz.TextWriter(p3.rect)
    tw3.append((50, 50), "2 FUEL SYSTEM", font=font, fontsize=14)
    tw3.append((50, 80), "2.1 FUEL INJECTION SYSTEM", font=font, fontsize=11)
    prose2 = (
        "The fuel injection system consists of an ECU, throttle body, and MAP sensor. "
        "Throttle body diameter is 36mm. Fuel pressure is 300 kPa. Fuel tank capacity is 15 litres."
    )
    tw3.append((50, 110), prose2, font=font, fontsize=9)
    tw3.write_text(p3)

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture(scope="session")
def doc_meta():
    return {
        "bike_brand": "Royal Enfield",
        "bike_model": "Meteor 350",
        "bike_year": "2022",
        "manual_type": "owner_manual",
        "manual_source": "library",
    }


@pytest.fixture(scope="session")
def indexed_doc(test_pdf, doc_meta):
    parsed = parse_pdf(test_pdf, doc_meta, use_mocks=True)
    document_id = parsed["document_id"]
    chunks = chunk_blocks(parsed["blocks"], doc_meta, document_id)
    embedded = embed_chunks(list(chunks), use_mocks=True)
    reset_qdrant_client()
    os.environ["QDRANT_URL"] = ""
    build_indexes(embedded, document_id)
    return document_id, doc_meta


# ---------------------------------------------------------------------------
# Intent classification tests
# ---------------------------------------------------------------------------

def test_intent_diagnostic():
    result = expand_query("white smoke coming from exhaust pipe", use_mocks=True)
    assert result["intent"] == "diagnostic"


def test_intent_specification():
    result = expand_query("engine oil drain plug torque specification Nm", use_mocks=True)
    assert result["intent"] == "specification"


def test_intent_procedure():
    result = expand_query("how to change engine oil steps", use_mocks=True)
    assert result["intent"] == "procedure"


def test_intent_out_of_scope_price():
    result = expand_query("what is the price of Meteor 350", use_mocks=True)
    assert result["intent"] == "out_of_scope"


def test_intent_out_of_scope_compare():
    result = expand_query("compare Meteor 350 with Classic 350 which is better", use_mocks=True)
    assert result["intent"] == "out_of_scope"


def test_expand_query_returns_required_fields():
    result = expand_query("engine oil capacity", use_mocks=True)
    for field in ("original", "expanded", "intent", "language"):
        assert field in result, f"expand_query missing field: {field}"


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------

def test_retrieve_returns_results(indexed_doc):
    from inference.retriever import retrieve
    document_id, _ = indexed_doc
    results = retrieve("engine oil change", document_id, intent="procedure", use_mocks=True)
    assert len(results) > 0


def test_retrieve_returns_chunk_score_tuples(indexed_doc):
    from inference.retriever import retrieve
    document_id, _ = indexed_doc
    results = retrieve("torque specification", document_id, intent="specification", use_mocks=True)
    assert len(results) > 0
    for chunk, score in results:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert isinstance(score, float)


def test_retrieve_rrf_scores_are_positive(indexed_doc):
    from inference.retriever import retrieve
    document_id, _ = indexed_doc
    results = retrieve("fuel injection system", document_id, intent="diagnostic", use_mocks=True)
    for _, score in results:
        assert score > 0


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------

def _make_chunks(n: int):
    return [
        {"chunk_id": str(i), "text": f"text chunk {i}", "section_number": str(i),
         "section_title": f"Section {i}", "page_number": i}
        for i in range(n)
    ]


def test_rerank_returns_top_n():
    chunks = _make_chunks(6)
    results = rerank("engine oil", chunks, top_n=3, use_mocks=True)
    assert len(results) == 3


def test_rerank_mock_scores_descending():
    chunks = _make_chunks(3)
    results = rerank("query", chunks, top_n=3, use_mocks=True)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_fewer_chunks_than_top_n():
    chunks = _make_chunks(2)
    results = rerank("query", chunks, top_n=3, use_mocks=True)
    assert len(results) == 2


def test_rerank_empty_chunks():
    results = rerank("query", [], top_n=3, use_mocks=True)
    assert results == []


# ---------------------------------------------------------------------------
# Confidence gate tests
# ---------------------------------------------------------------------------

def test_confidence_threshold_value():
    assert CONFIDENCE_THRESHOLD == 0.35


def test_confidence_gate_high(indexed_doc):
    document_id, _ = indexed_doc
    # Mock reranker returns 0.9 > 0.35 → high confidence
    result = run_query("engine oil", document_id, thread_id=str(uuid.uuid4()), use_mocks=True)
    assert result["context_confidence"] == "high"


def test_confidence_gate_low_on_empty_doc():
    # Non-existent document_id → retrieval returns empty → low confidence
    result = run_query("engine oil", "nonexistent_doc_id", thread_id=str(uuid.uuid4()), use_mocks=True)
    assert result["context_confidence"] == "low"


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

def test_generate_all_fields_present():
    chunks = [(_make_chunks(1)[0], 0.9)]
    result = generate("engine oil capacity", chunks, context_confidence="high", use_mocks=True)
    for field in ("answer_text", "spoken_summary", "citations", "severity_label",
                  "confidence", "suggested_followups"):
        assert field in result, f"generate() missing field: {field}"


def test_generate_in_scope_has_citation():
    chunks = [(_make_chunks(1)[0], 0.9)]
    result = generate("engine oil capacity", chunks, context_confidence="high", use_mocks=True)
    # Mock fixture contains citations
    assert isinstance(result["citations"], list)
    assert len(result["citations"]) > 0


def test_generate_out_of_scope_returns_refusal():
    result = generate("what is the price of this bike", [], context_confidence="low", use_mocks=True)
    assert result["citations"] == []
    assert "couldn't find" in result["answer_text"].lower()


def test_generate_out_of_scope_severity():
    result = generate("how much does this motorcycle cost", [], context_confidence="low", use_mocks=True)
    assert result["severity_label"] == "N/A"


# ---------------------------------------------------------------------------
# History tests
# ---------------------------------------------------------------------------

def test_history_empty_thread():
    tid = str(uuid.uuid4())
    ctx = get_context_history(tid, use_mocks=True)
    assert ctx == ""


def test_history_adds_turns():
    tid = str(uuid.uuid4())
    add_turn(tid, "What is the oil capacity?", "The oil capacity is 1.5 litres.")
    ctx = get_context_history(tid, use_mocks=True)
    assert "oil capacity" in ctx.lower() or "1.5" in ctx
    reset_thread(tid)


def test_history_summarization_triggers_at_threshold():
    tid = str(uuid.uuid4())
    for i in range(SUMMARY_THRESHOLD):
        add_turn(tid, f"Question {i} about engine oil", f"Answer {i} about engine.")
    # After SUMMARY_THRESHOLD turns, get_context_history should trigger compression
    ctx = get_context_history(tid, use_mocks=True)
    assert ctx != ""
    assert len(ctx) > 0
    reset_thread(tid)


def test_history_thread_isolation():
    tid_a = str(uuid.uuid4())
    tid_b = str(uuid.uuid4())
    add_turn(tid_a, "Oil capacity question", "1.5 litres answer")
    ctx_a = get_context_history(tid_a, use_mocks=True)
    ctx_b = get_context_history(tid_b, use_mocks=True)
    assert ctx_a != ""
    assert ctx_b == ""
    reset_thread(tid_a)


# ---------------------------------------------------------------------------
# Output validation test
# ---------------------------------------------------------------------------

def test_output_validation_fields_always_populated(indexed_doc):
    document_id, _ = indexed_doc
    tid = str(uuid.uuid4())
    result = run_query("engine oil change", document_id, thread_id=tid, use_mocks=True)
    for field in ("answer_text", "spoken_summary", "citations", "severity_label",
                  "confidence", "suggested_followups"):
        assert field in result, f"run_query missing field: {field}"
        assert result[field] is not None


# ---------------------------------------------------------------------------
# 20 parametrized pipeline end-to-end tests
# ---------------------------------------------------------------------------

PIPELINE_QUERIES = [
    # (query, is_in_scope, description)
    ("white smoke coming from exhaust what does it mean", True, "diagnostic: exhaust smoke"),
    ("engine making knocking sound thak thak noise", True, "diagnostic: knocking noise"),
    ("oil pressure warning light comes on while riding", True, "diagnostic: warning light"),
    ("engine stalling at traffic stops idle", True, "diagnostic: stalling"),
    ("vibration in handlebar at high speed", True, "diagnostic: vibration"),
    ("engine oil capacity how many litres to fill", True, "spec: oil capacity"),
    ("engine oil drain plug torque specification Nm", True, "spec: drain plug torque"),
    ("spark plug torque tightening value", True, "spec: spark plug torque"),
    ("fuel tank capacity total litres", True, "spec: fuel capacity"),
    ("recommended engine oil grade viscosity 15W-50", True, "spec: oil grade"),
    ("how to change engine oil step by step", True, "procedure: oil change"),
    ("how to remove oil drain plug steps", True, "procedure: drain plug removal"),
    ("how to replace oil filter procedure", True, "procedure: oil filter"),
    ("step by step engine oil drain change", True, "procedure: oil drain steps"),
    ("how to check engine oil level dipstick", True, "procedure: oil level check"),
    ("WARNING hot oil burns safety before oil change", True, "warning: safety before oil change"),
    ("caution safety note before starting engine", True, "warning: pre-start safety"),
    ("what is the price of Meteor 350 in India", False, "OOS: price query"),
    ("compare Meteor 350 with Classic 350 which is better", False, "OOS: comparison"),
    ("should I buy Meteor 350 or Honda SP 125", False, "OOS: purchase advice"),
]


@pytest.mark.parametrize("query,is_in_scope,description", PIPELINE_QUERIES)
def test_pipeline_query(query, is_in_scope, description, indexed_doc):
    document_id, _ = indexed_doc
    tid = str(uuid.uuid4())

    result = run_query(query, document_id, thread_id=tid, use_mocks=True)

    # All responses must have all 6 required output fields
    for field in ("answer_text", "spoken_summary", "citations", "severity_label",
                  "confidence", "suggested_followups"):
        assert field in result, f"[{description}] Missing field '{field}'"
        assert result[field] is not None, f"[{description}] Field '{field}' is None"

    # answer_text must always be non-empty
    assert result["answer_text"].strip(), f"[{description}] answer_text is empty"

    if is_in_scope:
        # In-scope: mock fixture always has citations
        assert isinstance(result["citations"], list), f"[{description}] citations not a list"
        assert len(result["citations"]) > 0, f"[{description}] Expected citations for in-scope query"
    else:
        # Out-of-scope: mock generator returns refusal with no citations
        assert result["citations"] == [], f"[{description}] OOS query should return empty citations"
        assert "couldn't find" in result["answer_text"].lower() or "service center" in result["answer_text"].lower(), (
            f"[{description}] OOS query should return refusal message, got: {result['answer_text'][:80]}"
        )
