"""
Part 3 tests: ingestion pipeline.
Run: pytest tests/test_part_3.py -v
Uses a synthetic test PDF with known content so results are deterministic.
Embeddings use real OpenAI (user has key). LlamaParse uses mock (no key).
"""
import json
import os
import sys
import pickle
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from ingestion.classifier import ContentType, classify_text
from ingestion.chunker import chunk_blocks
from ingestion.embedder import embed_chunks
from ingestion.indexer import build_indexes, reset_qdrant_client, vector_search
from ingestion.document_index import generate_document_index
from ingestion.parser import parse_pdf


# ---------------------------------------------------------------------------
# Synthetic test PDF fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_pdf(tmp_path_factory):
    """Create a minimal PDF with all five content types for testing."""
    import fitz

    tmp = tmp_path_factory.mktemp("pdf")
    pdf_path = tmp / "test_manual.pdf"
    doc = fitz.open()

    # Page 1: chapter heading + prose
    p1 = doc.new_page()
    tw = fitz.TextWriter(p1.rect)
    font = fitz.Font("helv")
    tw.append((50, 60), "1 ENGINE SYSTEM", font=font, fontsize=14)
    tw.append((50, 90), "1.1 ENGINE OVERVIEW", font=font, fontsize=11)
    prose = (
        "The Royal Enfield Meteor 350 features a single-cylinder 349cc SOHC air-cooled engine. "
        "It produces 20.2 bhp at 6100 rpm and 27 Nm of torque at 4000 rpm. "
        "The fuel delivery system uses electronic fuel injection with a 36mm throttle body "
        "for precise fuel metering across all operating conditions. The engine uses a wet sump "
        "lubrication system with a trochoid oil pump. Engine oil capacity is 1.5 litres. "
        "Recommended oil grade is 15W-50 fully synthetic engine oil."
    )
    tw.append((50, 120), prose, font=font, fontsize=9)
    tw.write_text(p1)

    # Page 2: warning + procedure
    p2 = doc.new_page()
    tw2 = fitz.TextWriter(p2.rect)
    tw2.append((50, 50), "1.2 OIL CHANGE PROCEDURE", font=font, fontsize=11)
    warning = (
        "WARNING: Hot oil can cause severe burns. "
        "Allow engine to cool for at least 30 minutes before draining."
    )
    tw2.append((50, 80), warning, font=font, fontsize=9)
    procedure = (
        "1. Place motorcycle on centre stand on level surface.\n"
        "2. Remove oil drain plug (10mm socket) and drain completely.\n"
        "3. Replace drain plug and torque to 24 Nm.\n"
        "4. Remove old oil filter and install a new filter hand-tight.\n"
        "5. Fill with 1.5 litres of 15W-50 engine oil via filler cap.\n"
        "6. Start engine, check for leaks, verify oil pressure warning light extinguishes."
    )
    tw2.append((50, 120), procedure, font=font, fontsize=9)
    tw2.write_text(p2)

    # Page 3: more prose for retrieval tests
    p3 = doc.new_page()
    tw3 = fitz.TextWriter(p3.rect)
    tw3.append((50, 50), "2 FUEL SYSTEM", font=font, fontsize=14)
    tw3.append((50, 80), "2.1 FUEL INJECTION SYSTEM", font=font, fontsize=11)
    prose2 = (
        "The fuel injection system consists of an ECU, throttle body, fuel injector, "
        "and MAP sensor. The throttle body diameter is 36mm. Fuel pressure is maintained "
        "at 300 kPa by the in-tank fuel pump. The fuel filter is integral to the fuel pump "
        "assembly and is replaced as a unit. Fuel tank capacity is 15 litres."
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
        "manual_type": "service_manual",
        "manual_source": "library",
    }


@pytest.fixture(scope="session")
def parsed_result(test_pdf, doc_meta):
    return parse_pdf(test_pdf, doc_meta, use_mocks=True)


@pytest.fixture(scope="session")
def all_chunks(parsed_result, doc_meta):
    blocks = parsed_result["blocks"]
    document_id = parsed_result["document_id"]
    return chunk_blocks(blocks, doc_meta, document_id)


@pytest.fixture(scope="session")
def embedded_chunks(all_chunks):
    # Use real embeddings — user has OPENAI_API_KEY
    use_mocks = os.getenv("USE_MOCKS", "false").lower() == "true"
    return embed_chunks(list(all_chunks), use_mocks=use_mocks)


@pytest.fixture(scope="session")
def indexed_result(embedded_chunks, parsed_result):
    # Use in-memory Qdrant for tests — don't pollute production Qdrant
    reset_qdrant_client()
    os.environ["QDRANT_URL"] = ""
    document_id = parsed_result["document_id"]
    result = build_indexes(embedded_chunks, document_id)
    return result, document_id


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

def test_warning_classification():
    assert classify_text("WARNING: Hot oil causes burns.") == ContentType.WARNING


def test_caution_classification():
    assert classify_text("CAUTION: Wear eye protection.") == ContentType.WARNING


def test_note_classification():
    assert classify_text("NOTE: Use only genuine parts.") == ContentType.WARNING


def test_procedure_classification():
    text = "1. Do this.\n2. Do that.\n3. Check result.\n4. Tighten bolt."
    assert classify_text(text) == ContentType.PROCEDURE


def test_prose_classification():
    assert classify_text("The engine uses a wet sump lubrication system.") == ContentType.PROSE


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

def test_parse_returns_blocks(parsed_result):
    assert "blocks" in parsed_result
    assert len(parsed_result["blocks"]) > 0


def test_parse_document_id_present(parsed_result):
    assert parsed_result["document_id"] != ""


def test_parse_finds_warning_block(parsed_result):
    types = [b["content_type"] for b in parsed_result["blocks"]]
    assert ContentType.WARNING in types, "No warning block found in parsed output"


def test_parse_finds_procedure_block(parsed_result):
    types = [b["content_type"] for b in parsed_result["blocks"]]
    assert ContentType.PROCEDURE in types, "No procedure block found in parsed output"


def test_parse_finds_specification_block(parsed_result):
    types = [b["content_type"] for b in parsed_result["blocks"]]
    assert ContentType.SPECIFICATION in types, "No specification (table) block found"


def test_parse_finds_prose_block(parsed_result):
    types = [b["content_type"] for b in parsed_result["blocks"]]
    assert ContentType.PROSE in types, "No prose block found"


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

def test_chunks_not_empty(all_chunks):
    assert len(all_chunks) > 0


def test_warning_chunk_isolated(all_chunks):
    warnings = [c for c in all_chunks if c["content_type"] == ContentType.WARNING]
    assert len(warnings) >= 1
    # Warning chunk must contain WARNING/CAUTION/NOTE
    for w in warnings:
        assert any(kw in w["text"].upper() for kw in ("WARNING", "CAUTION", "NOTE"))


def test_warning_chunk_has_severity(all_chunks):
    warnings = [c for c in all_chunks if c["content_type"] == ContentType.WARNING]
    for w in warnings:
        assert w["severity"] in ("WARNING", "CAUTION", "NOTE"), "Missing severity on warning chunk"


def test_procedure_parent_exists(all_chunks):
    parents = [c for c in all_chunks if c["content_type"] == ContentType.PROCEDURE and c["is_parent"]]
    assert len(parents) >= 1


def test_procedure_children_link_to_parent(all_chunks):
    parents = {c["chunk_id"] for c in all_chunks if c.get("is_parent")}
    children = [c for c in all_chunks if c["content_type"] == ContentType.PROCEDURE and not c["is_parent"]]
    for child in children:
        assert child["parent_chunk_id"] in parents, "Child chunk parent_chunk_id doesn't point to a parent"


def test_procedure_parent_step_count(all_chunks):
    parents = [c for c in all_chunks if c["content_type"] == ContentType.PROCEDURE and c["is_parent"]]
    for p in parents:
        assert p["procedure_step_count"] is not None and p["procedure_step_count"] >= 1


def test_specification_chunks_are_key_value(all_chunks):
    specs = [c for c in all_chunks if c["content_type"] == ContentType.SPECIFICATION]
    assert len(specs) > 0, "No specification chunks found"
    for spec in specs:
        # Must have "component: value" format, not flat garbled text
        assert ":" in spec["text"], f"Spec chunk not key-value format: {spec['text'][:80]}"


def test_all_metadata_fields_present(all_chunks):
    required = [
        "chunk_id", "content_type", "text",
        "bike_brand", "bike_model", "bike_year", "manual_type", "manual_source", "document_id",
        "chapter_number", "chapter_title", "section_number", "section_title", "page_number",
        "parent_chunk_id", "is_parent",
        "table_title", "spec_unit", "image_path", "diagram_type",
        "severity", "related_procedure", "procedure_step_count", "tools_required",
    ]
    for chunk in all_chunks:
        for field in required:
            assert field in chunk, f"Chunk missing field '{field}': {chunk.get('chunk_id')}"


def test_document_id_consistent_across_chunks(all_chunks, parsed_result):
    doc_id = parsed_result["document_id"]
    for chunk in all_chunks:
        assert chunk["document_id"] == doc_id


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------

def test_every_chunk_has_vector(embedded_chunks):
    for chunk in embedded_chunks:
        assert "vector" in chunk
        assert len(chunk["vector"]) == 1536


# ---------------------------------------------------------------------------
# Indexing tests
# ---------------------------------------------------------------------------

def test_qdrant_index_built(indexed_result):
    result, _ = indexed_result
    assert result["qdrant_count"] > 0


def test_bm25_index_file_exists(indexed_result):
    _, document_id = indexed_result
    bm25_path = ROOT / "data" / "indexes" / f"{document_id}_bm25.pkl"
    assert bm25_path.exists(), "BM25 index pickle not found"


def test_bm25_index_loadable(indexed_result):
    _, document_id = indexed_result
    bm25_path = ROOT / "data" / "indexes" / f"{document_id}_bm25.pkl"
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    assert "bm25" in data
    assert "chunk_ids" in data
    assert len(data["chunk_ids"]) > 0


# ---------------------------------------------------------------------------
# Document index tests
# ---------------------------------------------------------------------------

def test_document_index_json_generated(embedded_chunks, parsed_result, doc_meta):
    document_id = parsed_result["document_id"]
    index = generate_document_index(document_id, doc_meta, embedded_chunks, confidence_threshold=0.35)
    idx_path = ROOT / "data" / "indexes" / f"{document_id}_index.json"
    assert idx_path.exists()
    loaded = json.loads(idx_path.read_text())
    required_fields = [
        "document_id", "bike_brand", "bike_model", "bike_year", "manual_type",
        "chapters", "content_type_counts", "total_chunks",
        "confidence_threshold", "ingestion_timestamp",
    ]
    for field in required_fields:
        assert field in loaded, f"Document index missing field: {field}"
    assert loaded["total_chunks"] == len(embedded_chunks)
    assert loaded["confidence_threshold"] == 0.35


# ---------------------------------------------------------------------------
# Retrieval quality tests (20 queries against indexed test content)
# ---------------------------------------------------------------------------

RETRIEVAL_QUERIES = [
    # (query_text, expected_content_types_any_of, description, top_k_window)
    ("engine oil change drain plug torque specification", [ContentType.SPECIFICATION], "torque spec lookup", 3),
    ("front axle nut torque value", [ContentType.SPECIFICATION], "axle torque spec", 3),
    ("rear axle nut torque", [ContentType.SPECIFICATION], "rear axle torque spec", 3),
    ("spark plug torque", [ContentType.SPECIFICATION], "spark plug torque", 3),
    ("oil change procedure steps how to drain", [ContentType.PROCEDURE], "oil change procedure", 5),
    ("step by step engine oil drain change", [ContentType.PROCEDURE], "oil change procedure", 5),
    ("drain oil drain plug remove engine", [ContentType.PROCEDURE], "drain step in procedure", 5),
    ("WARNING hot oil burns safety hazard", [ContentType.WARNING], "warning chunk retrieval", 3),
    ("safety caution before oil change warning", [ContentType.WARNING], "warning caution retrieval", 3),
    ("engine oil capacity litres wet sump", [ContentType.PROSE, ContentType.PROCEDURE, ContentType.SPECIFICATION], "oil capacity", 3),
    ("Meteor 350 engine displacement cc cubic", [ContentType.PROSE], "prose: engine spec", 3),
    ("rpm bhp power output engine performance", [ContentType.PROSE], "prose: power output", 3),
    ("fuel injection throttle body diameter mm", [ContentType.PROSE], "prose: fuel system", 3),
    ("fuel tank capacity total litres", [ContentType.PROSE], "prose: fuel capacity", 3),
    ("EFI MAP sensor fuel pump pressure system", [ContentType.PROSE], "prose: fuel system component", 3),
    ("oil filter step remove install replace", [ContentType.PROCEDURE], "procedure: oil filter step", 5),
    ("engine oil grade viscosity 15W-50", [ContentType.PROSE, ContentType.SPECIFICATION], "oil grade", 3),
    ("lubrication system wet sump trochoid pump", [ContentType.PROSE, ContentType.PROCEDURE, ContentType.SPECIFICATION], "lubrication system", 3),
    ("torque specifications table maintenance values", [ContentType.SPECIFICATION], "spec table", 3),
    ("cylinder head bolt tightening torque", [ContentType.SPECIFICATION], "cylinder head spec", 3),
]


@pytest.mark.parametrize("query,expected_cts,description,top_k", RETRIEVAL_QUERIES)
def test_retrieval_query(query, expected_cts, description, top_k, indexed_result, embedded_chunks):
    _, document_id = indexed_result
    use_mocks = os.getenv("USE_MOCKS", "false").lower() == "true"

    if use_mocks:
        from ingestion.embedder import _MOCK_VECTOR
        query_vector = _MOCK_VECTOR
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
        query_vector = resp.data[0].embedding

    results = vector_search(query_vector, document_id, top_k=top_k)
    assert len(results) > 0, f"No results returned for: {query}"

    top_types = [r[0].get("content_type") for r in results]
    matched = any(ct in top_types for ct in expected_cts)
    assert matched, (
        f"[{description}] Query '{query}' expected one of {expected_cts} in top-{top_k}, "
        f"got {top_types}"
    )
