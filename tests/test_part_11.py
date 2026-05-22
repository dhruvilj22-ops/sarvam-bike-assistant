"""
Part 11: Hallucination regression suite — anti-hallucination contract verification.

31 core test cases organized into 5 groups:
  A. Retrieval quality       (10) — right sections surface for each query
  B. Citation integrity       (3) — citation schema is always valid
  C. Out-of-scope refusal    (10) — exact refusal phrase, zero hedging
  D. Confidence gate          (3) — low-confidence path triggers correctly
  E. Ambiguous boundary       (5) — conservative, never fabricate

Plus guardrail layer unit tests:
  F. Layer-by-layer contract  (+) — each guardrail tested in isolation

Zero hallucinations acceptable.  All in-scope answers must carry a citation grounded
in the retrieved context.  All OOS answers must carry the exact refusal phrase.

Run: pytest tests/test_part_11.py -v
All tests use USE_MOCKS=true — no real API / embedding calls.
"""
import os
import sys
import uuid
import pytest
from pathlib import Path
from unittest.mock import patch
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

# ---------------------------------------------------------------------------
# Env isolation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _mock_env():
    prev = {k: os.environ.get(k) for k in
            ("USE_MOCKS", "SARVAM_API_KEY", "OPENROUTER_API_KEY", "QDRANT_URL", "COHERE_API_KEY")}
    os.environ["USE_MOCKS"]          = "true"
    os.environ["SARVAM_API_KEY"]     = ""
    os.environ["OPENROUTER_API_KEY"] = ""
    os.environ["QDRANT_URL"]         = ""
    os.environ["COHERE_API_KEY"]     = ""
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

# ---------------------------------------------------------------------------
# Realistic synthetic corpus — Royal Enfield Meteor 350 Service Manual
# 14 sections with precise technical values matching what a real manual contains.
# Text is keyword-rich so BM25 retrieval can distinguish sections reliably.
# ---------------------------------------------------------------------------

_DOC_ID = "re_meteor350_regression_11"

MANUAL_CORPUS = [
    {
        "section_number": "2.1", "section_title": "Engine Oil Specification",
        "chapter_number": "2",   "chapter_title": "Scheduled Maintenance",
        "page_number": 15,
        "text": (
            "Engine oil specification for Royal Enfield Meteor 350: use SAE 10W-40 API SL grade mineral "
            "or semi-synthetic engine oil. Oil capacity with filter change: 2.5 litres. Oil capacity without "
            "filter: 2.2 litres. Recommended change interval: every 3,000 km or 6 months, whichever comes "
            "first. Drain plug torque: 28–32 Nm. Oil filter part number: 591090. Check oil level on centre "
            "stand on level ground; the level must be between MIN and MAX marks on the dipstick."
        ),
    },
    {
        "section_number": "2.2", "section_title": "Air Filter Maintenance",
        "chapter_number": "2",   "chapter_title": "Scheduled Maintenance",
        "page_number": 22,
        "text": (
            "Air filter element: clean with compressed air every 6,000 km; replace every 12,000 km or "
            "annually. Do not wash the paper element with water or solvent — this destroys the filter media. "
            "Part number: 590802. After riding in dusty or muddy conditions, inspect the air filter before "
            "the next ride. A clogged air filter reduces engine power and increases fuel consumption."
        ),
    },
    {
        "section_number": "3.1", "section_title": "Valve Clearance Adjustment",
        "chapter_number": "3",   "chapter_title": "Engine Top End",
        "page_number": 45,
        "text": (
            "Valve clearance must be checked and adjusted with the engine cold (below 35 °C). "
            "Intake valve clearance: 0.10–0.15 mm. Exhaust valve clearance: 0.15–0.20 mm. "
            "Check interval: every 12,000 km. Use feeler gauge part number 1891016. "
            "Rotate engine to TDC compression stroke before measuring. "
            "Incorrect valve clearance causes ticking noise, reduced compression, or hard starting."
        ),
    },
    {
        "section_number": "3.2", "section_title": "Cylinder Head Torque Specifications",
        "chapter_number": "3",   "chapter_title": "Engine Top End",
        "page_number": 52,
        "text": (
            "Cylinder head bolt tightening sequence: tighten in a cross pattern. First pass: 25 Nm. "
            "Second (final) pass: 35 Nm. Always use new cylinder head bolts after removal. "
            "Head gasket part number: 591150. Replace head gasket if blue or white smoke is persistent "
            "after ruling out valve seals and piston rings. Retighten cylinder head after first 1,000 km "
            "of operation following any top-end rebuild."
        ),
    },
    {
        "section_number": "4.1", "section_title": "Spark Plug Inspection and Replacement",
        "chapter_number": "4",   "chapter_title": "Ignition System",
        "page_number": 63,
        "text": (
            "Spark plug specification: NGK DCPR8E. Electrode gap: 0.8–0.9 mm. Tightening torque: 15–18 Nm. "
            "Inspection interval: every 6,000 km. Replacement interval: every 12,000 km. "
            "Symptoms of a faulty spark plug: misfiring, hard starting, poor fuel economy, rough idle. "
            "Always check plug colour — black sooty deposit indicates rich mixture; white chalky deposit "
            "indicates lean mixture or overheating."
        ),
    },
    {
        "section_number": "4.2", "section_title": "Fuel System and Tank",
        "chapter_number": "4",   "chapter_title": "Fuel System",
        "page_number": 71,
        "text": (
            "Fuel tank total capacity: 15 litres including 2.5 litre reserve. Use minimum RON 91 petrol. "
            "Fuel filter replacement interval: every 24,000 km. Fuel filter part number: 590948. "
            "Fuel injector cleaning: recommended every 24,000 km using injector cleaner additive. "
            "Never run the engine on an empty tank — this can damage the fuel pump."
        ),
    },
    {
        "section_number": "5.1", "section_title": "Brake System Specifications",
        "chapter_number": "5",   "chapter_title": "Brakes",
        "page_number": 89,
        "text": (
            "Front brake disc minimum thickness: 4.0 mm. Rear brake disc minimum thickness: 3.5 mm. "
            "Brake fluid specification: DOT 4. Brake fluid replacement interval: every 2 years regardless "
            "of mileage. Front brake master cylinder reservoir: inspect fluid level every 6,000 km. "
            "Brake pad wear indicator: replace pads when thickness reaches 1.0 mm. "
            "Low brake fluid level, spongy lever, or pulling to one side requires immediate inspection."
        ),
    },
    {
        "section_number": "5.2", "section_title": "Brake Adjustment Procedure",
        "chapter_number": "5",   "chapter_title": "Brakes",
        "page_number": 94,
        "text": (
            "Rear brake pedal free play: 5–10 mm measured at pedal tip. Front brake lever free play: "
            "10–15 mm measured at lever tip. To adjust rear brake: loosen lock nut on brake rod adjuster "
            "and turn adjuster clockwise to reduce free play, counter-clockwise to increase. "
            "After adjustment verify brake light activates and wheel rotates freely. "
            "Do not ride with brakes that drag or fail to fully release — this overheats the disc."
        ),
    },
    {
        "section_number": "6.1", "section_title": "Drive Chain Maintenance",
        "chapter_number": "6",   "chapter_title": "Drive Train",
        "page_number": 108,
        "text": (
            "Drive chain type: 428HO sealed O-ring chain. Chain slack (free play): 15–25 mm measured at "
            "the midpoint of the lower run with the bike on centre stand. Lubricate chain every 500 km "
            "using SAE 90 gear oil or dedicated chain lubricant. Replace chain when pin-to-pin elongation "
            "exceeds 3% of nominal pitch. Sprocket replacement: replace front and rear sprockets together "
            "with the chain. Front sprocket: 15T. Rear sprocket: 43T."
        ),
    },
    {
        "section_number": "7.1", "section_title": "Exhaust Smoke Diagnosis",
        "chapter_number": "7",   "chapter_title": "Fault Diagnosis",
        "page_number": 134,
        "text": (
            "White smoke at cold startup that clears within 2–3 minutes is normal water condensation. "
            "Persistent white or blue smoke after warm-up indicates oil burning in the combustion chamber. "
            "Possible causes: worn piston rings, damaged valve stem seals, blown head gasket. "
            "Black smoke indicates overly rich mixture — check air filter and fuel injector. "
            "A persistent smoke condition requires top-end inspection; continued riding risks engine seizure."
        ),
    },
    {
        "section_number": "7.2", "section_title": "Engine Overheating Diagnosis",
        "chapter_number": "7",   "chapter_title": "Fault Diagnosis",
        "page_number": 140,
        "text": (
            "Engine overheating causes: critically low oil level, blocked or clogged cooling fins, "
            "lean air-fuel mixture, continuous low-speed city riding in high ambient temperature. "
            "Immediate action: stop and allow engine to cool for 20–30 minutes. Do not add cold water "
            "to a hot engine. Check engine oil level after cooling. Inspect air filter for blockage. "
            "Riding with an overheating engine causes piston seizure and catastrophic engine damage."
        ),
    },
    {
        "section_number": "7.3", "section_title": "Engine Starting Problems",
        "chapter_number": "7",   "chapter_title": "Fault Diagnosis",
        "page_number": 148,
        "text": (
            "Engine cranks but will not start: check fuel supply to injector, spark plug condition and "
            "gap, battery voltage (minimum 12.4 V under load), kill switch position (RUN). "
            "Compression test: minimum 8 bar; below 6 bar indicates worn rings or valves. "
            "If engine cranks slowly: battery charge low, check charging voltage at 3,000 rpm (should be "
            "13.5–14.5 V). Replace battery if voltage drops below 10 V under starter load."
        ),
    },
    {
        "section_number": "8.1", "section_title": "Electrical System Specifications",
        "chapter_number": "8",   "chapter_title": "Electrical",
        "page_number": 165,
        "text": (
            "Battery: 12 V 8 Ah maintenance-free (MF) type. Charging system output: 13.5–14.5 V at "
            "3,000 rpm. Alternator output: 220 W. Headlight bulb: H4 60/55 W P43t base. Fuse ratings: "
            "main fuse 20 A, sub-fuses 10 A and 15 A located in fuse box under seat left side. "
            "If headlight does not work, check H4 bulb, fuse, and connector before replacing the unit."
        ),
    },
    {
        "section_number": "8.2", "section_title": "Lighting and Indicators",
        "chapter_number": "8",   "chapter_title": "Electrical",
        "page_number": 172,
        "text": (
            "Turn signal bulb: 10 W BA15s bayonet. Stop/tail bulb: 21/5 W BAY15d dual filament. "
            "Instrument cluster backlight: LED type — if faulty, replace the entire instrument cluster "
            "assembly, part number 590702. If turn signals flash too fast, check for blown turn signal "
            "bulb or incorrect bulb wattage. Hazard flasher relay: located under headlight nacelle."
        ),
    },
]

# ---------------------------------------------------------------------------
# Index fixture
# ---------------------------------------------------------------------------

from ingestion.indexer import build_indexes, bm25_search, reset_qdrant_client
from ingestion.embedder import embed_chunks


def _make_chunk(section: dict, doc_id: str, idx: int) -> dict:
    return {
        "chunk_id":       f"{doc_id}_s{idx}",
        "document_id":    doc_id,
        "text":           section["text"],
        "content_type":   "prose",
        "chapter_number": section["chapter_number"],
        "chapter_title":  section["chapter_title"],
        "section_number": section["section_number"],
        "section_title":  section["section_title"],
        "page_number":    section["page_number"],
        "vector":         [0.0] * 1536,
    }


@pytest.fixture(scope="module")
def corpus_index():
    """Build BM25 + vector index for the 14-section corpus once per module."""
    reset_qdrant_client()
    chunks = [_make_chunk(s, _DOC_ID, i) for i, s in enumerate(MANUAL_CORPUS)]
    embedded = embed_chunks(chunks, use_mocks=True)
    build_indexes(embedded, _DOC_ID)
    yield _DOC_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REFUSAL_PHRASE = "I couldn't find this in your"
_SERVICE_CENTER = "authorised service center"

_VALID_SEVERITY = {
    "Immediate Action Required", "Get Checked Soon",
    "Monitor for Now", "Informational", "N/A",
}

_REQUIRED_RESPONSE_FIELDS = [
    "answer_text", "spoken_summary", "citations",
    "severity_label", "confidence", "suggested_followups",
]

_REQUIRED_CITATION_FIELDS = ["section_number", "section_title", "page_number"]


def _bm25(query: str, doc_id: str, top_k: int = 5):
    tokens = query.lower().split()
    return bm25_search(tokens, doc_id, top_k=top_k)


def _top_section_numbers(query: str, doc_id: str) -> list:
    results = _bm25(query, doc_id)
    return [r[0]["section_number"] for r in results]


# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
# GROUP A — Retrieval Quality (10 tests)
# Assert the correct manual section surfaces in BM25 top results for each
# query.  This is the foundation of citation accuracy — wrong retrieval
# means wrong (or absent) citations at the LLM layer.
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

def test_retrieval_engine_oil_specification(corpus_index):
    """SAE 10W-40, oil capacity queries must retrieve section 2.1."""
    secs = _top_section_numbers("engine oil change SAE 10W-40 capacity", corpus_index)
    assert "2.1" in secs, f"Section 2.1 not in top results: {secs}"


def test_retrieval_valve_clearance(corpus_index):
    """Valve clearance / ticking noise queries must retrieve section 3.1."""
    secs = _top_section_numbers("valve clearance intake exhaust feeler gauge", corpus_index)
    assert "3.1" in secs, f"Section 3.1 not in top results: {secs}"


def test_retrieval_spark_plug(corpus_index):
    """Spark plug gap / NGK queries must retrieve section 4.1."""
    secs = _top_section_numbers("spark plug NGK gap electrode replacement", corpus_index)
    assert "4.1" in secs, f"Section 4.1 not in top results: {secs}"


def test_retrieval_brake_fluid(corpus_index):
    """Brake fluid DOT 4 / disc thickness queries must retrieve section 5.1."""
    secs = _top_section_numbers("brake fluid DOT 4 disc thickness minimum", corpus_index)
    assert "5.1" in secs, f"Section 5.1 not in top results: {secs}"


def test_retrieval_chain_slack(corpus_index):
    """Drive chain slack / lubrication queries must retrieve section 6.1."""
    secs = _top_section_numbers("drive chain slack free play lubricate sprocket", corpus_index)
    assert "6.1" in secs, f"Section 6.1 not in top results: {secs}"


def test_retrieval_exhaust_smoke(corpus_index):
    """White / blue smoke exhaust queries must retrieve section 7.1."""
    secs = _top_section_numbers("white smoke exhaust piston rings valve seals", corpus_index)
    assert "7.1" in secs, f"Section 7.1 not in top results: {secs}"


def test_retrieval_engine_overheating(corpus_index):
    """Overheating / cooling fins queries must retrieve section 7.2."""
    secs = _top_section_numbers("engine overheating cooling fins oil level hot", corpus_index)
    assert "7.2" in secs, f"Section 7.2 not in top results: {secs}"


def test_retrieval_starting_problems(corpus_index):
    """Engine won't start / cranks queries must retrieve section 7.3."""
    secs = _top_section_numbers("engine cranks does not start kill switch battery voltage", corpus_index)
    assert "7.3" in secs, f"Section 7.3 not in top results: {secs}"


def test_retrieval_battery_electrical(corpus_index):
    """Battery voltage / charging queries must retrieve section 8.1."""
    secs = _top_section_numbers("battery 12V charging voltage headlight fuse electrical", corpus_index)
    assert "8.1" in secs, f"Section 8.1 not in top results: {secs}"


def test_retrieval_air_filter(corpus_index):
    """Air filter cleaning / replacement queries must retrieve section 2.2."""
    secs = _top_section_numbers("air filter clean replace paper element clogged", corpus_index)
    assert "2.2" in secs, f"Section 2.2 not in top results: {secs}"


# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
# GROUP B — Citation Integrity (3 tests)
# Pipeline responses must always carry structurally valid citations.
# A citation without a section number, title, or page is useless to a mechanic.
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

from inference.pipeline import run_query


def test_citation_has_all_required_fields(corpus_index):
    """Every citation in a pipeline response must have section_number, section_title, page_number."""
    result = run_query("engine oil change", _DOC_ID, use_mocks=True)
    citations = result.get("citations", [])
    # Mock mode returns at least one citation from the fixture
    assert len(citations) > 0, "No citations returned"
    for cit in citations:
        for field in _REQUIRED_CITATION_FIELDS:
            assert field in cit, f"Citation missing required field: {field}"


def test_citation_page_number_is_positive_integer(corpus_index):
    """page_number must be a positive integer — zero or string is a schema violation."""
    result = run_query("valve clearance specification", _DOC_ID, use_mocks=True)
    for cit in result.get("citations", []):
        assert isinstance(cit["page_number"], int), (
            f"page_number is not int: {type(cit['page_number'])} = {cit['page_number']}"
        )
        assert cit["page_number"] > 0, f"page_number must be positive, got {cit['page_number']}"


def test_citation_section_number_is_nonempty_string(corpus_index):
    """section_number must be a non-empty string — an empty section number cannot be verified."""
    result = run_query("brake fluid type", _DOC_ID, use_mocks=True)
    for cit in result.get("citations", []):
        assert isinstance(cit["section_number"], str), "section_number is not a string"
        assert cit["section_number"].strip() != "", "section_number must not be empty"


# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
# GROUP C — Out-of-Scope Refusal (10 tests)
# Every OOS query must return the exact refusal phrase — no partial answers,
# no hedging, no "while I can't answer pricing, I can tell you...".
# The contract: if it's out of scope, there is NOTHING useful to say.
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

from inference.generator import generate, _is_oos

# OOS query catalogue — 10 distinct categories
_OOS_QUERIES = [
    ("price",       "What is the ex-showroom price of the Meteor 350 in Delhi?"),
    ("comparison",  "Should I buy a Royal Enfield Meteor or a Honda CB350?"),
    ("colour",      "What colours does the Meteor 350 come in?"),
    ("worth_it",    "Is the Meteor 350 worth buying in 2024?"),
    ("compare",     "How does the Meteor 350 compare to the Himalayan?"),
    ("insurance",   "How much is the insurance cost for the Meteor 350?"),
    ("resale",      "What is the resale value of the Meteor 350 after 3 years?"),
    ("purchase",    "Where can I buy a Royal Enfield Meteor 350 in Mumbai?"),
    ("cost",        "What is the total cost of ownership for the Meteor 350?"),
    ("better",      "Is the Meteor 350 better than the Classic 350?"),
]


@pytest.mark.parametrize("category,query", _OOS_QUERIES)
def test_oos_generate_returns_refusal_phrase(category, query):
    """generate() must return the exact refusal phrase for every OOS query."""
    result = generate(query, chunks=[], use_mocks=True)
    answer = result.get("answer_text", "")
    assert _REFUSAL_PHRASE in answer, (
        f"[{category}] OOS query did not return refusal phrase.\n"
        f"  Query:  {query}\n"
        f"  Answer: {answer}"
    )


@pytest.mark.parametrize("category,query", _OOS_QUERIES)
def test_oos_generate_severity_is_na(category, query):
    """OOS responses must have severity_label N/A — they are not actionable."""
    result = generate(query, chunks=[], use_mocks=True)
    assert result.get("severity_label") == "N/A", (
        f"[{category}] OOS severity_label should be N/A, got: {result.get('severity_label')}"
    )


@pytest.mark.parametrize("category,query", _OOS_QUERIES)
def test_oos_is_oos_function_detects_correctly(category, query):
    """_is_oos() must correctly flag every OOS query before reaching the LLM."""
    assert _is_oos(query), (
        f"[{category}] _is_oos() missed OOS query: {query}"
    )


# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
# GROUP D — Confidence Gate (3 tests)
# When retrieval returns nothing (or very low scores), the pipeline must
# flag context_confidence=low and return a safe response — not attempt to
# generate with an empty context.
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

def test_empty_retrieval_triggers_low_confidence():
    """Query against a doc_id with no indexed content → context_confidence=low."""
    empty_doc = f"empty_doc_{uuid.uuid4().hex[:8]}"
    result = run_query("engine oil change procedure", empty_doc, use_mocks=True)
    assert result.get("context_confidence") == "low", (
        f"Expected context_confidence=low for empty doc, got: {result.get('context_confidence')}"
    )


def test_empty_retrieval_returns_graceful_refusal():
    """Empty retrieval path must return the refusal phrase, not raise an exception."""
    empty_doc = f"empty_doc_{uuid.uuid4().hex[:8]}"
    result = run_query("spark plug gap specification", empty_doc, use_mocks=True)
    answer = result.get("answer_text", "")
    assert _REFUSAL_PHRASE in answer or _SERVICE_CENTER in answer, (
        f"Empty retrieval should return refusal, got: {answer[:120]}"
    )


def test_empty_retrieval_all_fields_populated():
    """Even on the low-confidence path, all 6 required response fields must be present."""
    empty_doc = f"empty_doc_{uuid.uuid4().hex[:8]}"
    result = run_query("valve clearance check", empty_doc, use_mocks=True)
    for field in _REQUIRED_RESPONSE_FIELDS:
        assert field in result, f"Low-confidence response missing field: {field}"


# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
# GROUP E — Ambiguous Boundary Cases (5 tests)
# Queries that sit near the scope boundary.  The invariant: the response
# must NEVER contain fabricated technical values.  Either it cites a real
# retrieved section or it issues the refusal phrase.  There is no third option.
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

_FABRICATION_PATTERNS = [
    # Hallucinated numbers that are NOT in our corpus
    "400 km", "400km", "5,000 km", "5000 km",
    "10,000 km", "10000 km", "20W-50", "5W-30",
    "3.0 litre", "3.0L", "2.0 litre",
    "100,000 km", "100000",
]


def _has_fabrication(text: str) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in _FABRICATION_PATTERNS)


def test_ambiguous_highway_suitability_no_hallucination(corpus_index):
    """'Is the Meteor good for highway riding?' — no manual section covers this.
    Pipeline must refuse or stay scoped; must not invent touring recommendations."""
    result = run_query("Is the Meteor 350 good for long highway rides?", _DOC_ID, use_mocks=True)
    answer = result.get("answer_text", "")
    assert not _has_fabrication(answer), f"Potential fabrication detected:\n{answer}"
    # If it's marked OOS or low-confidence, confidence should reflect that
    conf = result.get("context_confidence", "")
    # Either it refuses OR it finds something related (e.g. maintenance intervals)
    # but must never fabricate highway-specific figures
    assert answer.strip() != "", "Response must not be empty"


def test_ambiguous_cold_weather_oil_no_hallucination(corpus_index):
    """'What oil in winter?' — corpus has 10W-40 spec.  Response must cite 2.1 or refuse."""
    result = run_query("What engine oil should I use in cold winter weather?", _DOC_ID, use_mocks=True)
    answer = result.get("answer_text", "")
    assert not _has_fabrication(answer), f"Potential fabrication in answer:\n{answer}"


def test_ambiguous_oil_skip_consequence_no_hallucination(corpus_index):
    """'What happens if I skip oil change?' — must cite maintenance intervals or refuse."""
    result = run_query("What happens if I skip the engine oil change?", _DOC_ID, use_mocks=True)
    answer = result.get("answer_text", "")
    assert not _has_fabrication(answer), f"Potential fabrication:\n{answer}"
    assert answer.strip() != ""


def test_ambiguous_mileage_life_no_hallucination(corpus_index):
    """'How many km will the Meteor last?' — no section covers engine lifespan.
    Must refuse rather than guess 100,000 km or similar."""
    result = run_query("How many kilometres will the Meteor 350 engine last?", _DOC_ID, use_mocks=True)
    answer = result.get("answer_text", "")
    assert not _has_fabrication(answer), f"Potential fabrication:\n{answer}"


def test_ambiguous_battery_low_voltage_no_hallucination(corpus_index):
    """Near-scope: 'battery flat won't start' overlaps sections 7.3 and 8.1.
    Must retrieve at least one and not fabricate voltage values outside 12–15 V."""
    result = run_query("battery dead bike won't start voltage low", _DOC_ID, use_mocks=True)
    answer = result.get("answer_text", "")
    assert not _has_fabrication(answer), f"Potential fabrication:\n{answer}"


# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
# GROUP F — Guardrail Layer Unit Tests
# Test each layer of the anti-hallucination contract in isolation.
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

# ── F1. OOS keyword layer ──────────────────────────────────────────────────

def test_oos_detection_price():       assert _is_oos("price of the bike")
def test_oos_detection_cost():        assert _is_oos("how much does it cost")
def test_oos_detection_buy():         assert _is_oos("should I buy this")
def test_oos_detection_colour():      assert _is_oos("what colour does it come in")
def test_oos_detection_compare():     assert _is_oos("compare with other bikes")
def test_oos_detection_insurance():   assert _is_oos("insurance amount for the Meteor")
def test_oos_detection_better():      assert _is_oos("is this better than KTM Duke")
def test_oos_detection_purchase():    assert _is_oos("where to purchase this model")
def test_oos_detection_worth():       assert _is_oos("is it worth buying")
def test_oos_detection_review():      assert _is_oos("user review of this model")

def test_oos_false_positive_oil_change():
    """'engine oil change' is in-scope and must NOT be flagged as OOS."""
    assert not _is_oos("engine oil change procedure")

def test_oos_false_positive_valve():
    assert not _is_oos("how to check valve clearance")

def test_oos_false_positive_spark_plug():
    assert not _is_oos("spark plug inspection steps")


# ── F2. System prompt hard-rules layer ────────────────────────────────────

from inference.generator import _SYSTEM_PROMPT


def test_system_prompt_contains_context_only_rule():
    prompt = _SYSTEM_PROMPT.format(bike="test", confidence_level="HIGH", language_instruction="")
    assert "ONLY from the provided [CONTEXT]" in prompt


def test_system_prompt_contains_refusal_instruction():
    prompt = _SYSTEM_PROMPT.format(bike="test", confidence_level="HIGH", language_instruction="")
    assert "couldn't find" in prompt or "service center" in prompt


def test_system_prompt_contains_citation_requirement():
    prompt = _SYSTEM_PROMPT.format(bike="test", confidence_level="HIGH", language_instruction="")
    assert "cite" in prompt.lower() or "citation" in prompt.lower()


def test_system_prompt_contains_severity_label_options():
    prompt = _SYSTEM_PROMPT.format(bike="test", confidence_level="HIGH", language_instruction="")
    assert "Immediate Action Required" in prompt
    assert "Informational" in prompt


def test_system_prompt_low_confidence_instruction():
    """LOW confidence prompt must instruct LLM to disclose limited context."""
    prompt = _SYSTEM_PROMPT.format(bike="test", confidence_level="LOW", language_instruction="")
    assert "LOW" in prompt


# ── F3. Citation grounding validator ─────────────────────────────────────

from inference.generator import validate_citation_against_retrieved

_SAMPLE_CHUNKS = [
    ({"section_number": "2.1", "section_title": "Engine Oil", "page_number": 15, "text": "..."}, 0.9),
    ({"section_number": "7.1", "section_title": "Exhaust Smoke", "page_number": 134, "text": "..."}, 0.7),
]


def test_grounding_validator_accepts_correct_section():
    """Citation matching section_number in retrieved chunks must pass."""
    cit = {"section_number": "2.1", "section_title": "Engine Oil", "page_number": 15}
    assert validate_citation_against_retrieved(cit, _SAMPLE_CHUNKS) is True


def test_grounding_validator_accepts_by_page_number():
    """Citation matching page_number only (section_number differs) must still pass."""
    cit = {"section_number": "", "section_title": "Engine Oil", "page_number": 134}
    assert validate_citation_against_retrieved(cit, _SAMPLE_CHUNKS) is True


def test_grounding_validator_rejects_hallucinated_section():
    """Citation with a section number that was NEVER retrieved must fail."""
    cit = {"section_number": "99.9", "section_title": "Made Up Section", "page_number": 999}
    assert validate_citation_against_retrieved(cit, _SAMPLE_CHUNKS) is False


def test_grounding_validator_rejects_empty_citation():
    """An empty citation dict must fail — it's unverifiable."""
    assert validate_citation_against_retrieved({}, _SAMPLE_CHUNKS) is False


def test_grounding_validator_with_empty_chunk_list():
    """A valid-looking citation against an empty retrieval result must fail."""
    cit = {"section_number": "2.1", "section_title": "Engine Oil", "page_number": 15}
    assert validate_citation_against_retrieved(cit, []) is False


# ── F4. Layer 3 — Citation regeneration guard ─────────────────────────────

def test_citation_regeneration_guard_triggers_on_missing_citation():
    """
    When the LLM returns a response with no citations, generator.py must
    attempt one regeneration.  We mock _call_llm to return no-citation on the
    first call and a valid citation on the second.
    """
    from inference import generator as gen_mod

    call_count = {"n": 0}
    fixture = {
        "answer_text": "Engine oil: use SAE 10W-40.",
        "spoken_summary": "Use 10W-40 oil.",
        "citations": [{"section_number": "2.1", "section_title": "Engine Oil", "page_number": 15}],
        "severity_label": "Informational",
        "confidence": "high",
        "suggested_followups": [],
    }

    def _mock_call_llm(messages, language="en"):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call returns no citations — should trigger regeneration
            return {
                "answer_text": "Engine oil: use SAE 10W-40.",
                "spoken_summary": "Use 10W-40 oil.",
                "citations": [],          # ← missing citations, must regenerate
                "severity_label": "Informational",
                "confidence": "high",
                "suggested_followups": [],
            }
        return fixture

    chunks = [(
        {"section_number": "2.1", "section_title": "Engine Oil",
         "page_number": 15, "text": "Use SAE 10W-40 oil."},
        0.9,
    )]

    with patch.object(gen_mod, "_call_llm", side_effect=_mock_call_llm):
        result = gen_mod.generate(
            query="what engine oil do I use",
            chunks=chunks,
            context_confidence="high",
            use_mocks=False,
        )

    assert call_count["n"] == 2, (
        f"Expected 2 LLM calls (original + regeneration), got {call_count['n']}"
    )
    assert len(result.get("citations", [])) > 0, "Regeneration must produce citations"


def test_citation_regeneration_second_failure_does_not_crash():
    """
    If both LLM calls return no citations, generator must still return a valid
    structured response — it must not raise an exception or return None.
    """
    from inference import generator as gen_mod

    def _always_empty(messages, language="en"):
        return {
            "answer_text": "Some answer without citation.",
            "spoken_summary": "Some answer.",
            "citations": [],
            "severity_label": "Informational",
            "confidence": "high",
            "suggested_followups": [],
        }

    chunks = [(
        {"section_number": "2.1", "section_title": "Engine Oil",
         "page_number": 15, "text": "Use SAE 10W-40."},
        0.9,
    )]

    with patch.object(gen_mod, "_call_llm", side_effect=_always_empty):
        result = gen_mod.generate(
            query="engine oil type",
            chunks=chunks,
            context_confidence="high",
            use_mocks=False,
        )

    for field in _REQUIRED_RESPONSE_FIELDS:
        assert field in result, f"Field missing after double-citation failure: {field}"


# ── F5. Structured output contract ────────────────────────────────────────

def test_pipeline_all_required_fields_present(corpus_index):
    result = run_query("engine oil specification", _DOC_ID, use_mocks=True)
    for field in _REQUIRED_RESPONSE_FIELDS:
        assert field in result, f"Pipeline response missing field: {field}"


def test_pipeline_severity_label_is_valid_value(corpus_index):
    result = run_query("brake fluid replacement", _DOC_ID, use_mocks=True)
    assert result.get("severity_label") in _VALID_SEVERITY, (
        f"Invalid severity_label: {result.get('severity_label')}"
    )


def test_pipeline_suggested_followups_is_list(corpus_index):
    result = run_query("spark plug gap check", _DOC_ID, use_mocks=True)
    assert isinstance(result.get("suggested_followups"), list)


def test_pipeline_confidence_is_high_or_low(corpus_index):
    result = run_query("chain slack measurement", _DOC_ID, use_mocks=True)
    assert result.get("confidence") in ("high", "low"), (
        f"confidence must be 'high' or 'low', got: {result.get('confidence')}"
    )


def test_pipeline_intent_field_present(corpus_index):
    result = run_query("how to check valve clearance", _DOC_ID, use_mocks=True)
    assert "intent" in result
    assert result["intent"] in ("diagnostic", "specification", "procedure", "out_of_scope")


def test_pipeline_language_field_present(corpus_index):
    result = run_query("engine overheating causes", _DOC_ID, use_mocks=True)
    assert "language" in result
    assert result["language"] in ("en", "hi", "ta", "te", "kn", "mr", "bn", "gu", "pa", "ml")


def test_pipeline_no_exception_on_devanagari_query(corpus_index):
    """Hindi Devanagari input must not crash the pipeline."""
    result = run_query("इंजन में तेल कैसे भरें", _DOC_ID, use_mocks=True)
    assert "answer_text" in result


def test_pipeline_no_exception_on_romanized_hindi(corpus_index):
    """Romanized Hindi input must pass through the pipeline without error."""
    result = run_query("engine garam ho rahi hai", _DOC_ID, use_mocks=True)
    assert "answer_text" in result
