"""
POST /ingest — accepts PDF upload, runs ingestion pipeline as a background task.
POST /ingest/extract-meta — reads first 3 pages of a PDF and returns extracted bike metadata.
GET /ingest/status/{job_id} — returns job status and progress.
"""
import json
import logging
import os
import re
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

import store

logger = logging.getLogger(__name__)

router = APIRouter()

_ROOT = Path(__file__).parent.parent.parent
_MANUALS_DIR = _ROOT / "data" / "manuals"

# Known brand names for mock/fallback extraction
_BRAND_PATTERNS = {
    "royal enfield": "Royal Enfield",
    "yamaha": "Yamaha",
    "honda": "Honda",
    "bajaj": "Bajaj",
    "tvs": "TVS",
    "hero": "Hero",
    "suzuki": "Suzuki",
    "kawasaki": "Kawasaki",
    "ktm": "KTM",
    "triumph": "Triumph",
}

_EXTRACT_SYSTEM = """\
You are given the title page and first few pages of a motorcycle service manual.
Extract these fields from the text:
- bike_brand: Manufacturer name (e.g. "Royal Enfield", "Honda", "Yamaha", "Bajaj")
- bike_model: Specific model name (e.g. "Meteor 350", "CB300R", "R15 V4")
- bike_year: 4-digit model year string (e.g. "2022"). If a range, use the latest year.
- manual_type: exactly one of "service_manual", "owner_manual", or "user_guide"

Return ONLY valid JSON: {"bike_brand": "...", "bike_model": "...", "bike_year": "...", "manual_type": "..."}
Use "" for any field you cannot confidently determine. Never guess.\
"""


def _read_first_pages(pdf_bytes: bytes, max_pages: int = 3) -> str:
    """Extract plain text from the first max_pages pages using PyMuPDF."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        parts.append(page.get_text())
    doc.close()
    return "\n".join(parts)[:4000]


def _mock_extract(text: str) -> dict:
    """Regex-based extraction used in mock mode — no LLM call."""
    text_lower = text.lower()
    brand = next(
        (name for pattern, name in _BRAND_PATTERNS.items() if pattern in text_lower),
        "",
    )
    years = re.findall(r"\b(20\d{2}|19\d{2})\b", text)
    year = years[0] if years else ""
    if "owner" in text_lower:
        manual_type = "owner_manual"
    elif "user guide" in text_lower:
        manual_type = "user_guide"
    else:
        manual_type = "service_manual"
    return {"bike_brand": brand, "bike_model": "", "bike_year": year,
            "manual_type": manual_type, "confidence": "low"}


def _llm_extract(text: str) -> dict:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": f"[MANUAL TEXT]\n{text}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    result = json.loads(resp.choices[0].message.content)
    result.setdefault("bike_brand", "")
    result.setdefault("bike_model", "")
    result.setdefault("bike_year", "")
    result.setdefault("manual_type", "service_manual")
    result["confidence"] = "high"
    return result


def _run_ingest(job_id: str, pdf_path: str, doc_meta: dict) -> None:
    use_mocks = os.getenv("USE_MOCKS", "false").lower() == "true"
    try:
        store.update_job(job_id, status="processing", progress_pct=10, message="Reading your PDF...")

        from ingestion.parser import parse_pdf
        parsed = parse_pdf(pdf_path, doc_meta, use_mocks=use_mocks)
        document_id = parsed["document_id"]

        store.update_job(job_id, status="processing", progress_pct=35, message="Splitting into sections...")

        from ingestion.chunker import chunk_blocks
        chunks = chunk_blocks(parsed["blocks"], doc_meta, document_id)

        store.update_job(job_id, status="processing", progress_pct=60, message="Analysing content...")

        from ingestion.embedder import embed_chunks
        embedded = embed_chunks(list(chunks), use_mocks=use_mocks)

        store.update_job(job_id, status="processing", progress_pct=85, message="Building search index...")

        from ingestion.indexer import build_indexes
        build_indexes(embedded, document_id)

        from ingestion.document_index import generate_document_index
        generate_document_index(document_id, doc_meta, embedded)

        store.update_job(
            job_id,
            status="complete",
            progress_pct=100,
            message="Manual ready!",
            document_id=document_id,
        )
    except Exception as exc:
        store.update_job(job_id, status="error", progress_pct=0, message=str(exc))


@router.post("/ingest/extract-meta")
async def extract_meta(file: UploadFile = File(...)):
    """
    Read the first 3 pages of a PDF and return extracted bike metadata.
    Fast — does not start a full ingestion job.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    use_mocks = os.getenv("USE_MOCKS", "false").lower() == "true"
    try:
        pdf_bytes = await file.read()
        text = _read_first_pages(pdf_bytes)
        if use_mocks or not os.getenv("OPENROUTER_API_KEY", "").strip():
            return _mock_extract(text)
        return _llm_extract(text)
    except Exception as exc:
        logger.warning("extract-meta failed: %s", exc)
        return {"bike_brand": "", "bike_model": "", "bike_year": "",
                "manual_type": "service_manual", "confidence": "error"}


@router.post("/ingest", status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    brand: str = Form(...),
    model: str = Form(...),
    year: str = Form(...),
    manual_type: str = Form("service_manual"),
    save_to_library: bool = Form(False),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    _MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    contents = await file.read()

    safe_name = Path(file.filename).stem[:60].replace(" ", "_") + ".pdf"
    pdf_path = _MANUALS_DIR / safe_name
    pdf_path.write_bytes(contents)

    doc_meta = {
        "bike_brand": brand,
        "bike_model": model,
        "bike_year": year,
        "manual_type": manual_type,
        "manual_source": "library" if save_to_library else "user_uploaded",
    }

    job_id = store.create_job()
    background_tasks.add_task(_run_ingest, job_id, str(pdf_path), doc_meta)
    return {"job_id": job_id}


@router.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found", headers={"X-Code": "JOB_NOT_FOUND"})
    return job
