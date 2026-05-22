"""
Chunking and metadata tagging.
Applies content-type-specific chunking rules and the full metadata schema from AGENTS.md.
"""
import re
import uuid
from typing import Any, Dict, List, Optional

from .classifier import ContentType

# Prose chunking parameters
_PROSE_MAX_CHARS = 2400   # ~600 tokens at 4 chars/token
_PROSE_MIN_CHARS = 1600   # ~400 tokens
_PROSE_OVERLAP_CHARS = 360  # ~90 tokens

_STEP_RE = re.compile(r'(?=^\s*\d+[\.\)]\s+)', re.MULTILINE)
# Splits inline "1. text 2. text" by lookahead on step number
_INLINE_SPLIT_RE = re.compile(r'(?=\s+\d+[\.\)]\s+)')
_TOOLS_RE = re.compile(r'\b(wrench|socket|torque wrench|screwdriver|pliers|hex key|spanner)\b', re.IGNORECASE)
_SEVERITY_RE = re.compile(r'^\s*(WARNING|CAUTION|NOTE)', re.IGNORECASE)


def _new_id() -> str:
    return str(uuid.uuid4())


def _base_meta(block: Dict, doc_meta: Dict, document_id: str) -> Dict:
    return {
        "bike_brand": doc_meta.get("bike_brand", ""),
        "bike_model": doc_meta.get("bike_model", ""),
        "bike_year": doc_meta.get("bike_year", ""),
        "manual_type": doc_meta.get("manual_type", ""),
        "manual_source": doc_meta.get("manual_source", "library"),
        "document_id": document_id,
        "chapter_number": block.get("chapter_number", ""),
        "chapter_title": block.get("chapter_title", ""),
        "section_number": block.get("section_number", ""),
        "section_title": block.get("section_title", ""),
        "page_number": block.get("page_number", 0),
    }


def _chunk_prose(blocks: List[Dict], doc_meta: Dict, document_id: str) -> List[Dict]:
    chunks = []
    buffer = ""
    buffer_meta = None

    def flush(buf: str, meta: Dict) -> None:
        if buf.strip():
            chunks.append({
                **meta,
                "chunk_id": _new_id(),
                "content_type": ContentType.PROSE,
                "text": buf.strip(),
                "parent_chunk_id": None,
                "is_parent": False,
                "table_title": None, "spec_unit": None,
                "image_path": None, "diagram_type": None,
                "severity": None, "related_procedure": None,
                "procedure_step_count": None, "tools_required": None,
            })

    for block in blocks:
        text = block["text"]
        meta = _base_meta(block, doc_meta, document_id)

        if not buffer:
            buffer = text
            buffer_meta = meta
            continue

        if len(buffer) + len(text) + 1 <= _PROSE_MAX_CHARS:
            buffer += " " + text
        else:
            flush(buffer, buffer_meta)
            # Overlap: carry last _PROSE_OVERLAP_CHARS of previous buffer
            overlap = buffer[-_PROSE_OVERLAP_CHARS:] if len(buffer) > _PROSE_OVERLAP_CHARS else buffer
            buffer = overlap + " " + text
            buffer_meta = meta

    if buffer:
        flush(buffer, buffer_meta)

    return chunks


def _chunk_warning(block: Dict, doc_meta: Dict, document_id: str) -> List[Dict]:
    text = block["text"].strip()
    m = _SEVERITY_RE.match(text)
    severity = m.group(1).upper() if m else "NOTE"
    return [{
        **_base_meta(block, doc_meta, document_id),
        "chunk_id": _new_id(),
        "content_type": ContentType.WARNING,
        "text": text,
        "parent_chunk_id": None,
        "is_parent": False,
        "table_title": None, "spec_unit": None,
        "image_path": None, "diagram_type": None,
        "severity": severity,
        "related_procedure": block.get("section_number", ""),
        "procedure_step_count": None,
        "tools_required": None,
    }]


def _chunk_procedure(block: Dict, doc_meta: Dict, document_id: str) -> List[Dict]:
    text = block["text"].strip()
    # Try newline-separated split first, fall back to inline split
    steps = [s.strip() for s in _STEP_RE.split(text) if s.strip()]
    if len(steps) < 3:
        steps = [s.strip() for s in _INLINE_SPLIT_RE.split(text) if s.strip()]
    if len(steps) < 3:
        steps = [text]

    parent_id = _new_id()
    tools = _TOOLS_RE.findall(text)
    base = _base_meta(block, doc_meta, document_id)

    parent = {
        **base,
        "chunk_id": parent_id,
        "content_type": ContentType.PROCEDURE,
        "text": text,
        "parent_chunk_id": None,
        "is_parent": True,
        "table_title": None, "spec_unit": None,
        "image_path": None, "diagram_type": None,
        "severity": None, "related_procedure": None,
        "procedure_step_count": len(steps),
        "tools_required": list(set(t.lower() for t in tools)),
    }

    children = []
    for step in steps:
        children.append({
            **base,
            "chunk_id": _new_id(),
            "content_type": ContentType.PROCEDURE,
            "text": step,
            "parent_chunk_id": parent_id,
            "is_parent": False,
            "table_title": None, "spec_unit": None,
            "image_path": None, "diagram_type": None,
            "severity": None, "related_procedure": None,
            "procedure_step_count": None,
            "tools_required": None,
        })

    return [parent] + children


def _chunk_specification(block: Dict, doc_meta: Dict, document_id: str) -> List[Dict]:
    table = block.get("table_data", {})
    rows = table.get("rows", [])
    base = _base_meta(block, doc_meta, document_id)
    chunks = []

    for row in rows:
        component = row.get("component", "")
        # Find value field (anything that isn't "component" or "notes")
        value = next((v for k, v in row.items() if k not in ("component", "notes") and v), "")
        notes = row.get("notes", "")
        text = f"{component}: {value}" + (f" ({notes})" if notes else "")
        # Detect unit from value (Nm, mm, rpm, L)
        unit_m = re.search(r'\b(Nm|mm|rpm|L|kPa|MPa|bar)\b', value)
        chunks.append({
            **base,
            "chunk_id": _new_id(),
            "content_type": ContentType.SPECIFICATION,
            "text": text,
            "parent_chunk_id": None,
            "is_parent": False,
            "table_title": table.get("table_title", ""),
            "spec_unit": unit_m.group(1) if unit_m else None,
            "image_path": None, "diagram_type": None,
            "severity": None, "related_procedure": None,
            "procedure_step_count": None, "tools_required": None,
        })

    # If no rows, store the whole table text as one chunk
    if not chunks:
        chunks.append({
            **base,
            "chunk_id": _new_id(),
            "content_type": ContentType.SPECIFICATION,
            "text": block["text"],
            "parent_chunk_id": None,
            "is_parent": False,
            "table_title": table.get("table_title", ""),
            "spec_unit": None,
            "image_path": None, "diagram_type": None,
            "severity": None, "related_procedure": None,
            "procedure_step_count": None, "tools_required": None,
        })

    return chunks


def _chunk_image(block: Dict, doc_meta: Dict, document_id: str) -> List[Dict]:
    return [{
        **_base_meta(block, doc_meta, document_id),
        "chunk_id": _new_id(),
        "content_type": ContentType.IMAGE,
        "text": block["text"],
        "parent_chunk_id": None,
        "is_parent": False,
        "table_title": None, "spec_unit": None,
        "image_path": block.get("image_path", ""),
        "diagram_type": None,
        "severity": None, "related_procedure": None,
        "procedure_step_count": None, "tools_required": None,
    }]


def chunk_blocks(blocks: List[Dict], doc_meta: Dict, document_id: str) -> List[Dict]:
    """
    Apply content-type-specific chunking rules to parsed blocks.
    Returns a flat list of chunks with the full metadata schema.
    """
    prose_buf: List[Dict] = []
    chunks: List[Dict] = []

    def flush_prose():
        if prose_buf:
            chunks.extend(_chunk_prose(prose_buf, doc_meta, document_id))
            prose_buf.clear()

    for block in blocks:
        ct = block["content_type"]

        if ct == ContentType.PROSE:
            prose_buf.append(block)
        else:
            flush_prose()
            if ct == ContentType.WARNING:
                chunks.extend(_chunk_warning(block, doc_meta, document_id))
            elif ct == ContentType.PROCEDURE:
                chunks.extend(_chunk_procedure(block, doc_meta, document_id))
            elif ct == ContentType.SPECIFICATION:
                chunks.extend(_chunk_specification(block, doc_meta, document_id))
            elif ct == ContentType.IMAGE:
                chunks.extend(_chunk_image(block, doc_meta, document_id))

    flush_prose()
    return chunks
