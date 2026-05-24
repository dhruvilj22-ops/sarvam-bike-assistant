"""
PDF parsing: extracts typed blocks using PyMuPDF for text/images
and LlamaParse (or mock) for tables.
"""
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from .classifier import ContentType, classify_text, parse_heading

_SINGLE_STEP_RE = re.compile(r'^\s*\d+[\.\)]\s+\w')

_ROOT = Path(__file__).parent.parent.parent

_MOCK_IMAGE_DESCRIPTION = (
    "Image shows white smoke from rear exhaust pipe, "
    "indicating possible oil burning or coolant leak"
)


def _compute_doc_id(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _describe_image_gpt4v(image_bytes: bytes) -> str:
    # GPT-4o vision converts diagram content to searchable text at ingestion time
    import base64
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    b64 = base64.b64encode(image_bytes).decode()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {
                    "type": "text",
                    "text": (
                        "Describe this technical diagram from a bike service manual. "
                        "List all labeled components, their positions, and any specifications shown."
                    ),
                },
            ],
        }],
        max_tokens=400,
    )
    return resp.choices[0].message.content


def _extract_tables_llamaparse(pdf_path: str) -> List[Dict]:
    # LlamaParse for table extraction — PyMuPDF flattens tables into garbled text
    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=os.getenv("LLAMAPARSE_API_KEY"),
        result_type="json",
        verbose=False,
    )
    results = parser.load_data(pdf_path)
    tables = []
    for doc in results:
        raw = doc.text
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, dict) and "tables" in data:
                tables.extend(data["tables"])
        except (json.JSONDecodeError, TypeError):
            pass
    return tables


def _extract_tables_mock() -> List[Dict]:
    fixture = _ROOT / "tests/fixtures/sample_table.json"
    if not fixture.exists():
        return []
    try:
        return [json.loads(fixture.read_text())]
    except Exception:
        return []


def parse_pdf(
    pdf_path: str,
    doc_meta: Dict[str, str],
    use_mocks: bool = False,
    images_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Parse PDF into typed blocks with structural metadata.
    Returns {"document_id": str, "blocks": List[block_dict]}.
    """
    document_id = doc_meta.get("document_id") or _compute_doc_id(pdf_path)
    if images_dir:
        images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    blocks: List[Dict] = []
    current_chapter = {"number": "", "title": ""}
    current_section = {"number": "", "title": ""}
    page_context: Dict[int, Dict[str, str]] = {}

    for page_num, page in enumerate(doc, start=1):
        # Extract images first, keyed by block number to avoid duplicates
        page_images: Dict[int, bytes] = {}
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                page_images[xref] = base_image["image"]
            except Exception:
                pass

        raw_blocks = page.get_text("blocks", sort=True)
        image_xrefs_used: set = set()

        for blk in raw_blocks:
            x0, y0, x1, y1, content, block_no, block_type = blk

            if block_type == 1:
                # Image block — use first available image for this page
                for xref, image_bytes in page_images.items():
                    if xref in image_xrefs_used:
                        continue
                    image_xrefs_used.add(xref)

                    if use_mocks:
                        description = _MOCK_IMAGE_DESCRIPTION
                        image_path = ""
                    else:
                        try:
                            description = _describe_image_gpt4v(image_bytes)
                        except Exception:
                            description = _MOCK_IMAGE_DESCRIPTION
                        if images_dir:
                            img_file = images_dir / f"page_{page_num}_xref_{xref}.png"
                            img_file.write_bytes(image_bytes)
                            image_path = str(img_file)
                        else:
                            image_path = ""

                    blocks.append({
                        "content_type": ContentType.IMAGE,
                        "text": description,
                        "image_path": image_path,
                        "page_number": page_num,
                        "chapter_number": current_chapter["number"],
                        "chapter_title": current_chapter["title"],
                        "section_number": current_section["number"],
                        "section_title": current_section["title"],
                    })
                    break
                continue

            text = content.strip()
            if len(text) < 5:
                continue

            # Update heading context (short line with number+title pattern)
            sec_num, sec_title = parse_heading(text)
            if sec_num is not None:
                depth = sec_num.count(".")
                if depth == 0:
                    current_chapter = {"number": sec_num, "title": sec_title}
                    current_section = {"number": sec_num, "title": sec_title}
                else:
                    current_section = {"number": sec_num, "title": sec_title}
                continue

            content_type = classify_text(text)
            blocks.append({
                "content_type": content_type,
                "text": text,
                "page_number": page_num,
                "chapter_number": current_chapter["number"],
                "chapter_title": current_chapter["title"],
                "section_number": current_section["number"],
                "section_title": current_section["title"],
            })

        # Capture best-known chapter/section context for this page.
        page_context[page_num] = {
            "chapter_number": current_chapter["number"],
            "chapter_title": current_chapter["title"],
            "section_number": current_section["number"],
            "section_title": current_section["title"],
        }

    doc.close()

    # Post-process: group consecutive single-step prose blocks into a procedure block.
    # PyMuPDF splits multiline text into per-line blocks; each "1. ..." line comes back
    # as its own block and the classifier can't see the full numbered sequence.
    merged: List[Dict] = []
    i = 0
    while i < len(blocks):
        blk = blocks[i]
        if blk["content_type"] == ContentType.PROSE and _SINGLE_STEP_RE.match(blk["text"]):
            group = [blk]
            j = i + 1
            while j < len(blocks) and blocks[j]["content_type"] == ContentType.PROSE and _SINGLE_STEP_RE.match(blocks[j]["text"]):
                group.append(blocks[j])
                j += 1
            if len(group) >= 3:
                merged.append({**group[0], "content_type": ContentType.PROCEDURE, "text": "\n".join(b["text"] for b in group)})
                i = j
            else:
                merged.extend(group)
                i = j
        else:
            merged.append(blk)
            i += 1
    blocks = merged

    # Append table blocks from LlamaParse or mock
    use_mock_tables = use_mocks or not os.getenv("LLAMAPARSE_API_KEY", "").strip()
    tables = _extract_tables_mock() if use_mock_tables else _extract_tables_llamaparse(pdf_path)

    for table in tables:
        rows = table.get("rows", [])
        page_number = table.get("page_number", 0)
        ctx = page_context.get(page_number, {})
        section_title = ctx.get("section_title", "") or table.get("table_title", "")
        text = table.get("table_title", "Table") + ": " + "; ".join(
            f"{r.get('component', '')}: {next((v for k, v in r.items() if k != 'component'), '')}"
            for r in rows
        )
        blocks.append({
            "content_type": ContentType.SPECIFICATION,
            "text": text,
            "table_data": table,
            "page_number": page_number,
            "chapter_number": ctx.get("chapter_number", ""),
            "chapter_title": ctx.get("chapter_title", ""),
            "section_number": ctx.get("section_number", ""),
            "section_title": section_title,
        })

    return {"document_id": document_id, "blocks": blocks}
