"""
Content type classification pass over text blocks.
Labels each block before any chunking occurs. Classification drives chunking strategy.
"""
import re
from enum import Enum

class ContentType(str, Enum):
    PROSE = "prose"
    SPECIFICATION = "specification"
    IMAGE = "image"
    WARNING = "warning"
    PROCEDURE = "procedure"

_WARNING_RE = re.compile(r'^\s*(WARNING|CAUTION|NOTE)[:\s]', re.IGNORECASE)
# Newline-separated steps: "1. Do X\n2. Do Y\n3. Do Z"
_STEP_RE = re.compile(r'^\s*\d+[\.\)]\s+\w', re.MULTILINE)
# Inline steps: "1. Do X 2. Do Y 3. Do Z" (PyMuPDF sometimes collapses newlines)
_INLINE_STEP_RE = re.compile(r'\b1[\.\)]\s+\w.{5,}\s+2[\.\)]\s+\w.{5,}\s+3[\.\)]\s+\w')
# Matches headings like:
# "1 INTRODUCTION", "2.3 Location of Key Parts", "3-1 Engine Oil"
_HEADING_RE = re.compile(
    r'^(\d+(?:[.\-]\d+)*)\s+([A-Za-z][A-Za-z0-9\s\-/,&()]{2,})\s*$'
)


def classify_text(text: str) -> ContentType:
    if _WARNING_RE.match(text.strip()):
        return ContentType.WARNING
    if len(_STEP_RE.findall(text)) >= 3 or _INLINE_STEP_RE.search(text):
        return ContentType.PROCEDURE
    return ContentType.PROSE


def parse_heading(text: str):
    """Return (section_number, title) if text looks like a section heading, else (None, None)."""
    m = _HEADING_RE.match(text.strip())
    if m:
        title = re.sub(r"\s+", " ", m.group(2)).strip()
        return m.group(1), title
    return None, None
