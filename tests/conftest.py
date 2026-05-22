"""
Session-level fixture file generation.
Creates minimal valid audio and image files for Part 6 tests and local curl testing.
Binary files are generated at test time — not committed to git.
"""
import io
import struct
import wave
import zlib
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_wav() -> bytes:
    """Minimal valid WAV: 100 samples of silence at 16kHz mono 16-bit."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 250)
    return buf.getvalue()


def _make_png() -> bytes:
    """Minimal valid PNG: 1x1 white pixel."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xFF"))
    iend = chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def pytest_configure(config):
    """Generate binary fixture files before any test runs."""
    FIXTURES_DIR.mkdir(exist_ok=True)

    for name in ("sample_hindi.wav", "sample_english.wav"):
        p = FIXTURES_DIR / name
        if not p.exists():
            p.write_bytes(_make_wav())

    exhaust = FIXTURES_DIR / "sample_exhaust.jpg"
    if not exhaust.exists():
        # PNG bytes stored with .jpg extension — mock doesn't validate content
        exhaust.write_bytes(_make_png())
