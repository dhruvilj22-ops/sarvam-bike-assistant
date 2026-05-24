import os
import sys
import logging
import time
import uuid
from pathlib import Path

# Ensure backend/ is on the path so ingestion.* and inference.* imports resolve
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from routes.bikes import router as bikes_router
from routes.ingest import router as ingest_router
from routes.input import router as input_router
from routes.output import router as output_router
from routes.query import router as query_router
from routes.session import router as session_router

app = FastAPI(title="Bike Troubleshooting Assistant")
logger = logging.getLogger("bike-assistant")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
if _extra := os.getenv("ALLOWED_ORIGINS", ""):
    _ALLOWED_ORIGINS.extend(o.strip() for o in _extra.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "request_failed trace_id=%s method=%s path=%s duration_ms=%s",
            trace_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = int((time.perf_counter() - start) * 1000)
    response.headers["x-trace-id"] = trace_id
    logger.info(
        "request trace_id=%s method=%s path=%s status=%s duration_ms=%s",
        trace_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

# ---------------------------------------------------------------------------
# Health — registered first so wildcard SPA handler never shadows it
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------

app.include_router(query_router)
app.include_router(ingest_router)
app.include_router(bikes_router)
app.include_router(session_router)
app.include_router(input_router)
app.include_router(output_router)


# ---------------------------------------------------------------------------
# Consistent error response shape: {error, message, code}
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    trace_id = request.headers.get("x-trace-id", "")
    logger.warning("validation_error trace_id=%s path=%s detail=%s", trace_id, request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Request validation failed",
            "code": "VALIDATION_ERROR",
            "detail": exc.errors(),
            "trace_id": trace_id,
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    trace_id = request.headers.get("x-trace-id", "")
    logger.warning("not_found trace_id=%s path=%s detail=%s", trace_id, request.url.path, getattr(exc, "detail", ""))
    return JSONResponse(
        status_code=404,
        content={"error": True, "message": str(exc.detail), "code": "NOT_FOUND", "trace_id": trace_id},
    )


@app.exception_handler(400)
async def bad_request_handler(request: Request, exc):
    trace_id = request.headers.get("x-trace-id", "")
    logger.warning("bad_request trace_id=%s path=%s detail=%s", trace_id, request.url.path, getattr(exc, "detail", ""))
    return JSONResponse(
        status_code=400,
        content={"error": True, "message": str(exc.detail), "code": "BAD_REQUEST", "trace_id": trace_id},
    )


# ---------------------------------------------------------------------------
# Next.js static build — mounted last so all API routes take priority
# ---------------------------------------------------------------------------

_FRONTEND = Path(__file__).parent.parent / "frontend" / "out"
if _FRONTEND.exists():
    app.mount("/_next", StaticFiles(directory=str(_FRONTEND / "_next")), name="nextjs_assets")

    if (_FRONTEND / "chat").exists():
        app.mount("/chat", StaticFiles(directory=str(_FRONTEND / "chat"), html=True), name="chat")

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(str(_FRONTEND / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        candidate = _FRONTEND / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_FRONTEND / "index.html"))
