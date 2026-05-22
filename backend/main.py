import sys
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return JSONResponse(
        status_code=422,
        content={
            "error": True,
            "message": "Request validation failed",
            "code": "VALIDATION_ERROR",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": True, "message": str(exc.detail), "code": "NOT_FOUND"},
    )


@app.exception_handler(400)
async def bad_request_handler(request: Request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": True, "message": str(exc.detail), "code": "BAD_REQUEST"},
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
