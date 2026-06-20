"""
Gol D Lyric — Backend API
FastAPI + Hybrid IR Engine (TF-IDF VSM + Boolean Retrieval)

Startup:  python -m uvicorn main:app --reload --port 8000
"""

import time
import os
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.search_engine import SearchEngine

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)

# ─── Global engine instance (pre-computed at startup) ──────────────────────

engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Build the TF-IDF index once at startup so every request
    hits the pre-computed matrix without re-fitting.
    """
    global engine
    logger.info("⚙️  Building TF-IDF index …")
    engine = SearchEngine(data_path="data/songs.json")
    engine.build_index()
    logger.info(f"✅  Index ready — {len(engine.songs)} songs loaded.")
    yield
    # cleanup (nothing needed for in-memory index)
    logger.info("🛑  Shutting down.")


# ─── App factory ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gol D Lyric API",
    version="1.0.0",
    description="Hybrid IR engine: TF-IDF VSM + Boolean Retrieval",
    lifespan=lifespan,
)

# Jika FRONTEND_URL belum diatur di Render, izinkan semua ("*")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "*") 

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Response schemas ─────────────────────────────────────────────────────────

class SongResult(BaseModel):
    id: str
    title: str
    artist: str
    snippet: str          # may contain safe HTML <b> tags for highlighting
    score: float


class SearchMeta(BaseModel):
    total_results: int
    compute_time_ms: int
    search_type: str


class SearchResponse(BaseModel):
    status: str
    data: list[SongResult]
    meta: SearchMeta


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/v1/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=200, description="Search keyword"),
    type: Literal["lirik", "judul", "penyanyi"] = Query(
        ..., description="Search filter type"
    ),
):
    """
    Unified search endpoint.
    - type=lirik   → VSM cosine similarity on TF-IDF lyric matrix
    - type=judul   → Boolean substring match on song titles
    - type=penyanyi→ Boolean substring match on artist names
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready.")

    t0 = time.perf_counter()

    if type == "lirik":
        results = engine.search_by_lyric(q)
    elif type == "judul":
        results = engine.search_by_title(q)
    else:
        results = engine.search_by_artist(q)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return SearchResponse(
        status="success",
        data=results,
        meta=SearchMeta(
            total_results=len(results),
            compute_time_ms=elapsed_ms,
            search_type=type,
        ),
    )


@app.get("/api/v1/song/{song_id}")
async def get_song(song_id: str):
    """
    Retrieve full lyrics for a single song by its UUID.
    Called when user clicks a result card to populate the LyricViewer.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready.")

    song = engine.get_song_by_id(song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found.")

    return {"status": "success", "data": song}


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "songs_indexed": len(engine.songs) if engine else 0,
    }