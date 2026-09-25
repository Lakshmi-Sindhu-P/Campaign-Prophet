"""Thin read-only API over the generated capacity artifacts.

Run with:  uvicorn app.main:app --reload
Requires:  .venv/bin/python -m pip install -r requirements-serve.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recommend import load_artifacts, summarize  # noqa: E402

app = FastAPI(title="Campaign Prophet", version="1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/recommend")
def recommend(capacity: int = 20) -> dict:
    if capacity <= 0 or capacity > 100:
        raise HTTPException(status_code=400, detail="capacity must be between 1 and 100")
    scored, _ = load_artifacts()
    return summarize(scored, capacity)
