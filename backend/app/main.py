from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .service import get_service


settings = get_settings()
service = get_service()
WINDOWS_LIMIT_MAX = 5000

app = FastAPI(title=settings.project_name, version=settings.version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrieveRequest(BaseModel):
    model: str = Field(default="hopfield")
    window_id: str
    k: int = Field(default=settings.retrieval_top_k, ge=1, le=10)
    beta: float | None = Field(default=None, ge=0.5, le=20.0)
    steps: int | None = Field(default=None, ge=1, le=8)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return service.health()


@app.get("/api/models")
def models() -> dict[str, Any]:
    return service.models()


@app.get("/api/dashboard")
def dashboard(model: str = Query(default="hopfield")) -> dict[str, Any]:
    return service.dashboard(model=model)


@app.get("/api/windows")
def windows(
    patient_id: str | None = Query(default=None),
    label: str | None = Query(default=None),
    meal_segment: str | None = Query(default=None),
    split: str | None = Query(default=None),
    query: str | None = Query(default=None),
    limit: int = Query(default=250, ge=1, le=WINDOWS_LIMIT_MAX),
) -> list[dict[str, Any]]:
    return service.windows(
        patient_id=patient_id,
        label=label,
        meal_segment=meal_segment,
        split=split,
        query=query,
        limit=limit,
    )


@app.get("/api/windows/{window_id}")
def window(window_id: str) -> dict[str, Any]:
    try:
        return service.window(window_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/retrieve")
def retrieve(request: RetrieveRequest) -> dict[str, Any]:
    try:
        return service.retrieve(
            model=request.model,
            window_id=request.window_id,
            k=request.k,
            beta=request.beta,
            steps=request.steps,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/evaluation")
def evaluation(model: str | None = Query(default=None)) -> dict[str, Any]:
    return service.evaluation(selected_model=model)


@app.get("/api/about")
def about() -> dict[str, Any]:
    return service.about()


@app.post("/api/memory/retrieve")
def retrieve_legacy(request: RetrieveRequest) -> dict[str, Any]:
    return retrieve(request)
