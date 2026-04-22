from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .service import get_service


settings = get_settings()
service = get_service()

app = FastAPI(title=settings.project_name, version=settings.version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrieveRequest(BaseModel):
    window_id: str
    k: int = Field(default=settings.retrieval_top_k, ge=1, le=10)
    beta: float = Field(default=settings.hopfield_beta, ge=0.5, le=20.0)
    steps: int = Field(default=settings.recall_steps, ge=1, le=8)


class CustomQueryRequest(BaseModel):
    patient_id: str | None = None
    meal_type: str | None = None
    meal_hour: int | None = Field(default=None, ge=0, le=23)
    carbs: float | None = None
    bolus: float | None = None
    baseline_glucose: float | None = None
    trend_30m: float | None = None
    trend_90m: float | None = None
    active_basal: float | None = None
    hr_mean: float | None = None
    hr_std: float | None = None
    hr_min: float | None = None
    hr_max: float | None = None
    heart_rate_missing: bool | None = None
    premeal_cgm: list[float] | None = None
    premeal_missingness: list[float] | None = None
    k: int = Field(default=settings.retrieval_top_k, ge=1, le=10)
    beta: float = Field(default=settings.hopfield_beta, ge=0.5, le=20.0)
    steps: int = Field(default=settings.recall_steps, ge=1, le=8)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return service.health()


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return service.dashboard()


@app.get("/api/windows")
def windows(
    patient_id: str | None = Query(default=None),
    label: str | None = Query(default=None),
    meal_segment: str | None = Query(default=None),
    split: str | None = Query(default=None),
    limit: int = Query(default=250, ge=1, le=1000),
) -> list[dict[str, Any]]:
    return service.windows(patient_id=patient_id, label=label, meal_segment=meal_segment, split=split, limit=limit)


@app.get("/api/windows/{window_id}")
def window(window_id: str) -> dict[str, Any]:
    try:
        return service.window(window_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/memory/retrieve")
def retrieve(request: RetrieveRequest) -> dict[str, Any]:
    try:
        return service.retrieve(request.window_id, k=request.k, beta=request.beta, steps=request.steps)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/memory/custom-query")
def custom_query(request: CustomQueryRequest) -> dict[str, Any]:
    return service.custom_query(request.model_dump(exclude_none=True), k=request.k, beta=request.beta, steps=request.steps)


@app.get("/api/prototypes")
def prototypes() -> list[dict[str, Any]]:
    return service.prototypes()


@app.get("/api/prototypes/{label}")
def prototype(label: str) -> dict[str, Any]:
    try:
        return service.prototype(label)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/evaluation")
def evaluation() -> dict[str, Any]:
    return service.evaluation()


@app.get("/api/evaluation/noise")
def evaluation_noise() -> list[dict[str, Any]]:
    return service.noise()


@app.get("/api/about")
def about() -> dict[str, Any]:
    return service.about()

