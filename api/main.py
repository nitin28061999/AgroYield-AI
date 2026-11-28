import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.inference.predictor import AgroYieldPredictor


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Verify the API key supplied by the client."""

    expected_key = os.getenv("AGROYIELD_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate configuration and load the frozen model once."""

    if not os.getenv("AGROYIELD_API_KEY"):
        raise RuntimeError(
            "AGROYIELD_API_KEY must be configured before startup."
        )

    app.state.predictor = AgroYieldPredictor()

    yield

    app.state.predictor = None


app = FastAPI(
    title="AgroYield AI API",
    description="Crop yield prediction using the frozen CatBoost v1 model.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Public health check for hosting-platform monitoring."""

    predictor = getattr(request.app.state, "predictor", None)

    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict(
    payload: PredictionRequest,
    request: Request,
):
    """Predict crop yield in tonnes per hectare."""

    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not available.",
        )

    raw_prediction = predictor.predict(payload.model_dump())

    warnings = []

    if not 1997 <= payload.start_year <= 2022:
        warnings.append(
            "The requested year is outside the model's evaluated "
            "period (1997-2022). Prediction is not validated "
            "for this year."
        )

    is_valid = predictor.validate_prediction(raw_prediction)

    if not is_valid:
        return PredictionResponse(
            status="invalid_prediction",
            predicted_yield=None,
            start_year=payload.start_year,
            warnings=warnings + [
                "The raw model output is negative or non-finite. "
                "It is not returned as a valid yield estimate."
            ],
            message="The model produced an invalid yield prediction.",
        )

    return PredictionResponse(
        status="success",
        predicted_yield=raw_prediction,
        start_year=payload.start_year,
        warnings=warnings,
        message="Crop yield prediction generated successfully.",
    )
