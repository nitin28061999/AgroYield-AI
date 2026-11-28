from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Input schema for crop-yield prediction."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    start_year: int = Field(gt=0)
    area: float = Field(gt=0, allow_inf_nan=False)

    state_code: str = Field(min_length=1)
    district_code: str = Field(min_length=1)
    crop_code: str = Field(min_length=1)

    crop_name: str = Field(min_length=1)
    crop_type: str = Field(min_length=1)
    season: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    """Standard response for crop-yield prediction."""

    status: Literal["success", "invalid_prediction"]

    predicted_yield: float | None
    unit: Literal["tonnes/hectare"] = "tonnes/hectare"

    model_name: str = "catboost_v1"
    start_year: int

    warnings: list[str] = Field(default_factory=list)
    message: str


class HealthResponse(BaseModel):
    """API health response."""

    status: Literal["healthy", "unhealthy"]
    application: str = "AgroYield AI"
    model_loaded: bool
    model_name: str = "catboost_v1"
