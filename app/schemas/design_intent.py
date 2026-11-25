"""Pydantic schemas: the boundary between the LLM and the deterministic pipeline.

Anything the model returns must validate against DesignIntent before the
deterministic steps (image handling, vectorization) are allowed to see it.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Dimensions(BaseModel):
    """Physical size of the final piece, in millimeters."""

    width_mm: float = Field(ge=1, le=2000)
    height_mm: float = Field(ge=1, le=2000)


class DesignIntent(BaseModel):
    """What the user wants made, as structured data the pipeline can trust."""

    subject: str = Field(min_length=1, description="What to draw, e.g. 'a cedar tree'")
    style: Literal["line_art", "silhouette", "geometric"] = "line_art"
    dimensions: Dimensions
    notes: Optional[str] = Field(
        default=None, description="Anything else the user asked for that affects the design"
    )


class Candidate(BaseModel):
    """One design option shown to the user."""

    id: str
    image_url: str
    svg: Optional[str] = None
    local_path: Optional[str] = Field(
        default=None, description="Server-side raster file this candidate was traced from"
    )


class UserRequest(BaseModel):
    message: str = Field(min_length=1)


class DesignResponse(BaseModel):
    intent: DesignIntent
    candidates: list[Candidate]
    explanation: str
