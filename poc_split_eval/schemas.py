"""Lightweight schemas for the split-eval POC.

Wraps ADK types with JSON-serialisable containers so that inference
artefacts can be persisted to disk and reloaded for evaluation.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class InferenceArtifact(BaseModel):
    """One inference run's output, ready for serialisation.

    Stores the raw ADK InferenceResult JSON alongside the eval set
    JSON that produced it, so evaluation can be performed later
    without the original agent.
    """

    eval_set_json: dict[str, Any]
    inference_result_json: dict[str, Any]
    session_id: str = ""
    trace_spans_json: list[dict[str, Any]] = Field(default_factory=list)


class InferenceBundle(BaseModel):
    """Collection of inference artifacts from a full run."""

    app_name: str = "eval_app"
    agent_module: str = ""
    artifacts: list[InferenceArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
