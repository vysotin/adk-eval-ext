"""Stage 2 — Evaluation: score persisted inference results without re-running agent.

Usage:
    bundle = load_inference_bundle("inference_output.json")
    results = await run_evaluation_from_bundle(bundle, metrics)

This stage only needs the inference artifacts and metric configuration.
No agent module is loaded. For non-LLM metrics (tool_trajectory, safety),
no external calls are made at all.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import aclosing as Aclosing
from typing import Any, Optional

from google.adk.evaluation.base_eval_service import (
    EvaluateConfig,
    EvaluateRequest,
    InferenceResult,
)
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_metrics import EvalMetric, EvalStatus
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.local_eval_service import LocalEvalService

from poc_split_eval.schemas import InferenceArtifact, InferenceBundle


# ---------------------------------------------------------------------------
# Metric builders
# ---------------------------------------------------------------------------


def build_eval_metrics(
    metric_configs: list[dict[str, Any]],
) -> list[EvalMetric]:
    """Build ADK EvalMetric objects from simple config dicts.

    Each dict should have:
        - metric_name: str
        - threshold: float (default 0.8)
        - match_type: str (for tool_trajectory, default "IN_ORDER")
        - judge_model: str (for LLM-based metrics)

    Example:
        [{"metric_name": "tool_trajectory_avg_score", "threshold": 0.8}]
    """
    from google.adk.evaluation.eval_config import EvalConfig
    from google.adk.evaluation.eval_config import get_eval_metrics_from_config
    from google.adk.evaluation.eval_metrics import (
        BaseCriterion,
        ToolTrajectoryCriterion,
        JudgeModelOptions,
        LlmAsAJudgeCriterion,
    )

    criteria: dict = {}
    for mc in metric_configs:
        name = mc["metric_name"]
        threshold = mc.get("threshold", 0.8)

        if name == "tool_trajectory_avg_score":
            match_type_val = {"EXACT": 0, "IN_ORDER": 1, "ANY_ORDER": 2}.get(
                mc.get("match_type", "IN_ORDER"), 1
            )
            mt_field = ToolTrajectoryCriterion.model_fields["match_type"]
            mt_enum = mt_field.annotation
            mt = mt_enum(match_type_val)
            criteria[name] = ToolTrajectoryCriterion(
                threshold=threshold, match_type=mt
            )
        elif name in ("final_response_match_v2", "response_evaluation_score"):
            judge = mc.get("judge_model", "gemini-2.0-flash")
            criteria[name] = LlmAsAJudgeCriterion(
                threshold=threshold,
                judge_model_options=JudgeModelOptions(judge_model=judge),
            )
        else:
            criteria[name] = BaseCriterion(threshold=threshold)

    config = EvalConfig(criteria=criteria)
    return get_eval_metrics_from_config(config)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class EvalResult:
    """Friendly wrapper around ADK's EvalCaseResult."""

    def __init__(
        self,
        eval_set_id: str,
        eval_case_id: str,
        status: str,
        overall_scores: dict[str, Optional[float]],
        per_invocation: list[dict[str, Any]],
        session_id: str = "",
    ):
        self.eval_set_id = eval_set_id
        self.eval_case_id = eval_case_id
        self.status = status
        self.overall_scores = overall_scores
        self.per_invocation = per_invocation
        self.session_id = session_id

    def __repr__(self) -> str:
        return (
            f"EvalResult(eval_id={self.eval_case_id!r}, "
            f"status={self.status!r}, scores={self.overall_scores})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_set_id": self.eval_set_id,
            "eval_case_id": self.eval_case_id,
            "status": self.status,
            "overall_scores": self.overall_scores,
            "per_invocation": self.per_invocation,
            "session_id": self.session_id,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _content_to_text(content) -> str:
    if content and content.parts:
        return "\n".join(
            p.text for p in content.parts if hasattr(p, "text") and p.text
        )
    return ""


def _extract_eval_result(
    eval_case_result: EvalCaseResult,
) -> EvalResult:
    """Convert ADK EvalCaseResult to our EvalResult."""
    status_map = {
        EvalStatus.PASSED: "PASSED",
        EvalStatus.FAILED: "FAILED",
        EvalStatus.NOT_EVALUATED: "NOT_EVALUATED",
    }

    overall_scores = {
        mr.metric_name: mr.score
        for mr in eval_case_result.overall_eval_metric_results
    }

    per_inv = []
    for pi in eval_case_result.eval_metric_result_per_invocation:
        actual_tools = []
        if pi.actual_invocation.intermediate_data:
            idata = pi.actual_invocation.intermediate_data
            if hasattr(idata, "tool_uses"):
                actual_tools = [
                    {"name": tc.name, "args": dict(tc.args) if tc.args else {}}
                    for tc in idata.tool_uses
                ]
        expected_tools = []
        if pi.expected_invocation and pi.expected_invocation.intermediate_data:
            edata = pi.expected_invocation.intermediate_data
            if hasattr(edata, "tool_uses"):
                expected_tools = [
                    {"name": tc.name, "args": dict(tc.args) if tc.args else {}}
                    for tc in edata.tool_uses
                ]

        per_inv.append({
            "invocation_id": pi.actual_invocation.invocation_id,
            "user_message": _content_to_text(pi.actual_invocation.user_content),
            "actual_response": _content_to_text(pi.actual_invocation.final_response),
            "expected_response": _content_to_text(
                pi.expected_invocation.final_response
            ) if pi.expected_invocation else "",
            "actual_tool_calls": actual_tools,
            "expected_tool_calls": expected_tools,
            "scores": {
                mr.metric_name: mr.score for mr in pi.eval_metric_results
            },
        })

    return EvalResult(
        eval_set_id=eval_case_result.eval_set_id,
        eval_case_id=eval_case_result.eval_id,
        status=status_map.get(
            eval_case_result.final_eval_status, "NOT_EVALUATED"
        ),
        overall_scores=overall_scores,
        per_invocation=per_inv,
        session_id=eval_case_result.session_id or "",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_evaluation_from_bundle(
    bundle: InferenceBundle,
    metric_configs: list[dict[str, Any]],
) -> list[EvalResult]:
    """Stage 2: evaluate persisted inference results against metrics.

    No agent is loaded. Only the eval metrics and the inference artifacts
    are needed.

    Args:
        bundle: InferenceBundle loaded from disk.
        metric_configs: List of metric config dicts (see build_eval_metrics).

    Returns:
        List of EvalResult with scores for each inference artifact.
    """
    eval_metrics = build_eval_metrics(metric_configs)
    results: list[EvalResult] = []

    # Group artifacts by eval_set_id for efficiency
    sets_seen: dict[str, dict] = {}
    for artifact in bundle.artifacts:
        es_id = artifact.eval_set_json.get("eval_set_id", "")
        if es_id not in sets_seen:
            sets_seen[es_id] = artifact.eval_set_json

    for artifact in bundle.artifacts:
        eval_set_dict = artifact.eval_set_json
        eval_set = EvalSet.model_validate(eval_set_dict)

        # Reconstruct eval sets manager with the original eval cases
        eval_sets_manager = InMemoryEvalSetsManager()
        eval_sets_manager.create_eval_set(
            app_name=bundle.app_name,
            eval_set_id=eval_set.eval_set_id,
        )
        for eval_case in eval_set.eval_cases:
            eval_sets_manager.add_eval_case(
                app_name=bundle.app_name,
                eval_set_id=eval_set.eval_set_id,
                eval_case=eval_case,
            )

        # Reconstruct InferenceResult from JSON
        inference_result = InferenceResult.model_validate(
            artifact.inference_result_json
        )

        # LocalEvalService for evaluation only (root_agent not used for evaluate())
        # We pass a dummy agent — evaluate() doesn't call it.
        eval_service = LocalEvalService(
            root_agent=None,  # type: ignore[arg-type]
            eval_sets_manager=eval_sets_manager,
        )

        evaluate_request = EvaluateRequest(
            inference_results=[inference_result],
            evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
        )

        async with Aclosing(
            eval_service.evaluate(evaluate_request=evaluate_request)
        ) as agen:
            async for eval_case_result in agen:
                results.append(_extract_eval_result(eval_case_result))

    return results


async def run_evaluation_from_file(
    bundle_path: str,
    metric_configs: list[dict[str, Any]],
) -> list[EvalResult]:
    """Convenience: load bundle from file and evaluate.

    Args:
        bundle_path: Path to saved InferenceBundle JSON.
        metric_configs: Metric configuration dicts.

    Returns:
        List of EvalResult.
    """
    from poc_split_eval.inference import load_inference_bundle

    bundle = load_inference_bundle(bundle_path)
    return await run_evaluation_from_bundle(bundle, metric_configs)
