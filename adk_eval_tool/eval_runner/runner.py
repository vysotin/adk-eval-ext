"""Run ADK evaluations with trace collection and structured result capture.

Uses ADK's LocalEvalService internally (same as AgentEvaluator) but
captures EvalCaseResult objects instead of asserting, and sets up
OpenTelemetry tracing to record execution spans.
"""

from __future__ import annotations

import importlib
import time
import uuid
from typing import Optional

from google.adk.agents.base_agent import BaseAgent

from adk_eval_tool.schemas import (
    EvalRunConfig,
    EvalRunResult,
    MetricConfig,
)
from adk_eval_tool.eval_runner.trace_collector import (
    setup_trace_collection,
    get_trace_tree_for_session,
    compute_basic_metrics,
)
from adk_eval_tool.eval_runner.result_store import ResultStore


def _build_eval_config_from_metrics(metrics: list[MetricConfig], judge_model: str):
    """Convert our MetricConfig list to ADK EvalConfig."""
    from google.adk.evaluation.eval_config import EvalConfig
    from google.adk.evaluation.eval_metrics import (
        BaseCriterion,
        ToolTrajectoryCriterion,
        HallucinationsCriterion,
        LlmAsAJudgeCriterion,
        JudgeModelOptions,
    )

    criteria: dict = {}
    for mc in metrics:
        jm = mc.judge_model or judge_model
        judge_opts = JudgeModelOptions(judge_model=jm, num_samples=mc.judge_num_samples)

        if mc.metric_name == "tool_trajectory_avg_score":
            match_type_val = {"EXACT": 0, "IN_ORDER": 1, "ANY_ORDER": 2}.get(
                mc.match_type or "IN_ORDER", 1
            )
            mt_field = ToolTrajectoryCriterion.model_fields["match_type"]
            mt_enum = mt_field.annotation
            mt = mt_enum(match_type_val)
            criteria[mc.metric_name] = ToolTrajectoryCriterion(
                threshold=mc.threshold, match_type=mt
            )
        elif mc.metric_name == "hallucinations_v1":
            criteria[mc.metric_name] = HallucinationsCriterion(
                threshold=mc.threshold,
                judge_model_options=judge_opts,
                evaluate_intermediate_nl_responses=mc.evaluate_intermediate,
            )
        elif mc.metric_name in (
            "final_response_match_v2",
            "response_evaluation_score",
        ):
            criteria[mc.metric_name] = LlmAsAJudgeCriterion(
                threshold=mc.threshold,
                judge_model_options=judge_opts,
            )
        else:
            criteria[mc.metric_name] = BaseCriterion(threshold=mc.threshold)

    return EvalConfig(criteria=criteria)


def _get_agent(module_name: str, agent_name: Optional[str] = None) -> BaseAgent:
    """Load agent from module."""
    module = importlib.import_module(module_name)
    agent_module = module.agent if hasattr(module, "agent") else module
    if hasattr(agent_module, "root_agent"):
        root = agent_module.root_agent
    else:
        raise ValueError(f"Module {module_name} has no root_agent")

    if agent_name:
        found = root.find_agent(agent_name)
        if not found:
            raise ValueError(f"Agent '{agent_name}' not found")
        return found
    return root


def _content_to_text(content) -> str:
    """Extract text from genai Content object."""
    if content and content.parts:
        return "\n".join(p.text for p in content.parts if hasattr(p, "text") and p.text)
    return ""


import re as _re


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s = _re.sub(r'([A-Z])', r'_\1', name).lower().lstrip('_')
    return s


def _camel_to_snake_dict(obj):
    """Recursively convert dict keys from camelCase to snake_case."""
    if isinstance(obj, dict):
        return {_camel_to_snake(k): _camel_to_snake_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_camel_to_snake_dict(item) for item in obj]
    return obj


async def run_evaluation(
    config: EvalRunConfig,
    eval_sets: list[dict],
    result_store: Optional[ResultStore] = None,
) -> list[EvalRunResult]:
    """Run ADK evaluation with trace collection and result capture.

    Args:
        config: Evaluation run configuration.
        eval_sets: List of EvalSet dicts (camelCase, ADK format).
        result_store: Optional ResultStore to persist results.

    Returns:
        List of EvalRunResult with scores and trace trees.
    """
    from google.adk.evaluation.eval_set import EvalSet
    from google.adk.evaluation.base_eval_service import (
        InferenceRequest,
        InferenceConfig,
        EvaluateRequest,
        EvaluateConfig,
    )
    from google.adk.evaluation.local_eval_service import LocalEvalService
    from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
    from google.adk.evaluation.eval_metrics import EvalStatus
    from google.adk.evaluation.eval_config import get_eval_metrics_from_config
    from contextlib import aclosing as Aclosing
    import json

    # Set up trace collection
    exporter = setup_trace_collection(config.trace_db_path)

    # Load agent
    agent = _get_agent(config.agent_module, config.agent_name)

    # Build ADK eval config from our metric configs
    eval_config = _build_eval_config_from_metrics(config.metrics, config.judge_model)
    eval_metrics = get_eval_metrics_from_config(eval_config)

    all_results: list[EvalRunResult] = []
    app_name = "eval_app"

    for eval_set_dict in eval_sets:
        # Our eval set dicts use camelCase (evalSetId, evalCases) but ADK's
        # EvalSet Pydantic model uses snake_case (eval_set_id, eval_cases).
        # Convert before validation.
        snake_dict = _camel_to_snake_dict(eval_set_dict)
        eval_set = EvalSet.model_validate(snake_dict)

        eval_sets_manager = InMemoryEvalSetsManager()
        eval_sets_manager.create_eval_set(
            app_name=app_name, eval_set_id=eval_set.eval_set_id
        )
        for eval_case in eval_set.eval_cases:
            eval_sets_manager.add_eval_case(
                app_name=app_name,
                eval_set_id=eval_set.eval_set_id,
                eval_case=eval_case,
            )

        eval_service = LocalEvalService(
            root_agent=agent,
            eval_sets_manager=eval_sets_manager,
        )

        # Run inference
        inference_requests = [
            InferenceRequest(
                app_name=app_name,
                eval_set_id=eval_set.eval_set_id,
                inference_config=InferenceConfig(),
            )
        ] * config.num_runs

        inference_results = []
        for req in inference_requests:
            async with Aclosing(
                eval_service.perform_inference(inference_request=req)
            ) as agen:
                async for result in agen:
                    inference_results.append(result)

        # Run evaluation
        evaluate_request = EvaluateRequest(
            inference_results=inference_results,
            evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
        )
        async with Aclosing(
            eval_service.evaluate(evaluate_request=evaluate_request)
        ) as agen:
            async for eval_case_result in agen:
                run_id = f"run-{uuid.uuid4().hex[:8]}"

                # Extract overall scores
                overall_scores = {}
                for metric_result in eval_case_result.overall_eval_metric_results:
                    overall_scores[metric_result.metric_name] = metric_result.score

                # Extract per-invocation scores with tool call details
                per_inv_scores = []
                for per_inv in eval_case_result.eval_metric_result_per_invocation:
                    actual_tools = []
                    expected_tools = []
                    if per_inv.actual_invocation.intermediate_data:
                        idata = per_inv.actual_invocation.intermediate_data
                        if hasattr(idata, "tool_uses"):
                            actual_tools = [
                                {"name": tc.name, "args": dict(tc.args) if tc.args else {}}
                                for tc in idata.tool_uses
                            ]
                    if per_inv.expected_invocation and per_inv.expected_invocation.intermediate_data:
                        edata = per_inv.expected_invocation.intermediate_data
                        if hasattr(edata, "tool_uses"):
                            expected_tools = [
                                {"name": tc.name, "args": dict(tc.args) if tc.args else {}}
                                for tc in edata.tool_uses
                            ]

                    inv_data = {
                        "invocation_id": per_inv.actual_invocation.invocation_id,
                        "user_message": _content_to_text(per_inv.actual_invocation.user_content),
                        "actual_response": _content_to_text(per_inv.actual_invocation.final_response),
                        "expected_response": _content_to_text(
                            per_inv.expected_invocation.final_response
                        ) if per_inv.expected_invocation else "",
                        "actual_tool_calls": actual_tools,
                        "expected_tool_calls": expected_tools,
                        "scores": {
                            mr.metric_name: mr.score
                            for mr in per_inv.eval_metric_results
                        },
                    }
                    per_inv_scores.append(inv_data)

                # Get trace tree for this session
                trace_tree = None
                if eval_case_result.session_id:
                    try:
                        exporter.force_flush()
                        trees = get_trace_tree_for_session(
                            exporter, eval_case_result.session_id
                        )
                        trace_tree = trees[0] if trees else None
                    except Exception:
                        pass

                # Compute basic metrics from trace tree
                basic_metrics = None
                if trace_tree:
                    basic_metrics = compute_basic_metrics(trace_tree)

                status_map = {
                    EvalStatus.PASSED: "PASSED",
                    EvalStatus.FAILED: "FAILED",
                    EvalStatus.NOT_EVALUATED: "NOT_EVALUATED",
                }

                run_result = EvalRunResult(
                    run_id=run_id,
                    eval_set_id=eval_set.eval_set_id,
                    eval_id=eval_case_result.eval_id,
                    status=status_map.get(
                        eval_case_result.final_eval_status, "NOT_EVALUATED"
                    ),
                    overall_scores=overall_scores,
                    per_invocation_scores=per_inv_scores,
                    basic_metrics=basic_metrics,
                    session_id=eval_case_result.session_id,
                    trace_tree=trace_tree,
                    timestamp=time.time(),
                )

                all_results.append(run_result)
                if result_store:
                    result_store.save_result(run_result)

    return all_results
