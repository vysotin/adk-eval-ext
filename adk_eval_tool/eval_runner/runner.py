"""Run ADK evaluations with trace collection and structured result capture.

Provides a clear two-stage pipeline:
  1. ``run_inference_only`` — run agent against eval cases, capture traces
  2. ``run_eval_scoring``  — score persisted inference results against metrics

The legacy ``run_evaluation`` convenience function chains both stages.
"""

from __future__ import annotations

import importlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from google.adk.agents.base_agent import BaseAgent

from adk_eval_tool.schemas import (
    EvalRunConfig,
    EvalRunResult,
    InferenceRunResult,
    MetricConfig,
    UserSimulatorConfig,
    CustomMetricDef,
)
from adk_eval_tool.eval_runner.trace_collector import (
    setup_trace_collection,
    get_trace_tree_for_session,
    compute_basic_metrics,
)
from adk_eval_tool.eval_runner.result_store import ResultStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_eval_config_from_metrics(
    metrics: list[MetricConfig],
    judge_model: str,
    user_sim: Optional[UserSimulatorConfig] = None,
    custom_metrics: Optional[list[CustomMetricDef]] = None,
):
    """Convert our config to ADK EvalConfig."""
    from google.adk.evaluation.eval_config import EvalConfig, CustomMetricConfig
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

    adk_user_sim = None
    if user_sim:
        try:
            from google.adk.evaluation.simulation.llm_backed_user_simulator import (
                LlmBackedUserSimulatorConfig,
            )
            adk_user_sim = LlmBackedUserSimulatorConfig(
                model=user_sim.model,
                max_allowed_invocations=user_sim.max_allowed_invocations,
                custom_instructions=user_sim.custom_instructions,
            )
        except ImportError:
            pass

    adk_custom_metrics = None
    if custom_metrics:
        from google.adk.agents.common_configs import CodeConfig
        adk_custom_metrics = {}
        for cm in custom_metrics:
            adk_custom_metrics[cm.name] = CustomMetricConfig(
                code_config=CodeConfig(name=cm.code_path),
                description=cm.description,
            )

    return EvalConfig(
        criteria=criteria,
        user_simulator_config=adk_user_sim,
        custom_metrics=adk_custom_metrics,
    )


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
    return _re.sub(r'([A-Z])', r'_\1', name).lower().lstrip('_')


def _camel_to_snake_dict(obj):
    if isinstance(obj, dict):
        return {_camel_to_snake(k): _camel_to_snake_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_camel_to_snake_dict(item) for item in obj]
    return obj


_FUNCTION_RESPONSE_FIELDS = {"will_continue", "scheduling", "parts", "id", "name", "response"}
_FUNCTION_CALL_FIELDS = {"id", "args", "name", "partial_args", "will_continue"}


def _coerce_to_dict(value) -> dict:
    """Ensure *value* is a dict.  JSON-encoded strings are decoded first."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            import json as _json
            parsed = _json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
        return {"result": value}
    return {"result": str(value)}


def _sanitize_eval_set(data: dict) -> dict:
    """Remove / fix fields that ADK's strict Pydantic models reject.

    Handles common issues produced by LLM-generated eval sets:
    1. ``tool_responses[].response`` is a JSON *string* instead of a dict.
    2. Extra fields in tool_uses / tool_responses that FunctionCall /
       FunctionResponse don't accept.
    3. ``tool_responses``, ``tool_uses``, or ``intermediate_responses``
       placed at the invocation level instead of inside ``intermediate_data``.
    """
    for case in data.get("eval_cases", []):
        for inv in case.get("conversation", []):
            # Fix misplaced fields: move them into intermediate_data
            _INTERMEDIATE_KEYS = ("tool_uses", "tool_responses", "intermediate_responses")
            for key in _INTERMEDIATE_KEYS:
                if key in inv and key != "intermediate_data":
                    idata = inv.setdefault("intermediate_data", {})
                    if key not in idata:
                        idata[key] = inv.pop(key)
                    else:
                        inv.pop(key)

            idata = inv.get("intermediate_data")
            if not idata:
                continue

            if "tool_responses" in idata:
                sanitized = []
                for tr in idata["tool_responses"]:
                    if isinstance(tr, dict):
                        clean = {k: v for k, v in tr.items() if k in _FUNCTION_RESPONSE_FIELDS}
                        if "name" in clean:
                            clean["response"] = _coerce_to_dict(clean.get("response", {}))
                            sanitized.append(clean)
                    else:
                        sanitized.append(tr)
                idata["tool_responses"] = sanitized

            if "tool_uses" in idata:
                sanitized = []
                for tu in idata["tool_uses"]:
                    if isinstance(tu, dict):
                        clean = {k: v for k, v in tu.items() if k in _FUNCTION_CALL_FIELDS}
                        if "name" in clean:
                            clean.setdefault("args", {})
                            sanitized.append(clean)
                    else:
                        sanitized.append(tu)
                idata["tool_uses"] = sanitized

    return data


def _prepare_eval_set(eval_set_dict: dict):
    """Convert camelCase dict → sanitised snake_case dict → ADK EvalSet."""
    from google.adk.evaluation.eval_set import EvalSet

    snake_dict = _camel_to_snake_dict(eval_set_dict)
    snake_dict = _sanitize_eval_set(snake_dict)
    return snake_dict, EvalSet.model_validate(snake_dict)


# ---------------------------------------------------------------------------
# Stage 1: Inference only
# ---------------------------------------------------------------------------


async def run_inference_only(
    config: EvalRunConfig,
    eval_sets: list[dict],
    save_dir: Optional[str] = None,
) -> list[InferenceRunResult]:
    """Run agent inference without scoring.

    Args:
        config: Evaluation run configuration (agent module, num_runs …).
        eval_sets: List of EvalSet dicts (camelCase, ADK format).
        save_dir: Optional directory to persist inference artefacts.

    Returns:
        List of InferenceRunResult with raw invocations and trace trees.
    """
    from google.adk.evaluation.base_eval_service import (
        InferenceRequest,
        InferenceConfig,
    )
    from google.adk.evaluation.local_eval_service import LocalEvalService
    from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager

    exporter = setup_trace_collection(config.trace_db_path)
    agent = _get_agent(config.agent_module, config.agent_name)

    all_results: list[InferenceRunResult] = []
    app_name = "eval_app"

    for eval_set_dict in eval_sets:
        snake_dict, eval_set = _prepare_eval_set(eval_set_dict)

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

        from adk_eval_tool.llm_runner import run_inference_with_timeout

        for _ in range(config.num_runs):
            req = InferenceRequest(
                app_name=app_name,
                eval_set_id=eval_set.eval_set_id,
                inference_config=InferenceConfig(),
            )
            inf_results_batch = await run_inference_with_timeout(
                eval_service, req, timeout=300, max_retries=2,
            )
            for result in inf_results_batch:
                    run_id = f"inf-{uuid.uuid4().hex[:8]}"

                    # Extract actual invocations summary
                    actual_invs = []
                    for inv in (result.inferences or []):
                        inv_data: dict[str, Any] = {
                            "invocation_id": inv.invocation_id,
                            "user_message": _content_to_text(inv.user_content),
                            "actual_response": _content_to_text(inv.final_response),
                        }
                        if inv.intermediate_data and hasattr(inv.intermediate_data, "tool_uses"):
                            inv_data["actual_tool_calls"] = [
                                {"name": tc.name, "args": dict(tc.args) if tc.args else {}}
                                for tc in inv.intermediate_data.tool_uses
                            ]
                        actual_invs.append(inv_data)

                    # Trace tree
                    trace_tree = None
                    basic_metrics = None
                    if result.session_id:
                        try:
                            exporter.force_flush()
                            trees = get_trace_tree_for_session(exporter, result.session_id)
                            trace_tree = trees[0] if trees else None
                        except Exception:
                            pass
                    if trace_tree:
                        basic_metrics = compute_basic_metrics(trace_tree)

                    inf_result = InferenceRunResult(
                        run_id=run_id,
                        eval_set_id=eval_set.eval_set_id,
                        eval_id=result.eval_case_id,
                        session_id=result.session_id or "",
                        inference_result_json=json.loads(result.model_dump_json()),
                        eval_set_json=snake_dict,
                        actual_invocations=actual_invs,
                        basic_metrics=basic_metrics,
                        trace_tree=trace_tree,
                        timestamp=time.time(),
                    )
                    all_results.append(inf_result)

    # Persist to disk
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in all_results:
            (out / f"{r.run_id}.json").write_text(r.model_dump_json(indent=2))

    return all_results


# ---------------------------------------------------------------------------
# Stage 2: Offline evaluation (scoring)
# ---------------------------------------------------------------------------


async def run_eval_scoring(
    config: EvalRunConfig,
    inference_results: list[InferenceRunResult],
    save_dir: Optional[str] = None,
    result_store: Optional[ResultStore] = None,
) -> list[EvalRunResult]:
    """Score persisted inference results against metrics (no agent needed).

    Args:
        config: Evaluation run configuration (metrics, judge_model …).
        inference_results: Output of ``run_inference_only()``.
        save_dir: Optional directory to persist eval results.
        result_store: Optional ResultStore to persist results.

    Returns:
        List of EvalRunResult with scores and trace trees.
    """
    from google.adk.evaluation.base_eval_service import (
        EvaluateRequest,
        EvaluateConfig,
        InferenceResult,
    )
    from google.adk.evaluation.eval_set import EvalSet
    from google.adk.evaluation.local_eval_service import LocalEvalService
    from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
    from google.adk.evaluation.eval_metrics import EvalStatus
    from google.adk.evaluation.eval_config import get_eval_metrics_from_config
    from contextlib import aclosing as Aclosing

    eval_config = _build_eval_config_from_metrics(
        config.metrics, config.judge_model, config.user_simulator, config.custom_metrics
    )
    eval_metrics = get_eval_metrics_from_config(eval_config)

    all_results: list[EvalRunResult] = []
    app_name = "eval_app"

    for inf_result in inference_results:
        eval_set = EvalSet.model_validate(inf_result.eval_set_json)

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

        adk_inference_result = InferenceResult.model_validate(
            inf_result.inference_result_json
        )

        eval_service = LocalEvalService(
            root_agent=None,  # type: ignore[arg-type]
            eval_sets_manager=eval_sets_manager,
        )

        evaluate_request = EvaluateRequest(
            inference_results=[adk_inference_result],
            evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
        )

        async with Aclosing(
            eval_service.evaluate(evaluate_request=evaluate_request)
        ) as agen:
            async for eval_case_result in agen:
                run_id = f"run-{uuid.uuid4().hex[:8]}"

                overall_scores = {
                    mr.metric_name: mr.score
                    for mr in eval_case_result.overall_eval_metric_results
                }

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
                    basic_metrics=inf_result.basic_metrics,
                    session_id=inf_result.session_id,
                    trace_tree=inf_result.trace_tree,
                    timestamp=time.time(),
                )
                all_results.append(run_result)
                if result_store:
                    result_store.save_result(run_result)

    # Persist to disk
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in all_results:
            (out / f"{r.run_id}.json").write_text(r.model_dump_json(indent=2))

    return all_results


# ---------------------------------------------------------------------------
# Legacy convenience function
# ---------------------------------------------------------------------------


async def run_evaluation(
    config: EvalRunConfig,
    eval_sets: list[dict],
    result_store: Optional[ResultStore] = None,
) -> list[EvalRunResult]:
    """Run inference + scoring in one call (legacy API).

    Args:
        config: Evaluation run configuration.
        eval_sets: List of EvalSet dicts (camelCase, ADK format).
        result_store: Optional ResultStore to persist results.

    Returns:
        List of EvalRunResult with scores and trace trees.
    """
    inference_results = await run_inference_only(config, eval_sets)
    return await run_eval_scoring(
        config, inference_results, result_store=result_store
    )
