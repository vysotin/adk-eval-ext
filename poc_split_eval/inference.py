"""Stage 1 — Inference: run agent and persist results without scoring.

Usage:
    bundle = await run_inference(agent_module, eval_sets, num_runs=2)
    save_inference_bundle(bundle, "inference_output.json")

The output file contains everything needed by Stage 2 (evaluation)
to score the results without re-running the agent.
"""

from __future__ import annotations

import importlib
import json
import time
import uuid
from contextlib import aclosing as Aclosing
from pathlib import Path
from typing import Any, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.evaluation.base_eval_service import (
    InferenceConfig,
    InferenceRequest,
    InferenceResult,
)
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.evaluation.local_eval_service import LocalEvalService

from poc_split_eval.schemas import InferenceArtifact, InferenceBundle


# ---------------------------------------------------------------------------
# Helpers (shared with main app's runner, copied here to keep POC isolated)
# ---------------------------------------------------------------------------

import re as _re


def _camel_to_snake(name: str) -> str:
    return _re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")


def _camel_to_snake_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {_camel_to_snake(k): _camel_to_snake_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_camel_to_snake_dict(item) for item in obj]
    return obj


_FUNCTION_RESPONSE_FIELDS = {"will_continue", "scheduling", "parts", "id", "name", "response"}
_FUNCTION_CALL_FIELDS = {"id", "args", "name", "partial_args", "will_continue"}


def _sanitize_eval_set(data: dict) -> dict:
    for case in data.get("eval_cases", []):
        for inv in case.get("conversation", []):
            idata = inv.get("intermediate_data")
            if not idata:
                continue
            if "tool_responses" in idata:
                sanitized = []
                for tr in idata["tool_responses"]:
                    if isinstance(tr, dict):
                        clean = {k: v for k, v in tr.items() if k in _FUNCTION_RESPONSE_FIELDS}
                        if "name" in clean:
                            clean.setdefault("response", {})
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


def _get_agent(module_name: str, agent_name: Optional[str] = None) -> BaseAgent:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_inference(
    agent_module: str,
    eval_sets: list[dict],
    num_runs: int = 1,
    agent_name: Optional[str] = None,
    app_name: str = "eval_app",
    mock_tools: bool = False,
    tool_response_map: Optional[dict[str, Any]] = None,
    stub_sub_agents: Optional[dict[str, str]] = None,
) -> InferenceBundle:
    """Stage 1: run agent inference and capture results (no scoring).

    Args:
        agent_module: Dotted Python module path containing the agent.
        eval_sets: List of eval set dicts (camelCase ADK format).
        num_runs: Number of inference runs per eval set.
        agent_name: Optional sub-agent name to evaluate.
        app_name: Application name for ADK eval service.
        mock_tools: If True, auto-build a tool response map from eval set
            data and install ``before_tool_callback`` on the agent so that
            tools return canned responses instead of executing.
        tool_response_map: Explicit tool_name → response dict.  Takes
            precedence over ``mock_tools`` auto-extraction.
        stub_sub_agents: Mapping of sub_agent_name → fixed response text.
            Named sub-agents are replaced with lightweight stubs so only
            the root agent makes real LLM calls.

    Returns:
        InferenceBundle with all inference artifacts ready for persistence.
    """
    from poc_split_eval.mocking import (
        build_tool_response_map,
        install_tool_mocks,
        install_sub_agent_stubs,
        MockContext,
    )

    agent = _get_agent(agent_module, agent_name)

    bundle = InferenceBundle(
        app_name=app_name,
        agent_module=agent_module,
        metadata={
            "num_runs": num_runs,
            "timestamp": time.time(),
            "mock_tools": mock_tools,
            "has_tool_response_map": tool_response_map is not None,
            "stub_sub_agents": list(stub_sub_agents.keys()) if stub_sub_agents else [],
        },
    )

    for eval_set_dict in eval_sets:
        snake_dict = _camel_to_snake_dict(eval_set_dict)
        snake_dict = _sanitize_eval_set(snake_dict)
        eval_set = EvalSet.model_validate(snake_dict)

        # Build effective tool response map
        effective_map = tool_response_map
        if effective_map is None and mock_tools:
            effective_map = build_tool_response_map(snake_dict)

        # Install mocking (if requested)
        mock_contexts: list[MockContext] = []
        try:
            if effective_map:
                mock_contexts.append(
                    install_tool_mocks(agent, effective_map, recursive=True)
                )
            if stub_sub_agents:
                mock_contexts.append(
                    install_sub_agent_stubs(agent, stub_sub_agents)
                )

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

            for _ in range(num_runs):
                req = InferenceRequest(
                    app_name=app_name,
                    eval_set_id=eval_set.eval_set_id,
                    inference_config=InferenceConfig(),
                )
                async with Aclosing(
                    eval_service.perform_inference(inference_request=req)
                ) as agen:
                    async for result in agen:
                        artifact = InferenceArtifact(
                            eval_set_json=snake_dict,
                            inference_result_json=json.loads(
                                result.model_dump_json()
                            ),
                            session_id=result.session_id or "",
                        )
                        bundle.artifacts.append(artifact)
        finally:
            for ctx in reversed(mock_contexts):
                ctx.uninstall()

    return bundle


def save_inference_bundle(bundle: InferenceBundle, path: str) -> None:
    """Save an InferenceBundle to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(bundle.model_dump_json(indent=2))


def load_inference_bundle(path: str) -> InferenceBundle:
    """Load an InferenceBundle from a JSON file."""
    return InferenceBundle.model_validate_json(Path(path).read_text())
