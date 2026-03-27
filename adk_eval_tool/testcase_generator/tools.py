"""Function tools for the test case generator agent."""

from __future__ import annotations

import json
from typing import Any


def build_eval_case_json(
    scenario: dict[str, Any],
    intent_id: str,
) -> dict[str, Any]:
    """Build an ADK EvalCase JSON object from a scenario.

    Handles both trajectory-based (static) and conversation_scenario (dynamic)
    eval types. For trajectory-based scenarios, maps ScenarioStep fields to ADK
    Invocation structure including toolUses, toolResponses, intermediateResponses,
    finalResponse, and per-invocation rubrics.

    Args:
        scenario: A scenario dict with scenario_id, steps, eval_type, etc.
        intent_id: The parent intent ID.

    Returns:
        An EvalCase dict in ADK camelCase format.
    """
    scenario_id = scenario.get("scenario_id", "unknown")
    eval_id = f"{intent_id}__{scenario_id}"
    eval_type = scenario.get("eval_type", "trajectory")

    # Dynamic conversation_scenario mode
    if eval_type == "conversation_scenario" and scenario.get("conversation_scenario"):
        eval_case: dict[str, Any] = {
            "evalId": eval_id,
            "conversation_scenario": scenario["conversation_scenario"],
        }
        if scenario.get("session_state"):
            eval_case["sessionInput"] = {
                "appName": "eval_app",
                "userId": "test_user",
                "state": scenario["session_state"],
            }
        return eval_case

    # Static trajectory mode
    conversation = []
    for i, step in enumerate(scenario.get("steps", [])):
        # Build toolUses with args
        tool_uses = []
        for tool_name in step.get("expected_tool_calls", []):
            tool_use = {"name": tool_name}
            tool_args = step.get("expected_tool_args", {})
            if tool_args and tool_name in tool_args:
                tool_use["args"] = tool_args[tool_name]
            else:
                tool_use["args"] = {}
            tool_uses.append(tool_use)

        # Build toolResponses (reference data for hallucination eval)
        tool_responses = []
        if step.get("tool_responses"):
            for tool_name, response_data in step["tool_responses"].items():
                tool_responses.append({
                    "name": tool_name,
                    "id": f"call_{i}_{tool_name}",
                    "response": response_data,
                })

        # Build intermediateResponses
        intermediate_responses = step.get("intermediate_responses", [])

        invocation: dict[str, Any] = {
            "invocationId": f"inv-{i + 1}",
            "userContent": {
                "role": "user",
                "parts": [{"text": step.get("user_message", "")}],
            },
            "intermediateData": {
                "toolUses": tool_uses,
                "toolResponses": tool_responses,
                "intermediateResponses": intermediate_responses,
            },
        }

        # Add finalResponse: prefer explicit expected_response, fall back to keywords
        expected_response = step.get("expected_response", "")
        keywords = step.get("expected_response_keywords", [])
        if expected_response:
            invocation["finalResponse"] = {
                "role": "model",
                "parts": [{"text": expected_response}],
            }
        elif keywords:
            invocation["finalResponse"] = {
                "role": "model",
                "parts": [{"text": f"Response should include: {', '.join(keywords)}"}],
            }

        # Add per-invocation rubric if present
        if step.get("rubric"):
            invocation["rubrics"] = [{
                "rubricId": f"rubric_inv_{i + 1}",
                "rubricContent": {"textProperty": step["rubric"]},
            }]

        conversation.append(invocation)

    eval_case = {
        "evalId": eval_id,
        "conversation": conversation,
    }

    # Add session state if present
    if scenario.get("session_state"):
        eval_case["sessionInput"] = {
            "appName": "eval_app",
            "userId": "test_user",
            "state": scenario["session_state"],
        }

    return eval_case


def build_eval_set_json(
    intent: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    """Build a complete ADK EvalSet JSON from an intent.

    Args:
        intent: An intent dict with intent_id, scenarios, etc.
        agent_name: The name of the agent being tested.

    Returns:
        An EvalSet dict in ADK camelCase format.
    """
    intent_id = intent.get("intent_id", "unknown")
    eval_cases = []
    for scenario in intent.get("scenarios", []):
        eval_cases.append(build_eval_case_json(scenario, intent_id))

    return {
        "evalSetId": f"{agent_name}__{intent_id}",
        "name": intent.get("name", intent_id),
        "description": intent.get("description", ""),
        "evalCases": eval_cases,
    }


def validate_eval_set(eval_set_json: str) -> dict[str, Any]:
    """Validate that a JSON string is a valid ADK EvalSet structure.

    Args:
        eval_set_json: JSON string of an EvalSet.

    Returns:
        Dict with 'valid' bool and optional 'errors' list.
    """
    errors = []
    try:
        data = json.loads(eval_set_json)
    except json.JSONDecodeError as e:
        return {"valid": False, "errors": [f"Invalid JSON: {e}"]}

    if "evalSetId" not in data:
        errors.append("Missing 'evalSetId'")
    if "evalCases" not in data:
        errors.append("Missing 'evalCases'")
    elif not isinstance(data["evalCases"], list):
        errors.append("'evalCases' must be a list")
    else:
        for i, case in enumerate(data["evalCases"]):
            if "evalId" not in case:
                errors.append(f"evalCases[{i}]: missing 'evalId'")
            if "conversation" not in case and "conversation_scenario" not in case:
                errors.append(f"evalCases[{i}]: must have 'conversation' or 'conversation_scenario'")
            if "conversation" in case:
                for j, inv in enumerate(case["conversation"]):
                    if "userContent" not in inv:
                        errors.append(f"evalCases[{i}].conversation[{j}]: missing 'userContent'")

    return {"valid": len(errors) == 0, "errors": errors}


def save_eval_set(eval_set_json: str) -> str:
    """Save and validate a generated EvalSet JSON. Called by the agent.

    Args:
        eval_set_json: The complete EvalSet JSON string.

    Returns:
        Validation result message.
    """
    result = validate_eval_set(eval_set_json)
    if result["valid"]:
        return "EvalSet validated successfully and ready to save."
    else:
        return f"Validation errors: {result['errors']}"
