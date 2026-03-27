"""Tests for test case generator."""

import json
import pytest
from adk_eval_tool.testcase_generator.tools import (
    build_eval_case_json,
    build_eval_set_json,
    validate_eval_set,
)
from adk_eval_tool.testcase_generator.prompts import build_testcase_system_instruction
from adk_eval_tool.schemas import (
    AgentMetadata,
    ToolMetadata,
    Intent,
    Scenario,
    ScenarioStep,
    TestCaseConfig,
)


def _sample_metadata() -> AgentMetadata:
    return AgentMetadata(
        name="travel_agent",
        agent_type="LlmAgent",
        description="Helps book flights",
        instruction="You are a travel assistant.",
        model="gemini-2.0-flash",
        tools=[
            ToolMetadata(name="search_flights", description="Search flights", parameters_schema={}),
        ],
        sub_agents=[],
    )


def _sample_intent() -> Intent:
    return Intent(
        intent_id="book_flight",
        name="Book Flight",
        description="User books a flight",
        category="booking",
        scenarios=[
            Scenario(
                scenario_id="happy_path",
                name="Successful booking",
                steps=[
                    ScenarioStep(
                        user_message="Find flights to London tomorrow",
                        expected_tool_calls=["search_flights"],
                        expected_tool_args={"search_flights": {"destination": "London"}},
                        tool_responses={"search_flights": {"flights": [{"id": "FL-1"}]}},
                        expected_response="I found flights to London.",
                        expected_response_keywords=["London", "flight"],
                        rubric="Agent should present flight options clearly.",
                    ),
                    ScenarioStep(
                        user_message="Book the cheapest one",
                        expected_tool_calls=["search_flights"],
                        expected_response_keywords=["booked"],
                    ),
                ],
            )
        ],
    )


def test_build_eval_case_json_structure():
    scenario = _sample_intent().scenarios[0]
    eval_case = build_eval_case_json(
        scenario=scenario.model_dump(),
        intent_id="book_flight",
    )
    assert eval_case["evalId"] == "book_flight__happy_path"
    assert "conversation" in eval_case
    assert len(eval_case["conversation"]) == 2
    inv = eval_case["conversation"][0]
    assert inv["userContent"]["parts"][0]["text"] == "Find flights to London tomorrow"
    assert inv["intermediateData"]["toolUses"][0]["name"] == "search_flights"
    assert inv["intermediateData"]["toolUses"][0]["args"]["destination"] == "London"
    assert len(inv["intermediateData"]["toolResponses"]) == 1
    assert inv["finalResponse"]["parts"][0]["text"] == "I found flights to London."
    assert "rubrics" in inv
    assert inv["rubrics"][0]["rubricId"] == "rubric_inv_1"


def test_build_eval_case_conversation_scenario():
    scenario = Scenario(
        scenario_id="dynamic_test",
        name="Dynamic test",
        eval_type="conversation_scenario",
        conversation_scenario={
            "starting_prompt": "I need help booking",
            "conversation_plan": "Ask for destination. Confirm booking.",
        },
        session_state={"user_id": "test_123"},
    )
    eval_case = build_eval_case_json(
        scenario=scenario.model_dump(),
        intent_id="book_flight",
    )
    assert eval_case["evalId"] == "book_flight__dynamic_test"
    assert "conversation_scenario" in eval_case
    assert "conversation" not in eval_case
    assert eval_case["sessionInput"]["state"]["user_id"] == "test_123"


def test_build_eval_set_json():
    intent = _sample_intent()
    eval_set = build_eval_set_json(
        intent=intent.model_dump(),
        agent_name="travel_agent",
    )
    assert eval_set["evalSetId"] == "travel_agent__book_flight"
    assert len(eval_set["evalCases"]) == 1


def test_validate_eval_set_valid():
    intent = _sample_intent()
    eval_set = build_eval_set_json(intent=intent.model_dump(), agent_name="travel_agent")
    result = validate_eval_set(json.dumps(eval_set))
    assert result["valid"] is True


def test_build_testcase_system_instruction():
    metadata = _sample_metadata()
    config = TestCaseConfig()
    instruction = build_testcase_system_instruction(metadata, config)
    assert "travel_agent" in instruction
    assert "evalset.json" in instruction.lower() or "EvalSet" in instruction
