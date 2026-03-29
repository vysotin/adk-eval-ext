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
    Task,
    Scenario,
    TestCaseConfig,
)


def _sample_metadata() -> AgentMetadata:
    return AgentMetadata(
        name="travel_agent", agent_type="LlmAgent",
        description="Helps book flights",
        instruction="You are a travel assistant.", model="gemini-2.0-flash",
        tools=[ToolMetadata(name="search_flights", description="Search flights", parameters_schema={})],
    )


def _sample_task() -> Task:
    return Task(
        task_id="book_flight", name="Book Flight",
        description="User books a flight",
        scenarios=[
            Scenario(
                scenario_id="happy_path", name="Successful booking",
                description="User provides valid cities and date, expects a list of available flights",
            ),
        ],
    )


def test_build_eval_case_json_structure():
    scenario = _sample_task().scenarios[0]
    eval_case = build_eval_case_json(scenario=scenario.model_dump(), task_id="book_flight")
    assert eval_case["evalId"] == "book_flight__happy_path"
    assert "conversation" in eval_case
    assert len(eval_case["conversation"]) == 1
    inv = eval_case["conversation"][0]
    assert inv["userContent"]["parts"][0]["text"] == scenario.description


def test_build_eval_case_conversation_scenario():
    scenario = Scenario(
        scenario_id="dynamic_test", name="Dynamic test",
        description="Dynamic booking flow",
        eval_type="conversation_scenario",
        conversation_scenario={
            "starting_prompt": "I need help booking",
            "conversation_plan": "Ask for destination. Confirm booking.",
        },
        session_state={"user_id": "test_123"},
    )
    eval_case = build_eval_case_json(scenario=scenario.model_dump(), task_id="book_flight")
    assert eval_case["evalId"] == "book_flight__dynamic_test"
    assert "conversation_scenario" in eval_case
    assert "conversation" not in eval_case
    assert eval_case["sessionInput"]["state"]["user_id"] == "test_123"


def test_build_eval_set_json():
    task = _sample_task()
    eval_set = build_eval_set_json(task=task.model_dump(), agent_name="travel_agent")
    assert eval_set["evalSetId"] == "travel_agent__book_flight"
    assert len(eval_set["evalCases"]) == 1


def test_validate_eval_set_valid():
    task = _sample_task()
    eval_set = build_eval_set_json(task=task.model_dump(), agent_name="travel_agent")
    result = validate_eval_set(json.dumps(eval_set))
    assert result["valid"] is True


def test_build_testcase_system_instruction():
    instruction = build_testcase_system_instruction(_sample_metadata(), TestCaseConfig())
    assert "travel_agent" in instruction
