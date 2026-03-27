"""Tests for intent/scenario generator agent."""

import json
import pytest
from adk_eval_tool.intent_generator.prompts import build_system_instruction
from adk_eval_tool.intent_generator.tools import (
    validate_intent_output,
    format_agent_metadata_summary,
)
from adk_eval_tool.schemas import (
    AgentMetadata,
    ToolMetadata,
    IntentScenarioSet,
    Intent,
    Scenario,
    ScenarioStep,
)


def _sample_metadata() -> AgentMetadata:
    return AgentMetadata(
        name="travel_agent",
        agent_type="LlmAgent",
        description="Helps users book flights and hotels",
        instruction="You are a travel assistant.",
        model="gemini-2.0-flash",
        tools=[
            ToolMetadata(name="search_flights", description="Search for flights", parameters_schema={}),
            ToolMetadata(name="book_flight", description="Book a flight", parameters_schema={}),
            ToolMetadata(name="search_hotels", description="Search for hotels", parameters_schema={}),
        ],
        sub_agents=[
            AgentMetadata(
                name="payment_agent",
                agent_type="LlmAgent",
                description="Handles payments",
                instruction="Process payments.",
                model="gemini-2.0-flash",
                tools=[ToolMetadata(name="process_payment", description="Process payment", parameters_schema={})],
                sub_agents=[],
            )
        ],
    )


def test_build_system_instruction_contains_metadata():
    metadata = _sample_metadata()
    instruction = build_system_instruction(metadata)
    assert "travel_agent" in instruction
    assert "search_flights" in instruction
    assert "payment_agent" in instruction


def test_format_agent_metadata_summary():
    metadata = _sample_metadata()
    summary = format_agent_metadata_summary(metadata)
    assert "travel_agent" in summary
    assert "search_flights" in summary
    assert "payment_agent" in summary


def test_validate_intent_output_valid():
    intent_set = IntentScenarioSet(
        agent_name="travel_agent",
        intents=[
            Intent(
                intent_id="book_flight",
                name="Book Flight",
                description="User wants to book a flight",
                scenarios=[
                    Scenario(
                        scenario_id="happy_path",
                        name="Successful booking",
                        steps=[
                            ScenarioStep(
                                user_message="Book a flight to London",
                                expected_tool_calls=["search_flights"],
                            )
                        ],
                    )
                ],
            )
        ],
    )
    result = validate_intent_output(intent_set.model_dump())
    assert result["valid"] is True


def test_validate_intent_output_invalid():
    result = validate_intent_output({"bad": "data"})
    assert result["valid"] is False
    assert "errors" in result
