"""Shared test fixtures."""

import pytest
from google.adk.agents import Agent, SequentialAgent


def _dummy_tool(query: str) -> str:
    """Search for information.

    Args:
        query: The search query string.

    Returns:
        Search results as text.
    """
    return f"Results for {query}"


def _another_tool(text: str, max_length: int = 100) -> str:
    """Summarize text to a shorter version.

    Args:
        text: The text to summarize.
        max_length: Maximum length of the summary.

    Returns:
        Summarized text.
    """
    return text[:max_length]


@pytest.fixture
def simple_agent():
    return Agent(
        name="simple_agent",
        model="gemini-2.0-flash",
        description="A simple test agent",
        instruction="You are a helpful assistant. Answer questions clearly.",
        tools=[_dummy_tool],
    )


@pytest.fixture
def agent_with_sub_agents():
    researcher = Agent(
        name="researcher",
        model="gemini-2.0-flash",
        description="Researches topics",
        instruction="You research topics using search.",
        tools=[_dummy_tool],
    )
    writer = Agent(
        name="writer",
        model="gemini-2.0-flash",
        description="Writes summaries",
        instruction="You write clear summaries.",
        tools=[_another_tool],
    )
    coordinator = Agent(
        name="coordinator",
        model="gemini-2.0-flash",
        description="Coordinates research and writing",
        instruction="You coordinate the researcher and writer.",
        sub_agents=[researcher, writer],
    )
    return coordinator


@pytest.fixture
def sequential_agent():
    step1 = Agent(
        name="step1",
        model="gemini-2.0-flash",
        instruction="First step.",
        output_key="step1_output",
    )
    step2 = Agent(
        name="step2",
        model="gemini-2.0-flash",
        instruction="Second step using {step1_output}.",
    )
    return SequentialAgent(
        name="pipeline",
        description="A sequential pipeline",
        sub_agents=[step1, step2],
    )
