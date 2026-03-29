"""Tests for the source-code-based agent parser."""

import json
import textwrap
from pathlib import Path

import pytest

from adk_eval_tool.agent_parser.parser import parse_agent_from_source
from adk_eval_tool.schemas import AgentMetadata


@pytest.fixture
def simple_agent_source(tmp_path) -> Path:
    src = tmp_path / "simple_agent.py"
    src.write_text(textwrap.dedent('''\
        from google.adk.agents import Agent


        def search(query: str) -> str:
            """Search for information.

            Args:
                query: The search query string.

            Returns:
                Search results as text.
            """
            return f"Results for {query}"


        root_agent = Agent(
            name="simple_agent",
            model="gemini-2.0-flash",
            description="A simple test agent",
            instruction="You are a helpful assistant. Answer questions clearly.",
            tools=[search],
        )
    '''))
    return src


@pytest.fixture
def multi_agent_source(tmp_path) -> Path:
    src = tmp_path / "multi_agent.py"
    src.write_text(textwrap.dedent('''\
        from google.adk.agents import Agent


        def search_flights(origin: str, destination: str, date: str) -> dict:
            """Search for available flights between two cities.

            Args:
                origin: Departure city.
                destination: Arrival city.
                date: Travel date in YYYY-MM-DD format.

            Returns:
                Dictionary with list of available flights.
            """
            return {}


        def search_hotels(city: str, check_in: str, check_out: str) -> dict:
            """Search for available hotels in a city.

            Args:
                city: City name.
                check_in: Check-in date.
                check_out: Check-out date.

            Returns:
                Dictionary with list of available hotels.
            """
            return {}


        flight_agent = Agent(
            name="flight_agent",
            model="gemini-2.0-flash",
            description="Flight specialist",
            instruction="You search flights.",
            tools=[search_flights],
        )

        hotel_agent = Agent(
            name="hotel_agent",
            model="gemini-2.0-flash",
            description="Hotel specialist",
            instruction="You search hotels.",
            tools=[search_hotels],
        )

        root_agent = Agent(
            name="travel_coordinator",
            model="gemini-2.0-flash",
            description="A travel coordinator.",
            instruction="You coordinate flights and hotels.",
            sub_agents=[flight_agent, hotel_agent],
        )
    '''))
    return src


@pytest.fixture
def sequential_agent_source(tmp_path) -> Path:
    src = tmp_path / "seq_agent.py"
    src.write_text(textwrap.dedent('''\
        from google.adk.agents import Agent, SequentialAgent

        step1 = Agent(
            name="step1",
            model="gemini-2.0-flash",
            instruction="First step.",
            output_key="step1_output",
        )

        step2 = Agent(
            name="step2",
            model="gemini-2.0-flash",
            instruction="Second step.",
        )

        root_agent = SequentialAgent(
            name="pipeline",
            description="A sequential pipeline",
            sub_agents=[step1, step2],
        )
    '''))
    return src


def test_parse_simple_agent(simple_agent_source):
    metadata = parse_agent_from_source(str(simple_agent_source))
    assert isinstance(metadata, AgentMetadata)
    assert metadata.name == "simple_agent"
    assert metadata.agent_type == "LlmAgent"
    assert metadata.description == "A simple test agent"
    assert "helpful assistant" in metadata.instruction
    assert metadata.model == "gemini-2.0-flash"
    assert len(metadata.tools) == 1
    assert metadata.tools[0].name == "search"
    assert metadata.tools[0].source == "function"
    assert "query" in str(metadata.tools[0].parameters_schema)


def test_parse_multi_agent(multi_agent_source):
    metadata = parse_agent_from_source(str(multi_agent_source))
    assert metadata.name == "travel_coordinator"
    assert len(metadata.sub_agents) == 2
    flight = metadata.sub_agents[0]
    assert flight.name == "flight_agent"
    assert len(flight.tools) == 1
    assert flight.tools[0].name == "search_flights"
    hotel = metadata.sub_agents[1]
    assert hotel.name == "hotel_agent"
    assert len(hotel.tools) == 1
    assert hotel.tools[0].name == "search_hotels"


def test_parse_sequential_agent(sequential_agent_source):
    metadata = parse_agent_from_source(str(sequential_agent_source))
    assert metadata.name == "pipeline"
    assert metadata.agent_type == "SequentialAgent"
    assert len(metadata.sub_agents) == 2
    assert metadata.sub_agents[0].output_key == "step1_output"


def test_parse_tool_schema(simple_agent_source):
    metadata = parse_agent_from_source(str(simple_agent_source))
    tool = metadata.tools[0]
    schema = tool.parameters_schema
    assert "properties" in schema
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["type"] == "string"
    assert "required" in schema
    assert "query" in schema["required"]


def test_parse_tool_with_default(tmp_path):
    src = tmp_path / "agent_with_defaults.py"
    src.write_text(textwrap.dedent('''\
        from google.adk.agents import Agent

        def summarize(text: str, max_length: int = 100) -> str:
            """Summarize text."""
            return text[:max_length]

        root_agent = Agent(
            name="summarizer",
            model="gemini-2.0-flash",
            instruction="Summarize things.",
            tools=[summarize],
        )
    '''))
    metadata = parse_agent_from_source(str(src))
    tool = metadata.tools[0]
    schema = tool.parameters_schema
    assert "text" in schema.get("required", [])
    assert "max_length" not in schema.get("required", [])
    assert schema["properties"]["max_length"]["default"] == 100


def test_parse_save_to_file(simple_agent_source, tmp_path):
    output_path = tmp_path / "out" / "metadata.json"
    metadata = parse_agent_from_source(
        str(simple_agent_source), save_path=str(output_path)
    )
    assert output_path.exists()
    loaded = json.loads(output_path.read_text())
    assert loaded["name"] == "simple_agent"


def test_parse_returns_dict(simple_agent_source):
    metadata = parse_agent_from_source(str(simple_agent_source))
    as_dict = metadata.model_dump()
    assert isinstance(as_dict, dict)
    assert as_dict["name"] == "simple_agent"


def test_parse_raises_on_missing_variable(simple_agent_source):
    with pytest.raises(ValueError, match="not_here"):
        parse_agent_from_source(str(simple_agent_source), agent_variable="not_here")


def test_parse_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_agent_from_source("/nonexistent/file.py")


def test_parse_real_weather_agent():
    """Parse the bundled weather_agent example from source."""
    src = str(Path(__file__).resolve().parent.parent / "examples" / "weather_agent" / "agent.py")
    metadata = parse_agent_from_source(src)
    assert metadata.name == "weather_agent"
    assert metadata.agent_type == "LlmAgent"
    assert len(metadata.tools) == 2
    tool_names = {t.name for t in metadata.tools}
    assert "get_weather" in tool_names
    assert "get_forecast" in tool_names


def test_parse_real_travel_multi_agent():
    """Parse the bundled travel_multi_agent example from source."""
    src = str(Path(__file__).resolve().parent.parent / "examples" / "travel_multi_agent" / "agent.py")
    metadata = parse_agent_from_source(src)
    assert metadata.name == "travel_coordinator"
    assert len(metadata.sub_agents) == 2
    sub_names = {a.name for a in metadata.sub_agents}
    assert "flight_agent" in sub_names
    assert "hotel_agent" in sub_names
