"""Tests for agent parser."""

from adk_eval_tool.agent_parser.parser import parse_agent
from adk_eval_tool.schemas import AgentMetadata


def test_parse_simple_agent(simple_agent):
    metadata = parse_agent(simple_agent)
    assert isinstance(metadata, AgentMetadata)
    assert metadata.name == "simple_agent"
    assert metadata.agent_type == "LlmAgent"
    assert metadata.description == "A simple test agent"
    assert "helpful assistant" in metadata.instruction
    assert metadata.model == "gemini-2.0-flash"
    assert len(metadata.tools) == 1
    assert metadata.tools[0].name == "_dummy_tool"
    assert metadata.tools[0].source == "function"
    assert "query" in str(metadata.tools[0].parameters_schema)


def test_parse_agent_with_sub_agents(agent_with_sub_agents):
    metadata = parse_agent(agent_with_sub_agents)
    assert metadata.name == "coordinator"
    assert len(metadata.sub_agents) == 2
    researcher = metadata.sub_agents[0]
    assert researcher.name == "researcher"
    assert len(researcher.tools) == 1
    assert researcher.tools[0].name == "_dummy_tool"
    writer = metadata.sub_agents[1]
    assert writer.name == "writer"
    assert len(writer.tools) == 1
    assert writer.tools[0].name == "_another_tool"


def test_parse_sequential_agent(sequential_agent):
    metadata = parse_agent(sequential_agent)
    assert metadata.name == "pipeline"
    assert metadata.agent_type == "SequentialAgent"
    assert len(metadata.sub_agents) == 2
    assert metadata.sub_agents[0].output_key == "step1_output"


def test_parse_agent_tool_schema(simple_agent):
    metadata = parse_agent(simple_agent)
    tool = metadata.tools[0]
    schema = tool.parameters_schema
    assert "properties" in schema
    assert "query" in schema["properties"]


def test_parse_returns_dict(simple_agent):
    metadata = parse_agent(simple_agent)
    as_dict = metadata.model_dump()
    assert isinstance(as_dict, dict)
    assert as_dict["name"] == "simple_agent"


def test_parse_save_to_file(simple_agent, tmp_path):
    output_path = tmp_path / "metadata.json"
    metadata = parse_agent(simple_agent, save_path=str(output_path))
    assert output_path.exists()
    import json
    loaded = json.loads(output_path.read_text())
    assert loaded["name"] == "simple_agent"
