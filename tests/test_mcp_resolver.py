"""Tests for MCP tool resolution."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from adk_eval_tool.agent_parser.mcp_resolver import resolve_mcp_toolset
from adk_eval_tool.schemas import ToolMetadata


@pytest.mark.asyncio
async def test_resolve_mcp_toolset_returns_tool_metadata():
    mock_tool = MagicMock()
    mock_tool.name = "read_file"
    mock_tool.description = "Read a file from disk"
    mock_tool._get_declaration.return_value = None

    mock_toolset = AsyncMock()
    mock_toolset.get_tools = AsyncMock(return_value=[mock_tool])

    tools = await resolve_mcp_toolset(mock_toolset, server_name="filesystem")
    assert len(tools) == 1
    assert tools[0].name == "read_file"
    assert tools[0].source == "mcp"
    assert tools[0].mcp_server_name == "filesystem"


@pytest.mark.asyncio
async def test_resolve_mcp_toolset_handles_failure():
    mock_toolset = AsyncMock()
    mock_toolset.get_tools = AsyncMock(side_effect=Exception("Connection refused"))

    tools = await resolve_mcp_toolset(mock_toolset, server_name="broken_server")
    assert len(tools) == 1
    assert "unresolved" in tools[0].name.lower() or "error" in tools[0].description.lower()


@pytest.mark.asyncio
async def test_resolve_mcp_toolset_multiple_tools():
    mock_tools = []
    for name, desc in [("read", "Read file"), ("write", "Write file"), ("list", "List dir")]:
        t = MagicMock()
        t.name = name
        t.description = desc
        t._get_declaration.return_value = None
        mock_tools.append(t)

    mock_toolset = AsyncMock()
    mock_toolset.get_tools = AsyncMock(return_value=mock_tools)

    tools = await resolve_mcp_toolset(mock_toolset, server_name="fs")
    assert len(tools) == 3
    assert all(t.source == "mcp" for t in tools)
    assert all(t.mcp_server_name == "fs" for t in tools)
