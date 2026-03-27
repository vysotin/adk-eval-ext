"""Resolve MCP toolset tools into ToolMetadata."""

from __future__ import annotations

from typing import Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset

from adk_eval_tool.schemas import ToolMetadata


async def resolve_mcp_toolset(
    toolset: BaseToolset,
    server_name: str = "unknown",
) -> list[ToolMetadata]:
    """Resolve all tools from an MCP toolset.

    Args:
        toolset: An MCPToolset or other BaseToolset instance.
        server_name: Name identifier for the MCP server.

    Returns:
        List of ToolMetadata with source="mcp".
    """
    try:
        tools: list[BaseTool] = await toolset.get_tools()
    except Exception as e:
        return [ToolMetadata(
            name=f"unresolved:{server_name}",
            description=f"Error resolving MCP tools: {e}",
            parameters_schema={},
            source="mcp",
            mcp_server_name=server_name,
        )]

    result = []
    for tool in tools:
        schema = _extract_tool_schema(tool)
        result.append(ToolMetadata(
            name=tool.name,
            description=tool.description or "",
            parameters_schema=schema,
            source="mcp",
            mcp_server_name=server_name,
        ))
    return result


def _extract_tool_schema(tool: BaseTool) -> dict:
    """Extract parameter schema from a BaseTool."""
    try:
        decl = tool._get_declaration()
        if decl and decl.parameters:
            schema = {"type": "object", "properties": {}}
            if hasattr(decl.parameters, "properties") and decl.parameters.properties:
                for prop_name, prop_schema in decl.parameters.properties.items():
                    schema["properties"][prop_name] = {
                        "type": getattr(prop_schema, "type", "string"),
                    }
            if hasattr(decl.parameters, "required") and decl.parameters.required:
                schema["required"] = list(decl.parameters.required)
            return schema
    except Exception:
        pass
    return {}
