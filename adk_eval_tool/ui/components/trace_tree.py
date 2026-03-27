"""Trace tree visualization component for Streamlit."""

from __future__ import annotations

import streamlit as st

from adk_eval_tool.schemas import TraceSpanNode


def render_trace_tree(node: TraceSpanNode, level: int = 0):
    """Render a trace span tree as nested Streamlit expanders."""
    duration_ms = (node.end_time - node.start_time) * 1000 if node.end_time > node.start_time else 0

    if "call_llm" in node.name:
        icon = "\U0001f9e0"
        label = f"LLM Call ({duration_ms:.0f}ms)"
    elif "execute_tool" in node.name:
        icon = "\U0001f527"
        tool_name = node.name.replace("execute_tool:", "").replace("execute_tool", "tool")
        label = f"Tool: {tool_name} ({duration_ms:.0f}ms)"
    elif "invocation" in node.name:
        icon = "\U0001f4cb"
        label = f"Invocation ({duration_ms:.0f}ms)"
    elif "send_data" in node.name:
        icon = "\U0001f4e4"
        label = f"Send Data ({duration_ms:.0f}ms)"
    else:
        icon = "\U0001f4ce"
        label = f"{node.name} ({duration_ms:.0f}ms)"

    if node.children:
        with st.expander(f"{'  ' * level}{icon} {label}", expanded=(level < 2)):
            _render_span_attributes(node)
            for child in node.children:
                render_trace_tree(child, level + 1)
    else:
        st.markdown(f"{'&nbsp;&nbsp;' * level * 2}{icon} {label}")
        _render_span_attributes(node)


def _render_span_attributes(node: TraceSpanNode):
    """Render relevant span attributes."""
    attrs = node.attributes
    interesting_keys = [
        "gcp.vertex.agent.invocation_id",
        "gcp.vertex.agent.event_id",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.response.finish_reasons",
    ]
    shown = {k: v for k, v in attrs.items() if k in interesting_keys}
    if shown:
        cols = st.columns(len(shown))
        for i, (key, value) in enumerate(shown.items()):
            short_key = key.split(".")[-1]
            cols[i].caption(f"{short_key}: {value}")


def render_trace_summary(nodes: list[TraceSpanNode]):
    """Render a summary of the trace tree."""
    if not nodes:
        st.info("No trace data available.")
        return

    llm_calls = 0
    tool_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    def _count(node: TraceSpanNode):
        nonlocal llm_calls, tool_calls, total_input_tokens, total_output_tokens
        if "call_llm" in node.name:
            llm_calls += 1
            total_input_tokens += int(node.attributes.get("gen_ai.usage.input_tokens", 0))
            total_output_tokens += int(node.attributes.get("gen_ai.usage.output_tokens", 0))
        elif "execute_tool" in node.name:
            tool_calls += 1
        for child in node.children:
            _count(child)

    for root in nodes:
        _count(root)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM Calls", llm_calls)
    col2.metric("Tool Calls", tool_calls)
    col3.metric("Input Tokens", total_input_tokens)
    col4.metric("Output Tokens", total_output_tokens)
