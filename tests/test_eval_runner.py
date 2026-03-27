"""Tests for eval runner trace collection and result capture."""

import pytest
from adk_eval_tool.eval_runner.trace_collector import (
    build_trace_tree,
    compute_basic_metrics,
    SpanData,
)
from adk_eval_tool.schemas import TraceSpanNode, BasicMetrics


def _make_span(span_id, name, parent_span_id=None, session_id="sess-1",
               start_time=1000, end_time=1001, attributes=None):
    return SpanData(
        span_id=span_id,
        name=name,
        parent_span_id=parent_span_id,
        start_time=start_time,
        end_time=end_time,
        session_id=session_id,
        attributes=attributes or {},
    )


def test_build_trace_tree_single_root():
    spans = [_make_span("s1", "invocation")]
    tree = build_trace_tree(spans)
    assert len(tree) == 1
    assert tree[0].name == "invocation"
    assert tree[0].span_id == "s1"


def test_build_trace_tree_nested():
    spans = [
        _make_span("s1", "invocation"),
        _make_span("s2", "call_llm", parent_span_id="s1", start_time=1000, end_time=1001),
        _make_span("s3", "execute_tool:search", parent_span_id="s1", start_time=1001, end_time=1002),
        _make_span("s4", "call_llm", parent_span_id="s1", start_time=1002, end_time=1003),
    ]
    tree = build_trace_tree(spans)
    assert len(tree) == 1
    root = tree[0]
    assert root.name == "invocation"
    assert len(root.children) == 3
    assert root.children[0].name == "call_llm"
    assert root.children[1].name == "execute_tool:search"


def test_build_trace_tree_deep_nesting():
    spans = [
        _make_span("s1", "invocation"),
        _make_span("s2", "call_llm", parent_span_id="s1"),
        _make_span("s3", "execute_tool:search", parent_span_id="s2"),
    ]
    tree = build_trace_tree(spans)
    root = tree[0]
    assert len(root.children) == 1
    llm = root.children[0]
    assert llm.name == "call_llm"
    assert len(llm.children) == 1
    assert llm.children[0].name == "execute_tool:search"


def test_build_trace_tree_empty():
    tree = build_trace_tree([])
    assert tree == []


def test_compute_basic_metrics():
    spans = [
        _make_span("s1", "invocation", start_time=0, end_time=5,
                    attributes={}),
        _make_span("s2", "call_llm", parent_span_id="s1", start_time=0, end_time=2,
                    attributes={"gen_ai.usage.input_tokens": "500", "gen_ai.usage.output_tokens": "100"}),
        _make_span("s3", "execute_tool:search", parent_span_id="s1", start_time=2, end_time=3),
        _make_span("s4", "call_llm", parent_span_id="s1", start_time=3, end_time=5,
                    attributes={"gen_ai.usage.input_tokens": "800", "gen_ai.usage.output_tokens": "150"}),
    ]
    tree = build_trace_tree(spans)
    metrics = compute_basic_metrics(tree[0])
    assert isinstance(metrics, BasicMetrics)
    assert metrics.num_llm_calls == 2
    assert metrics.num_tool_calls == 1
    assert metrics.total_input_tokens == 1300
    assert metrics.total_output_tokens == 250
    assert metrics.total_tokens == 1550
    assert metrics.max_context_size == 800
    assert metrics.total_duration_ms == 5000.0
