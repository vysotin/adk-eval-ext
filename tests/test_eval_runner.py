"""Tests for eval runner trace collection, result capture, and sanitisation."""

import json

import pytest
from adk_eval_tool.eval_runner.trace_collector import (
    build_trace_tree,
    compute_basic_metrics,
    SpanData,
)
from adk_eval_tool.eval_runner.runner import _sanitize_eval_set, _coerce_to_dict
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


# ---------------------------------------------------------------------------
# Sanitisation tests
# ---------------------------------------------------------------------------


def test_coerce_to_dict_from_dict():
    assert _coerce_to_dict({"a": 1}) == {"a": 1}


def test_coerce_to_dict_from_json_string():
    assert _coerce_to_dict('{"flights": []}') == {"flights": []}


def test_coerce_to_dict_from_plain_string():
    assert _coerce_to_dict("some text") == {"result": "some text"}


def test_coerce_to_dict_from_none():
    assert _coerce_to_dict(None) == {"result": "None"}


def test_sanitize_string_tool_response():
    """Tool responses with string 'response' values get parsed to dicts."""
    data = {
        "eval_cases": [{
            "eval_id": "test",
            "conversation": [{
                "invocation_id": "inv-1",
                "intermediate_data": {
                    "tool_uses": [{"name": "search_flights", "args": {"origin": "London"}}],
                    "tool_responses": [{
                        "name": "search_flights",
                        "id": "call_1",
                        "response": '{"flights": [{"id": "FL-101", "price": 200}]}',
                    }],
                    "intermediate_responses": [],
                },
            }],
        }],
    }
    result = _sanitize_eval_set(data)
    tr = result["eval_cases"][0]["conversation"][0]["intermediate_data"]["tool_responses"][0]
    assert isinstance(tr["response"], dict)
    assert tr["response"]["flights"][0]["id"] == "FL-101"


def test_sanitize_preserves_dict_tool_response():
    """Tool responses that are already dicts are kept as-is."""
    data = {
        "eval_cases": [{
            "eval_id": "test",
            "conversation": [{
                "invocation_id": "inv-1",
                "intermediate_data": {
                    "tool_uses": [],
                    "tool_responses": [{
                        "name": "get_weather",
                        "response": {"output": 15, "condition": "Cloudy"},
                    }],
                    "intermediate_responses": [],
                },
            }],
        }],
    }
    result = _sanitize_eval_set(data)
    tr = result["eval_cases"][0]["conversation"][0]["intermediate_data"]["tool_responses"][0]
    assert tr["response"] == {"output": 15, "condition": "Cloudy"}


def test_sanitize_strips_extra_fields():
    """Extra fields in tool_uses and tool_responses are removed."""
    data = {
        "eval_cases": [{
            "eval_id": "test",
            "conversation": [{
                "invocation_id": "inv-1",
                "intermediate_data": {
                    "tool_uses": [{"name": "search", "args": {}, "extra_field": "bad"}],
                    "tool_responses": [{"name": "search", "response": {}, "error": "should be removed"}],
                    "intermediate_responses": [],
                },
            }],
        }],
    }
    result = _sanitize_eval_set(data)
    idata = result["eval_cases"][0]["conversation"][0]["intermediate_data"]
    assert "extra_field" not in idata["tool_uses"][0]
    assert "error" not in idata["tool_responses"][0]


def test_sanitize_moves_misplaced_fields_into_intermediate_data():
    """tool_responses / intermediate_responses at invocation level get relocated."""
    data = {
        "eval_cases": [{
            "eval_id": "test",
            "conversation": [{
                "invocation_id": "inv-1",
                "user_content": {"role": "user", "parts": [{"text": "hello"}]},
                "tool_responses": [],
                "intermediate_responses": [],
            }],
        }],
    }
    result = _sanitize_eval_set(data)
    inv = result["eval_cases"][0]["conversation"][0]
    # Fields should be moved into intermediate_data
    assert "tool_responses" not in inv or inv.get("intermediate_data") is not None
    assert "intermediate_responses" not in inv or inv.get("intermediate_data") is not None
    idata = inv.get("intermediate_data", {})
    assert idata.get("tool_responses") == []
    assert idata.get("intermediate_responses") == []


def test_sanitize_misplaced_fields_validates_with_adk():
    """Invocation with misplaced fields passes ADK validation after sanitisation."""
    from google.adk.evaluation.eval_set import EvalSet

    data = {
        "eval_set_id": "test__misplaced",
        "eval_cases": [{
            "eval_id": "case_1",
            "conversation": [{
                "invocation_id": "inv-1",
                "user_content": {"role": "user", "parts": [{"text": "Find flights"}]},
                "tool_responses": [],
                "intermediate_responses": [],
                "final_response": {"role": "model", "parts": [{"text": "No flights found."}]},
            }],
        }],
    }
    sanitized = _sanitize_eval_set(data)
    eval_set = EvalSet.model_validate(sanitized)
    assert len(eval_set.eval_cases) == 1


def test_sanitize_validates_with_adk():
    """Sanitised data passes ADK EvalSet validation (the actual error scenario)."""
    from google.adk.evaluation.eval_set import EvalSet

    # Simulates what the LLM generates for travel_multi_agent
    data = {
        "eval_set_id": "travel__test",
        "eval_cases": [{
            "eval_id": "flight_test",
            "conversation": [{
                "invocation_id": "inv-1",
                "user_content": {"role": "user", "parts": [{"text": "Find flights"}]},
                "intermediate_data": {
                    "tool_uses": [
                        {"name": "flight_agent", "args": {"origin": "London", "destination": "Paris", "date": "2024-07-15"}},
                    ],
                    "tool_responses": [{
                        "name": "flight_agent",
                        "id": "call_1",
                        "response": '{"flights": [{"flight_id": "FL-101", "price": 200}]}',
                    }],
                    "intermediate_responses": [],
                },
                "final_response": {"role": "model", "parts": [{"text": "Found flights."}]},
            }],
        }],
    }
    sanitized = _sanitize_eval_set(data)
    # Should not raise
    eval_set = EvalSet.model_validate(sanitized)
    assert eval_set.eval_set_id == "travel__test"
    assert len(eval_set.eval_cases) == 1
