"""OpenTelemetry trace collection for ADK evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from adk_eval_tool.schemas import TraceSpanNode, BasicMetrics


@dataclass
class SpanData:
    """Lightweight span representation extracted from OTel ReadableSpan."""

    span_id: str
    name: str
    parent_span_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    session_id: str = ""
    invocation_id: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


def extract_span_data_from_readable(span) -> SpanData:
    """Convert an OTel ReadableSpan to SpanData.

    Args:
        span: An opentelemetry.sdk.trace.ReadableSpan instance.

    Returns:
        SpanData with extracted fields.
    """
    attrs = dict(span.attributes) if span.attributes else {}
    return SpanData(
        span_id=format(span.context.span_id, "016x"),
        name=span.name,
        parent_span_id=(
            format(span.parent.span_id, "016x") if span.parent else None
        ),
        start_time=span.start_time / 1e9 if span.start_time else 0.0,
        end_time=span.end_time / 1e9 if span.end_time else 0.0,
        session_id=attrs.get("gcp.vertex.agent.session_id", ""),
        invocation_id=attrs.get("gcp.vertex.agent.invocation_id", ""),
        attributes=attrs,
    )


def build_trace_tree(spans: list[SpanData]) -> list[TraceSpanNode]:
    """Build a tree of TraceSpanNode from flat span list.

    Args:
        spans: List of SpanData (flat, with parent_span_id references).

    Returns:
        List of root TraceSpanNode (usually one per invocation).
    """
    if not spans:
        return []

    nodes: dict[str, TraceSpanNode] = {}
    for span in spans:
        nodes[span.span_id] = TraceSpanNode(
            span_id=span.span_id,
            name=span.name,
            start_time=span.start_time,
            end_time=span.end_time,
            attributes=span.attributes,
        )

    roots = []
    for span in spans:
        node = nodes[span.span_id]
        if span.parent_span_id and span.parent_span_id in nodes:
            nodes[span.parent_span_id].children.append(node)
        else:
            roots.append(node)

    return roots


def setup_trace_collection(db_path: str):
    """Set up OTel with SqliteSpanExporter for trace persistence.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        The SqliteSpanExporter instance (for later span retrieval).
    """
    from pathlib import Path as _Path
    from google.adk.telemetry.sqlite_span_exporter import SqliteSpanExporter
    from google.adk.telemetry.setup import maybe_set_otel_providers, OTelHooks
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    exporter = SqliteSpanExporter(db_path=db_path)
    hooks = OTelHooks(span_processors=[BatchSpanProcessor(exporter)])
    maybe_set_otel_providers(otel_hooks_to_setup=[hooks])
    return exporter


def get_trace_tree_for_session(exporter, session_id: str) -> list[TraceSpanNode]:
    """Retrieve and build trace tree for a session from SqliteSpanExporter.

    Args:
        exporter: SqliteSpanExporter instance.
        session_id: The session ID to retrieve spans for.

    Returns:
        List of root TraceSpanNode for the session.
    """
    raw_spans = exporter.get_all_spans_for_session(session_id)
    span_data = [extract_span_data_from_readable(s) for s in raw_spans]
    return build_trace_tree(span_data)


def compute_basic_metrics(trace_tree: TraceSpanNode) -> BasicMetrics:
    """Compute standard metrics from a trace tree.

    Walks the trace tree to extract token counts, call counts, durations,
    and response sizes from span attributes.

    Args:
        trace_tree: Root TraceSpanNode.

    Returns:
        BasicMetrics with aggregated values.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    num_llm_calls = 0
    num_tool_calls = 0
    max_context_size = 0

    def _walk(node: TraceSpanNode):
        nonlocal total_input_tokens, total_output_tokens, num_llm_calls
        nonlocal num_tool_calls, max_context_size

        if "call_llm" in node.name:
            num_llm_calls += 1
            inp = int(node.attributes.get("gen_ai.usage.input_tokens", 0))
            out = int(node.attributes.get("gen_ai.usage.output_tokens", 0))
            total_input_tokens += inp
            total_output_tokens += out
            if inp > max_context_size:
                max_context_size = inp
        elif "execute_tool" in node.name:
            num_tool_calls += 1

        for child in node.children:
            _walk(child)

    _walk(trace_tree)

    total_duration_ms = (trace_tree.end_time - trace_tree.start_time) * 1000

    return BasicMetrics(
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_input_tokens + total_output_tokens,
        num_llm_calls=num_llm_calls,
        num_tool_calls=num_tool_calls,
        total_duration_ms=max(0, total_duration_ms),
        max_context_size=max_context_size,
    )
