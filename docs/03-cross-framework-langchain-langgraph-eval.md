# Task 3: Extending ADK Evaluation Framework to Langchain and Langgraph Agents

## Problem Statement

Google ADK's evaluation framework is tightly coupled to ADK agents. This report analyzes options to extend it so that Langchain and Langgraph agents can be evaluated using the same infrastructure: OpenTelemetry trace collection, tool/sub-agent mocking (from Task 1), and separated inference/evaluation (from Task 2).

---

## Current Framework Coupling Points

### ADK-Specific Dependencies in Eval Pipeline

| Component | ADK Dependency | What Needs Abstraction |
|-----------|---------------|----------------------|
| `EvaluationGenerator` | `InMemoryRunner(agent)` | Agent execution interface |
| `InferenceResult` | ADK `Invocation`, `Content`, `FunctionCall` types | Data model |
| `TrajectoryEvaluator` | `Invocation.intermediate_data.tool_uses` | Tool call extraction |
| `ResponseMatchEvaluator` | `Invocation.final_response` | Response extraction |
| Mock injection | `before_tool_callback` on ADK `Agent` | Tool interception mechanism |
| `adk web` UI | ADK event stream, `EvalCase`, `EvalSet` models | API/data format |
| CLI (`adk eval`) | Agent module with `root_agent` | Agent loading |

### Langchain/Langgraph Equivalents

| ADK Concept | Langchain Equivalent | Langgraph Equivalent |
|-------------|---------------------|---------------------|
| `Agent` | `AgentExecutor`, `create_react_agent` | `StateGraph.compile()` |
| `before_tool_callback` | `wrap_tool_call` middleware | `interrupt_before` + state injection |
| `sub_agents` | Nested chains/agents | Subgraph nodes |
| `InMemoryRunner` | `chain.invoke()` / `agent.astream()` | `graph.ainvoke()` / `graph.astream()` |
| `Invocation` | Run result dict | `ThreadState` snapshot |
| OTel spans | `opentelemetry-instrumentation-langchain` | Same (via LangSmith OTEL) |
| `Content` / `Part` | `AIMessage`, `HumanMessage`, `ToolMessage` | Same (uses LangChain messages) |

---

## Options Analysis

### Option A: Universal Trace-Based Evaluation (Recommended)

Standardize on OpenTelemetry traces as the universal data format. All frameworks export traces; evaluation operates on normalized trace data.

**Architecture:**

```
                    INFERENCE PHASE
     ┌──────────────────────────────────────────┐
     │                                          │
     │  ┌─────────┐  ┌───────────┐  ┌────────┐ │
     │  │ ADK     │  │ LangChain │  │ LangGr.│ │
     │  │ Agent   │  │ Agent     │  │ Graph  │ │
     │  └────┬────┘  └─────┬─────┘  └───┬────┘ │
     │       │OTel         │OTel        │OTel   │
     │       └──────┬──────┘────────────┘       │
     │              ▼                           │
     │     ┌────────────────┐                   │
     │     │ OTel Collector │                   │
     │     │ or FileExporter│                   │
     │     └───────┬────────┘                   │
     │             ▼                            │
     │     traces.jsonl + inference_results.json│
     └──────────────────────────────────────────┘
                         │
                    EVALUATION PHASE
     ┌───────────────────┼──────────────────────┐
     │                   ▼                      │
     │  ┌────────────────────────────┐          │
     │  │  Trace Normalizer          │          │
     │  │  ├─ ADK span parser       │          │
     │  │  ├─ LangChain span parser │          │
     │  │  ├─ LangGraph span parser │          │
     │  │  └─► UnifiedInvocation[]  │          │
     │  └───────────────┬────────────┘          │
     │                  ▼                       │
     │  ┌────────────────────────────┐          │
     │  │  Evaluators                │          │
     │  │  ├─ TrajectoryEvaluator   │          │
     │  │  ├─ ResponseMatchEval     │          │
     │  │  ├─ LLM-judged evals     │          │
     │  │  ├─ Custom metrics        │          │
     │  │  └─ DeepEval / Ragas      │          │
     │  └────────────────────────────┘          │
     └──────────────────────────────────────────┘
```

**Normalized Data Model:**

```python
# unified_eval_types.py -- Framework-agnostic evaluation data model

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

class FrameworkType(Enum):
    ADK = "adk"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    GENERIC = "generic"

@dataclass
class UnifiedToolCall:
    """Framework-agnostic tool call representation."""
    tool_name: str
    arguments: dict[str, Any]
    result: Optional[Any] = None
    duration_ms: Optional[float] = None
    span_id: Optional[str] = None  # Link back to OTel span

@dataclass
class UnifiedAgentCall:
    """Framework-agnostic sub-agent/subgraph call representation."""
    agent_name: str
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    tool_calls: list[UnifiedToolCall] = field(default_factory=list)
    duration_ms: Optional[float] = None
    span_id: Optional[str] = None

@dataclass
class UnifiedInvocation:
    """Framework-agnostic single-turn invocation."""
    invocation_id: str = ""
    user_input: str = ""
    final_response: str = ""
    tool_calls: list[UnifiedToolCall] = field(default_factory=list)
    agent_calls: list[UnifiedAgentCall] = field(default_factory=list)
    trace_id: Optional[str] = None
    framework: FrameworkType = FrameworkType.GENERIC
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata can hold: token counts, latency, model name, etc.

@dataclass
class UnifiedEvalCase:
    """Framework-agnostic test case."""
    eval_id: str
    expected_invocations: list[UnifiedInvocation]
    actual_invocations: Optional[list[UnifiedInvocation]] = None
    mock_responses: Optional[dict] = None  # From Task 1
    session_state: Optional[dict] = None
    rubrics: Optional[list[dict]] = None
```

**Trace Normalizers:**

```python
# normalizers.py -- Convert framework-specific traces to UnifiedInvocation

class ADKTraceNormalizer:
    """Parse ADK OTel spans into UnifiedInvocation objects."""

    # ADK span names: invocation, agent_run, call_llm, execute_tool

    def normalize(self, spans: list[dict]) -> list[UnifiedInvocation]:
        invocations = []

        # Group spans by parent invocation span
        inv_spans = [s for s in spans if s["name"] == "invocation"]

        for inv_span in inv_spans:
            trace_id = inv_span["traceId"]
            span_id = inv_span["spanId"]

            # Find child tool spans
            tool_spans = [
                s for s in spans
                if s.get("parentSpanId") == span_id
                and s["name"].startswith("execute_tool")
            ]

            tool_calls = []
            for ts in tool_spans:
                attrs = self._parse_attrs(ts)
                tool_calls.append(UnifiedToolCall(
                    tool_name=attrs.get("tool.name", ts["name"]),
                    arguments=json.loads(attrs.get("tool.args", "{}")),
                    result=json.loads(attrs.get("tool.result", "null")),
                    duration_ms=self._duration_ms(ts),
                    span_id=ts["spanId"],
                ))

            # Find agent_run child spans (sub-agent delegations)
            agent_spans = [
                s for s in spans
                if s.get("parentSpanId") == span_id
                and s["name"] == "agent_run"
            ]

            agent_calls = []
            for as_ in agent_spans:
                attrs = self._parse_attrs(as_)
                agent_calls.append(UnifiedAgentCall(
                    agent_name=attrs.get("agent.name", "unknown"),
                    output_text=attrs.get("agent.response", ""),
                    duration_ms=self._duration_ms(as_),
                    span_id=as_["spanId"],
                ))

            attrs = self._parse_attrs(inv_span)
            invocations.append(UnifiedInvocation(
                invocation_id=span_id,
                user_input=attrs.get("user.input", ""),
                final_response=attrs.get("agent.response", ""),
                tool_calls=tool_calls,
                agent_calls=agent_calls,
                trace_id=trace_id,
                framework=FrameworkType.ADK,
                metadata={
                    "model": attrs.get("gen_ai.request.model", ""),
                    "input_tokens": int(attrs.get("gen_ai.usage.input_tokens", 0)),
                    "output_tokens": int(attrs.get("gen_ai.usage.output_tokens", 0)),
                },
            ))

        return invocations

    def _parse_attrs(self, span):
        return {a["key"]: list(a["value"].values())[0] for a in span.get("attributes", [])}

    def _duration_ms(self, span):
        start = span.get("startTimeUnixNano", 0)
        end = span.get("endTimeUnixNano", 0)
        return (end - start) / 1e6


class LangChainTraceNormalizer:
    """Parse LangChain/LangGraph OTel spans into UnifiedInvocation objects."""

    # LangChain span attributes: langsmith.span.kind (llm, chain, tool, retriever)
    # LangGraph: each node is a span, edges are parent-child relationships

    def normalize(self, spans: list[dict]) -> list[UnifiedInvocation]:
        invocations = []

        # Find root chain spans (top-level agent invocations)
        root_spans = [
            s for s in spans
            if not s.get("parentSpanId")
            or self._get_attr(s, "langsmith.span.kind") == "chain"
        ]

        for root in root_spans:
            trace_id = root["traceId"]

            # Find all tool spans in this trace
            tool_spans = [
                s for s in spans
                if s["traceId"] == trace_id
                and self._get_attr(s, "langsmith.span.kind") == "tool"
            ]

            tool_calls = []
            for ts in tool_spans:
                attrs = self._parse_attrs(ts)
                tool_calls.append(UnifiedToolCall(
                    tool_name=ts.get("name", attrs.get("tool.name", "unknown")),
                    arguments=json.loads(attrs.get("tool.args", attrs.get("input", "{}"))),
                    result=json.loads(attrs.get("tool.result", attrs.get("output", "null"))),
                    duration_ms=self._duration_ms(ts),
                    span_id=ts["spanId"],
                ))

            # Find sub-chain/sub-agent spans (LangGraph nodes)
            chain_spans = [
                s for s in spans
                if s["traceId"] == trace_id
                and s.get("parentSpanId") == root["spanId"]
                and self._get_attr(s, "langsmith.span.kind") == "chain"
            ]

            agent_calls = []
            for cs in chain_spans:
                attrs = self._parse_attrs(cs)
                agent_calls.append(UnifiedAgentCall(
                    agent_name=cs.get("name", "unknown"),
                    input_text=attrs.get("input", ""),
                    output_text=attrs.get("output", ""),
                    duration_ms=self._duration_ms(cs),
                    span_id=cs["spanId"],
                ))

            # Extract user input and final response from root span
            root_attrs = self._parse_attrs(root)
            invocations.append(UnifiedInvocation(
                invocation_id=root["spanId"],
                user_input=root_attrs.get("input", ""),
                final_response=root_attrs.get("output", ""),
                tool_calls=tool_calls,
                agent_calls=agent_calls,
                trace_id=trace_id,
                framework=FrameworkType.LANGGRAPH if chain_spans else FrameworkType.LANGCHAIN,
                metadata={
                    "model": root_attrs.get("gen_ai.request.model", ""),
                },
            ))

        return invocations

    def _get_attr(self, span, key):
        for a in span.get("attributes", []):
            if a["key"] == key:
                return list(a["value"].values())[0]
        return None

    def _parse_attrs(self, span):
        return {a["key"]: list(a["value"].values())[0] for a in span.get("attributes", [])}

    def _duration_ms(self, span):
        start = span.get("startTimeUnixNano", 0)
        end = span.get("endTimeUnixNano", 0)
        return (end - start) / 1e6


class AutoNormalizer:
    """Auto-detect framework and apply appropriate normalizer."""

    def normalize(self, spans: list[dict]) -> list[UnifiedInvocation]:
        framework = self._detect_framework(spans)
        if framework == FrameworkType.ADK:
            return ADKTraceNormalizer().normalize(spans)
        elif framework in (FrameworkType.LANGCHAIN, FrameworkType.LANGGRAPH):
            return LangChainTraceNormalizer().normalize(spans)
        else:
            raise ValueError(f"Cannot auto-detect framework from spans")

    def _detect_framework(self, spans) -> FrameworkType:
        for span in spans:
            name = span.get("name", "")
            if name in ("invocation", "agent_run", "call_llm"):
                return FrameworkType.ADK
            attrs = {a["key"] for a in span.get("attributes", [])}
            if "langsmith.span.kind" in attrs:
                return FrameworkType.LANGCHAIN
        return FrameworkType.GENERIC
```

**Universal Evaluators:**

```python
# universal_evaluators.py -- Evaluators that work on UnifiedInvocation

class UniversalTrajectoryEvaluator:
    """Evaluate tool call trajectories across any framework."""

    def evaluate(
        self,
        actual: list[UnifiedInvocation],
        expected: list[UnifiedInvocation],
        match_type: str = "EXACT",  # EXACT, IN_ORDER, ANY_ORDER
    ) -> dict:
        results = []
        for act, exp in zip(actual, expected):
            actual_tools = [(tc.tool_name, tc.arguments) for tc in act.tool_calls]
            expected_tools = [(tc.tool_name, tc.arguments) for tc in exp.tool_calls]

            if match_type == "EXACT":
                score = 1.0 if actual_tools == expected_tools else 0.0
            elif match_type == "IN_ORDER":
                score = self._in_order_match(actual_tools, expected_tools)
            elif match_type == "ANY_ORDER":
                score = self._any_order_match(actual_tools, expected_tools)

            results.append({"invocation_id": act.invocation_id, "score": score})

        avg_score = sum(r["score"] for r in results) / len(results) if results else 0
        return {"overall_score": avg_score, "per_invocation": results}

    def _in_order_match(self, actual, expected):
        # Check if expected is a subsequence of actual
        j = 0
        for item in actual:
            if j < len(expected) and item == expected[j]:
                j += 1
        return 1.0 if j == len(expected) else j / max(len(expected), 1)

    def _any_order_match(self, actual, expected):
        actual_set = set(str(t) for t in actual)
        expected_set = set(str(t) for t in expected)
        if not expected_set:
            return 1.0
        return len(actual_set & expected_set) / len(expected_set)


class UniversalSubAgentEvaluator:
    """Evaluate sub-agent/subgraph delegation patterns."""

    def evaluate(
        self,
        actual: list[UnifiedInvocation],
        expected: list[UnifiedInvocation],
    ) -> dict:
        results = []
        for act, exp in zip(actual, expected):
            actual_agents = [ac.agent_name for ac in act.agent_calls]
            expected_agents = [ac.agent_name for ac in exp.agent_calls]

            # Check agent delegation sequence
            score = 1.0 if actual_agents == expected_agents else 0.0

            # Also check sub-agent outputs if provided
            if score == 1.0 and exp.agent_calls:
                output_scores = []
                for act_ac, exp_ac in zip(act.agent_calls, exp.agent_calls):
                    if exp_ac.output_text:
                        # Simple containment check; could use LLM judge
                        sim = self._text_similarity(act_ac.output_text, exp_ac.output_text)
                        output_scores.append(sim)
                if output_scores:
                    score = sum(output_scores) / len(output_scores)

            results.append({"invocation_id": act.invocation_id, "score": score})

        avg_score = sum(r["score"] for r in results) / len(results) if results else 0
        return {"overall_score": avg_score, "per_invocation": results}

    def _text_similarity(self, a, b):
        # Simplified; real implementation would use ROUGE or LLM judge
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not b_words:
            return 1.0
        return len(a_words & b_words) / len(b_words)
```

**Pros:**
- True framework-agnostic evaluation via normalized trace data
- Same evaluators work for ADK, LangChain, and LangGraph
- OTel traces are the universal interface -- any framework that emits OTel works
- Auto-detection simplifies user experience
- Builds on OTel GenAI semantic conventions (future-proof)

**Cons:**
- Trace normalization is lossy -- some framework-specific details may not map
- Semantic attribute differences require per-framework parsers
- More complex than framework-specific evaluation

---

### Option B: ADK Wrapper Agents for Langchain/Langgraph

Wrap Langchain/Langgraph agents in ADK `CustomAgent` or `FunctionTool` so they appear as ADK agents to the evaluation framework.

**Architecture:**

```
┌──────────────────────────────────────────┐
│  ADK EvalSet + ADK Evaluation Pipeline   │
│                                          │
│  ┌──────────────────────────────────┐    │
│  │  ADK CustomAgent (wrapper)       │    │
│  │  ├─ run_async_impl():           │    │
│  │  │   ├─ Convert ADK Content     │    │
│  │  │   │   → LangChain Message    │    │
│  │  │   ├─ langchain_agent.invoke()│    │
│  │  │   │   or graph.ainvoke()     │    │
│  │  │   ├─ Convert response back   │    │
│  │  │   │   → ADK Content          │    │
│  │  │   └─ Yield ADK Events        │    │
│  │  └─ before_tool_callback: mock  │    │
│  └──────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

**Prototype:**

```python
# langchain_wrapper.py -- Wrap LangChain agent as ADK CustomAgent

from google.adk.agents import CustomAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

class LangChainADKWrapper(CustomAgent):
    """Wraps a LangChain agent/chain to be evaluable by ADK eval framework."""

    def __init__(self, name: str, langchain_agent, tool_mocks: dict = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.langchain_agent = langchain_agent
        self.tool_mocks = tool_mocks or {}
        self._mock_middleware = None
        if tool_mocks:
            self._setup_mocking()

    def _setup_mocking(self):
        """Install mock middleware on the LangChain agent."""
        from langchain.agents.middleware import AgentMiddleware

        mocks = self.tool_mocks

        class MockMiddleware(AgentMiddleware):
            def wrap_tool_call(self, request, handler):
                tool_rules = mocks.get(request.tool_name, [])
                for rule in tool_rules:
                    match_args = rule.get("matchArgs", {})
                    if not match_args or all(
                        request.args.get(k) == v for k, v in match_args.items()
                    ):
                        return ToolMessage(
                            content=json.dumps(rule["response"]),
                            tool_call_id=request.tool_call_id,
                        )
                return handler(request)

        self._mock_middleware = MockMiddleware()

    async def run_async_impl(self, ctx: InvocationContext):
        """Execute LangChain agent and yield ADK-compatible events."""
        # Convert ADK user content to LangChain message
        user_text = ""
        for part in ctx.user_content.parts:
            if hasattr(part, "text"):
                user_text += part.text

        # Run LangChain agent
        if hasattr(self.langchain_agent, "astream"):
            # Streaming mode
            async for event in self.langchain_agent.astream(
                {"input": user_text},
                config={"callbacks": []},
            ):
                # Convert intermediate steps to ADK events
                if "intermediate_steps" in event:
                    for action, observation in event["intermediate_steps"]:
                        # Yield tool call event
                        yield types.Event(
                            actions=types.Content(parts=[
                                types.Part(function_call=types.FunctionCall(
                                    name=action.tool,
                                    args=action.tool_input,
                                ))
                            ])
                        )
                        # Yield tool response event
                        yield types.Event(
                            actions=types.Content(parts=[
                                types.Part(function_response=types.FunctionResponse(
                                    name=action.tool,
                                    response={"output": str(observation)},
                                ))
                            ])
                        )

                if "output" in event:
                    yield types.Event(
                        content=types.Content(parts=[
                            types.Part(text=event["output"])
                        ]),
                        is_final_response=True,
                    )
        else:
            # Non-streaming
            result = await self.langchain_agent.ainvoke({"input": user_text})
            yield types.Event(
                content=types.Content(parts=[
                    types.Part(text=result.get("output", str(result)))
                ]),
                is_final_response=True,
            )


class LangGraphADKWrapper(CustomAgent):
    """Wraps a LangGraph compiled graph to be evaluable by ADK eval framework."""

    def __init__(self, name: str, graph, tool_mocks: dict = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.graph = graph
        self.tool_mocks = tool_mocks or {}

    async def run_async_impl(self, ctx: InvocationContext):
        """Execute LangGraph and yield ADK-compatible events."""
        user_text = ""
        for part in ctx.user_content.parts:
            if hasattr(part, "text"):
                user_text += part.text

        # Run graph with streaming
        async for event in self.graph.astream(
            {"messages": [HumanMessage(content=user_text)]},
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                if node_name == "tools":
                    # Tool node output
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, ToolMessage):
                            yield types.Event(
                                actions=types.Content(parts=[
                                    types.Part(function_response=types.FunctionResponse(
                                        name=msg.name or "tool",
                                        response={"output": msg.content},
                                    ))
                                ])
                            )
                elif node_name == "agent":
                    # Agent node output
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    yield types.Event(
                                        actions=types.Content(parts=[
                                            types.Part(function_call=types.FunctionCall(
                                                name=tc["name"],
                                                args=tc["args"],
                                            ))
                                        ])
                                    )
                            elif msg.content:
                                yield types.Event(
                                    content=types.Content(parts=[
                                        types.Part(text=msg.content)
                                    ]),
                                    is_final_response=True,
                                )
```

**Usage:**

```python
# eval_langchain_agent.py

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from google.adk.evaluation import AgentEvaluator

# Create LangChain agent
llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, tools=[...], prompt=...)
executor = AgentExecutor(agent=agent, tools=[...])

# Wrap in ADK
wrapper = LangChainADKWrapper(
    name="langchain_weather_agent",
    langchain_agent=executor,
    tool_mocks={
        "get_weather": [{"matchArgs": {}, "response": {"temp": 72}}]
    },
)

# Use standard ADK evaluation
await AgentEvaluator.evaluate(
    agent_module_or_agent=wrapper,
    eval_dataset_file_path_or_dir="./eval_set.evalset.json",
)
```

**Pros:**
- Reuses 100% of ADK evaluation infrastructure (evaluators, CLI, Web UI)
- Tool mocking works via LangChain middleware (native to the framework)
- ADK Web UI can display wrapped agent's events normally
- Single eval dataset format works for both ADK and LangChain agents

**Cons:**
- Event conversion is lossy -- LangChain/LangGraph events don't map perfectly to ADK events
- Wrapper complexity grows with framework-specific features
- LangGraph subgraph nodes don't map cleanly to ADK sub-agents
- Maintaining wrapper compatibility across LangChain version updates is ongoing work

---

### Option C: External Eval Platform (Langfuse / Arize Phoenix)

Use a third-party platform that already supports both ADK and LangChain/LangGraph.

**Langfuse Architecture:**

```
┌─────────┐   OTEL   ┌──────────────┐   API   ┌───────────────┐
│ ADK     │──────────►│              │◄────────│ LangChain     │
│ Agent   │          │   Langfuse    │         │ Agent         │
└─────────┘          │              │         └───────────────┘
                     │  ├─ Traces   │
                     │  ├─ Scores   │   ┌───────────────────┐
                     │  ├─ Evals    │   │ Custom Eval Jobs  │
                     │  └─ UI       │   │ ├─ Python SDK     │
                     └──────┬───────┘   │ ├─ DeepEval       │
                            │           │ └─ Ragas           │
                            └───────────┘
```

**Setup for ADK + LangChain:**

```python
# langfuse_eval_setup.py

# --- ADK Agent ---
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Configure OTel to export to Langfuse
exporter = OTLPSpanExporter(
    endpoint="https://cloud.langfuse.com/api/public/otel/v1/traces",
    headers={"Authorization": "Bearer pk-..."},
)
GoogleADKInstrumentor().instrument()

# --- LangChain Agent ---
from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
)
# Pass langfuse_handler in callbacks=[...] to any LangChain invocation

# --- Evaluation ---
from langfuse import Langfuse
langfuse = Langfuse()

# Score traces
for trace in langfuse.get_traces(name="eval_run_001"):
    # Run custom evaluation
    score = custom_evaluate(trace)
    langfuse.score(trace_id=trace.id, name="custom_metric", value=score)
```

**Pros:**
- Zero evaluation infrastructure to build -- platform handles it
- Native support for both ADK and LangChain
- Rich UI for trace visualization and comparison
- Built-in LLM-as-judge evaluation
- Supports DeepEval and Ragas metrics as custom evaluators

**Cons:**
- External dependency (cloud or self-hosted)
- No offline evaluation without network
- Eval dataset format is platform-specific, not ADK's `.evalset.json`
- Less control over evaluation pipeline
- ADK Web UI compatibility lost (separate UI)

---

### Option D: Plugin Architecture for Framework Adapters

Design a plugin system where each framework provides an adapter implementing a standard interface.

**Architecture:**

```python
# framework_adapter.py -- Plugin interface

from abc import ABC, abstractmethod

class FrameworkAdapter(ABC):
    """Interface that each agent framework must implement."""

    @abstractmethod
    async def run_agent(self, user_input: str, session_state: dict) -> "AgentResult":
        """Execute the agent and return standardized result."""
        ...

    @abstractmethod
    def install_tool_mocks(self, mock_config: dict) -> None:
        """Configure tool mocking for this framework."""
        ...

    @abstractmethod
    def install_agent_mocks(self, mock_config: dict) -> None:
        """Configure sub-agent mocking for this framework."""
        ...

    @abstractmethod
    def get_otel_instrumentor(self) -> Optional[object]:
        """Return the OTel instrumentor for this framework."""
        ...

    @abstractmethod
    def extract_invocations(self, result: "AgentResult") -> list[UnifiedInvocation]:
        """Convert framework-specific result to UnifiedInvocation."""
        ...

@dataclass
class AgentResult:
    """Standardized agent execution result."""
    final_response: str
    events: list[dict]  # Framework-specific event stream
    duration_ms: float
    metadata: dict


# --- ADK Adapter ---

class ADKAdapter(FrameworkAdapter):
    def __init__(self, agent):
        self.agent = agent
        self.runner = InMemoryRunner(agent=agent)

    async def run_agent(self, user_input, session_state):
        session = await self.runner.session_service.create_session(...)
        events = []
        async for event in self.runner.run_async(
            user_id="eval", session_id=session.id,
            new_message=types.Content(parts=[types.Part(text=user_input)]),
        ):
            events.append(event.model_dump())
        return AgentResult(
            final_response=events[-1].get("content", {}).get("parts", [{}])[0].get("text", ""),
            events=events,
            duration_ms=0,
            metadata={},
        )

    def install_tool_mocks(self, mock_config):
        # Uses before_tool_callback (from Task 1)
        interceptor = MockInterceptor(mock_config)
        self.agent = interceptor.build_mocked_agent(self.agent)
        self.runner = InMemoryRunner(agent=self.agent)

    def install_agent_mocks(self, mock_config):
        interceptor = MockInterceptor({"subAgents": mock_config})
        self.agent = interceptor.build_mocked_agent(self.agent)
        self.runner = InMemoryRunner(agent=self.agent)

    def get_otel_instrumentor(self):
        return None  # ADK instruments itself

    def extract_invocations(self, result):
        return ADKTraceNormalizer().normalize_events(result.events)


# --- LangChain Adapter ---

class LangChainAdapter(FrameworkAdapter):
    def __init__(self, agent_executor):
        self.executor = agent_executor
        self.original_tools = list(agent_executor.tools)

    async def run_agent(self, user_input, session_state):
        result = await self.executor.ainvoke(
            {"input": user_input},
            config={"configurable": session_state},
        )
        return AgentResult(
            final_response=result.get("output", ""),
            events=result.get("intermediate_steps", []),
            duration_ms=0,
            metadata={},
        )

    def install_tool_mocks(self, mock_config):
        """Replace tools with mock versions."""
        from langchain_core.tools import StructuredTool

        mocked_tools = []
        for tool in self.original_tools:
            if tool.name in mock_config.get("tools", {}):
                rules = mock_config["tools"][tool.name]
                # Create a mock tool with same schema but canned responses
                def make_mock(tool_rules):
                    def mock_fn(**kwargs):
                        for rule in tool_rules:
                            match = rule.get("matchArgs", {})
                            if not match or all(kwargs.get(k) == v for k, v in match.items()):
                                return rule["response"]
                        return {"error": "no mock match"}
                    return mock_fn

                mocked_tools.append(StructuredTool(
                    name=tool.name,
                    description=tool.description,
                    func=make_mock(rules),
                    args_schema=tool.args_schema,
                ))
            else:
                mocked_tools.append(tool)

        self.executor.tools = mocked_tools

    def install_agent_mocks(self, mock_config):
        """For LangGraph: replace subgraph nodes with mock nodes."""
        # Implementation depends on graph structure
        pass

    def get_otel_instrumentor(self):
        from opentelemetry.instrumentation.langchain import LangChainInstrumentor
        return LangChainInstrumentor()

    def extract_invocations(self, result):
        invocation = UnifiedInvocation(
            final_response=result.final_response,
            framework=FrameworkType.LANGCHAIN,
        )
        for action, observation in result.events:
            invocation.tool_calls.append(UnifiedToolCall(
                tool_name=action.tool,
                arguments=action.tool_input,
                result=observation,
            ))
        return [invocation]


# --- LangGraph Adapter ---

class LangGraphAdapter(FrameworkAdapter):
    def __init__(self, compiled_graph):
        self.graph = compiled_graph

    async def run_agent(self, user_input, session_state):
        from langchain_core.messages import HumanMessage
        events = []
        async for event in self.graph.astream(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode="updates",
        ):
            events.append(event)

        # Extract final response from last agent message
        final = ""
        for event in reversed(events):
            for node_name, output in event.items():
                for msg in output.get("messages", []):
                    if hasattr(msg, "content") and msg.content:
                        final = msg.content
                        break

        return AgentResult(
            final_response=final,
            events=events,
            duration_ms=0,
            metadata={},
        )

    def install_tool_mocks(self, mock_config):
        """Inject mocks at the tools node level."""
        # LangGraph approach: use interrupt_before + state injection
        # Or replace tool node with mock node
        pass

    def install_agent_mocks(self, mock_config):
        """Replace subgraph nodes with mock nodes."""
        # LangGraph: replace node functions in the graph
        pass

    def get_otel_instrumentor(self):
        from opentelemetry.instrumentation.langchain import LangChainInstrumentor
        return LangChainInstrumentor()

    def extract_invocations(self, result):
        invocation = UnifiedInvocation(framework=FrameworkType.LANGGRAPH)
        for event in result.events:
            for node_name, output in event.items():
                if node_name == "tools":
                    for msg in output.get("messages", []):
                        if hasattr(msg, "name"):
                            invocation.tool_calls.append(UnifiedToolCall(
                                tool_name=msg.name or "tool",
                                arguments={},
                                result=msg.content,
                            ))
                elif node_name not in ("__start__", "__end__"):
                    invocation.agent_calls.append(UnifiedAgentCall(
                        agent_name=node_name,
                        output_text=str(output),
                    ))

        invocation.final_response = result.final_response
        return [invocation]
```

**Unified Eval Runner:**

```python
# unified_eval_runner.py

class UnifiedEvalRunner:
    """Framework-agnostic evaluation runner using adapters."""

    def __init__(self, adapter: FrameworkAdapter, output_dir: str):
        self.adapter = adapter
        self.output_dir = Path(output_dir)

    async def run_inference(self, eval_set_path: str, mock_config: dict = None):
        """Phase 1: Run inference with optional mocking."""
        with open(eval_set_path) as f:
            eval_data = json.load(f)

        if mock_config:
            self.adapter.install_tool_mocks(mock_config)
            self.adapter.install_agent_mocks(mock_config)

        # Setup OTel
        instrumentor = self.adapter.get_otel_instrumentor()
        if instrumentor:
            instrumentor.instrument()

        results = []
        for case in eval_data.get("evalCases", []):
            case_mock = case.get("mockResponses")
            if case_mock:
                self.adapter.install_tool_mocks(case_mock)
                self.adapter.install_agent_mocks(case_mock)

            for inv in case.get("conversation", []):
                user_text = inv["userContent"]["parts"][0]["text"]
                agent_result = await self.adapter.run_agent(
                    user_text, case.get("sessionInput", {}).get("state", {})
                )
                invocations = self.adapter.extract_invocations(agent_result)
                results.append({
                    "eval_id": case["evalId"],
                    "invocations": [asdict(inv) for inv in invocations],
                })

        # Save
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "inference_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    async def run_evaluation(self, eval_set_path: str, config: dict):
        """Phase 2: Evaluate saved results."""
        # Load inference results and apply evaluators
        # ... same as Option D in Task 2 ...
        pass
```

**Pros:**
- Clean plugin architecture -- new frameworks add an adapter, nothing else changes
- Each adapter uses native mocking mechanisms (ADK callbacks, LangChain middleware, LangGraph interrupts)
- Unified eval dataset format across all frameworks
- Testable: each adapter can be unit-tested independently

**Cons:**
- More abstraction layers = more complexity
- Adapter maintenance burden for each framework
- Some adapters (LangGraph) are hard to implement fully due to graph complexity
- Tool mocking approaches differ significantly per framework

---

## Tool and Sub-Agent Mocking Across Frameworks

### Mocking Comparison

| Mechanism | ADK | LangChain | LangGraph |
|-----------|-----|-----------|-----------|
| **Tool mocking** | `before_tool_callback` | `wrap_tool_call` middleware, or replace tool function | `interrupt_before` tools node + state injection |
| **Sub-agent mocking** | Replace in `sub_agents` list (Task 1 Option A) | Replace chain in sequence | Replace node function in graph |
| **Deterministic mock** | Stub agent with fixed instruction | `GenericFakeChatModel` | Mock node function |
| **JSON-driven** | Via `MockInterceptor` (Task 1) | Via mock tool factory | Via mock node factory |

### LangGraph-Specific Mocking Pattern

```python
# langgraph_mocking.py -- Mock tools and sub-agents in LangGraph

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolMessage

def create_mocked_graph(original_graph, mock_config):
    """Clone a LangGraph with mocked tool and agent nodes."""

    tool_mocks = mock_config.get("tools", {})
    agent_mocks = mock_config.get("subAgents", {})

    # For tool mocking: use interrupt_before pattern
    # This pauses execution before the tools node
    # We then inject mock responses into state

    class MockToolNode:
        def __init__(self, mocks):
            self.mocks = mocks

        def __call__(self, state):
            messages = state.get("messages", [])
            last_msg = messages[-1] if messages else None

            if last_msg and hasattr(last_msg, "tool_calls"):
                mock_responses = []
                for tc in last_msg.tool_calls:
                    rules = self.mocks.get(tc["name"], [])
                    response = None
                    for rule in rules:
                        match = rule.get("matchArgs", {})
                        if not match or all(tc["args"].get(k) == v for k, v in match.items()):
                            response = rule["response"]
                            break

                    if response is not None:
                        mock_responses.append(ToolMessage(
                            content=json.dumps(response),
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        ))
                    else:
                        # Fall through to real tool -- not mocked
                        # Would need to call the real tool here
                        mock_responses.append(ToolMessage(
                            content="Tool not mocked and real execution skipped",
                            tool_call_id=tc["id"],
                        ))

                return {"messages": mock_responses}

            return state

    # For agent mocking: replace subgraph nodes
    class MockAgentNode:
        def __init__(self, agent_name, mock_def):
            self.agent_name = agent_name
            self.default_response = mock_def.get("defaultResponse", "OK")

        def __call__(self, state):
            return {
                "messages": [AIMessage(content=self.default_response)]
            }

    # Rebuild graph with mocked nodes
    # NOTE: This requires access to the graph's node definitions,
    # which may not be publicly exposed in compiled graphs.
    # A workaround is to define the graph with replaceable node functions.

    return original_graph  # Placeholder; real impl depends on graph structure
```

---

## ADK Web UI Compatibility and Extension

### Challenge

The ADK Web UI (`adk web`) is designed around ADK-specific data models. To display LangChain/LangGraph evaluations, we need one of:

1. **Convert non-ADK results to ADK format** (for display in existing UI)
2. **Extend the UI** (add components for non-ADK agents)
3. **Use external trace viewers** (link out from ADK UI)

### Approach 1: Convert to ADK Format for Display

```python
# adk_format_converter.py

def unified_to_adk_invocation(unified: UnifiedInvocation) -> dict:
    """Convert UnifiedInvocation to ADK Invocation format for Web UI."""
    from google.adk.evaluation.eval_case import Invocation, IntermediateData
    from google.genai import types

    tool_uses = [
        types.FunctionCall(name=tc.tool_name, args=tc.arguments)
        for tc in unified.tool_calls
    ]
    tool_responses = [
        types.FunctionResponse(name=tc.tool_name, response=tc.result or {})
        for tc in unified.tool_calls
    ]
    intermediate_responses = [
        (ac.agent_name, [types.Part(text=ac.output_text or "")])
        for ac in unified.agent_calls
    ]

    return Invocation(
        invocation_id=unified.invocation_id,
        user_content=types.Content(parts=[types.Part(text=unified.user_input)]),
        final_response=types.Content(parts=[types.Part(text=unified.final_response)]),
        intermediate_data=IntermediateData(
            tool_uses=tool_uses,
            tool_responses=tool_responses,
            intermediate_responses=intermediate_responses,
        ),
    )
```

### Approach 2: Extend ADK Web Backend

```python
# Extended adk_web_server.py with cross-framework support

@app.post("/apps/{app_name}/eval_cross_framework")
async def run_cross_framework_eval(
    app_name: str,
    request: CrossFrameworkEvalRequest,
):
    """Run evaluation on traces from any supported framework."""
    # Load traces
    normalizer = AutoNormalizer()
    with open(request.traces_path) as f:
        spans = [json.loads(line) for line in f if line.strip()]

    actual_invocations = normalizer.normalize(spans)

    # Load eval set
    with open(request.eval_set_path) as f:
        eval_set = json.load(f)

    # Run evaluators
    results = {}
    for metric_name in request.metrics:
        evaluator = get_universal_evaluator(metric_name)
        result = evaluator.evaluate(actual_invocations, expected_invocations)
        results[metric_name] = result

    # Convert to ADK format for UI display
    adk_invocations = [unified_to_adk_invocation(inv) for inv in actual_invocations]

    # Save as ADK eval result (visible in existing UI)
    eval_result = format_as_adk_eval_result(adk_invocations, results)
    results_manager.save_eval_set_result(app_name, eval_set_id, eval_result)

    return {"status": "ok", "results": results}
```

### Approach 3: Link to External Trace Viewers

For rich trace visualization of LangChain/LangGraph agents, link to external tools:

```typescript
// Angular component in adk-web
@Component({
  selector: 'cross-framework-trace',
  template: `
    <div class="trace-links">
      <h3>External Trace Viewers</h3>
      <div *ngIf="framework === 'langchain'">
        <a [href]="langsmithUrl" target="_blank">View in LangSmith</a>
      </div>
      <div>
        <a [href]="jaegerUrl" target="_blank">View in Jaeger</a>
      </div>
      <div>
        <a [href]="langfuseUrl" target="_blank">View in Langfuse</a>
      </div>
    </div>

    <!-- Embedded trace viewer for OTLP JSON -->
    <trace-waterfall [spans]="spans"></trace-waterfall>
  `
})
export class CrossFrameworkTraceComponent {
  @Input() traceId: string;
  @Input() framework: 'adk' | 'langchain' | 'langgraph';

  get jaegerUrl() {
    return `http://localhost:16686/trace/${this.traceId}`;
  }
}
```

---

## Comparison Matrix

| Criterion | Option A (Trace-Based) | Option B (ADK Wrapper) | Option C (External Platform) | Option D (Plugin) |
|-----------|----------------------|----------------------|---------------------------|------------------|
| Framework coverage | Any OTel-emitting | LangChain, LangGraph | Platform-dependent | Any with adapter |
| ADK compatibility | Via format conversion | Native | Lost | Via format conversion |
| Tool mocking | Framework-native + trace replay | ADK `before_tool_callback` | Platform-dependent | Per-adapter |
| Sub-agent mocking | Framework-native | Via wrapper events | Limited | Per-adapter |
| Web UI integration | Convert or link externally | Native ADK UI | External UI | Convert + extend |
| Setup complexity | Medium | Low-Medium | Low | High |
| Maintenance burden | Low (OTel standard) | Medium (wrapper updates) | Low (vendor-managed) | High (per-adapter) |
| Evaluation accuracy | Good (trace-dependent) | Good (event conversion) | Good | Best (native extraction) |

---

## Recommendation

**Combine Options A and D**:

1. **Option A (Trace-Based)** as the primary evaluation path: Export OTel traces from any framework, normalize them via `AutoNormalizer`, and evaluate with universal evaluators. This is the most future-proof approach since it relies on the converging OTel GenAI semantic conventions.

2. **Option D (Plugin Adapters)** for mocking: Each framework needs its own mocking mechanism (ADK's `before_tool_callback`, LangChain's middleware, LangGraph's `interrupt_before`). The plugin adapter pattern encapsulates this per-framework complexity.

3. **Option B (ADK Wrapper)** as a quick-start path: For teams that want immediate ADK Web UI compatibility, wrapping LangChain/LangGraph agents as ADK `CustomAgent` is the fastest path. It can coexist with the trace-based approach.

4. **ADK Web UI**: Use format conversion (Approach 1) to display non-ADK evaluations in the existing UI. For rich trace debugging, link to Jaeger or Langfuse (Approach 3).

### End-to-End Flow

```
1. Define eval cases in .evalset.json (with mockResponses)
2. Select framework adapter (ADK / LangChain / LangGraph)
3. Adapter installs mocks using framework-native mechanisms
4. Run inference → OTel traces exported to file + structured results saved
5. Load saved data → normalize traces → run evaluators
6. Push results to ADK eval results manager → view in adk web
7. Optionally view traces in Jaeger/Langfuse for debugging
```
