# Task 2: Separating Inference and Evaluation via OpenTelemetry Trace Collection

## Problem Statement

ADK's current `adk eval` command runs inference (agent execution) and metric evaluation as a single coupled step. This report analyzes options to:

1. Run agent inference and collect all trace data to a local file via OpenTelemetry
2. Run evaluations separately on saved trace data with support for custom eval metrics
3. Maintain compatibility with ADK Web UI for trace and eval observation

---

## Current ADK Architecture

### Coupled Inference + Evaluation

```
adk eval agent_module eval_set.json --config test_config.json
       │
       ├── EvaluationGenerator.generate()
       │       ├── Create InMemoryRunner(agent)
       │       ├── For each EvalCase:
       │       │     └── runner.run_async(user_msg) → collect Invocations
       │       └── Return actual_invocations[]
       │
       └── Evaluators.evaluate(actual_invocations, expected_invocations)
               ├── TrajectoryEvaluator
               ├── ResponseMatchEvaluator
               ├── LLM-judged evaluators
               └── Return EvaluationResult
```

### LocalEvalService -- Already Partially Separated

The `LocalEvalService` class has separate methods:
- `perform_inference(eval_set, agent, eval_config)` -- runs agent, returns `InferenceResult`
- `evaluate(eval_set, eval_config, inference_results)` -- scores inference results

However, inference results are kept **in-memory only** -- no persistence between stages.

### ADK's Built-in OpenTelemetry Tracing

ADK emits spans automatically:
- `invocation` -- top-level span per agent invocation
- `agent_run` -- per agent's `run_async`
- `call_llm` -- LLM call spans with prompts/completions
- `execute_tool` -- tool call spans with args/results

These are real OTLP spans that can be exported to any backend.

---

## Options Analysis

### Option A: OTel File Exporter + Trace-Based Eval Runner (Recommended)

Export ADK traces to local JSONL files during inference, then load and evaluate them offline.

**Architecture:**

```
Phase 1: INFERENCE                     Phase 2: EVALUATION
┌──────────────────────┐               ┌──────────────────────┐
│  Agent Execution     │               │  Trace Loader        │
│  ├─ ADK Runner       │               │  ├─ Read .jsonl      │
│  ├─ OTel SDK         │               │  ├─ Parse spans      │
│  │  ├─ BatchSpanProc │──export──►    │  ├─ Reconstruct      │
│  │  └─ FileExporter  │  .jsonl       │  │  Invocations      │
│  └─ InferenceResult  │──export──►    │  └─ Merge w/ eval    │
│     serializer       │  .json        │     case data        │
└──────────────────────┘               │                      │
                                       │  Evaluators          │
                                       │  ├─ Built-in metrics │
                                       │  ├─ Custom metrics   │
                                       │  └─ EvaluationResult │
                                       └──────────────────────┘
```

**Phase 1 -- Inference with Trace Export:**

```python
# inference_runner.py -- Run agent and export traces to local files

import json
from pathlib import Path
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

class InferenceRunner:
    """Runs agent inference and persists both OTel traces and structured results."""

    def __init__(self, output_dir: str, otel_endpoint: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_otel(otel_endpoint)

    def _setup_otel(self, endpoint: str = None):
        """Configure OTel with file + optional remote export."""
        provider = TracerProvider()

        # Always export to local JSONL file
        from otel_file_exporter import FileSpanExporter  # or custom impl
        file_exporter = FileSpanExporter(
            file_path=str(self.output_dir / "traces.jsonl")
        )
        provider.add_span_processor(BatchSpanProcessor(file_exporter))

        # Optionally also export to remote (Jaeger, Cloud Trace, etc.)
        if endpoint:
            remote_exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(remote_exporter))

        trace.set_tracer_provider(provider)

    async def run_inference(self, agent, eval_set_path: str) -> str:
        """Run inference for all eval cases and save results."""
        from google.adk.evaluation import EvalSet
        from google.adk.runners import InMemoryRunner

        with open(eval_set_path) as f:
            eval_set = EvalSet.model_validate_json(f.read())

        results = []
        runner = InMemoryRunner(agent=agent, app_name=eval_set.eval_set_id)

        for case in eval_set.eval_cases:
            case_result = {
                "eval_id": case.eval_id,
                "invocations": [],
                "session_input": case.session_input.model_dump() if case.session_input else None,
                "timestamp": __import__("time").time(),
            }

            session = await runner.session_service.create_session(
                app_name=eval_set.eval_set_id,
                user_id=case.session_input.user_id if case.session_input else "eval_user",
                state=case.session_input.state if case.session_input else {},
            )

            for inv in (case.conversation or []):
                actual_events = []
                async for event in runner.run_async(
                    user_id=session.user_id,
                    session_id=session.id,
                    new_message=inv.user_content,
                ):
                    actual_events.append(event.model_dump())

                case_result["invocations"].append({
                    "invocation_id": inv.invocation_id,
                    "user_content": inv.user_content.model_dump(),
                    "actual_events": actual_events,
                })

            results.append(case_result)

        # Save structured inference results
        results_path = self.output_dir / "inference_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Flush OTel spans
        trace.get_tracer_provider().force_flush()

        return str(results_path)
```

**Phase 2 -- Offline Evaluation:**

```python
# offline_evaluator.py -- Load saved traces and run evaluations

import json
from pathlib import Path
from typing import Optional

class OfflineEvaluator:
    """Evaluates saved inference results against eval cases with configurable metrics."""

    def __init__(self, inference_dir: str):
        self.inference_dir = Path(inference_dir)
        self.results = self._load_inference_results()
        self.traces = self._load_traces()

    def _load_inference_results(self) -> list:
        path = self.inference_dir / "inference_results.json"
        with open(path) as f:
            return json.load(f)

    def _load_traces(self) -> list:
        """Load OTel trace spans from JSONL file."""
        path = self.inference_dir / "traces.jsonl"
        spans = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        spans.append(json.loads(line))
        return spans

    def reconstruct_invocations(self, case_result: dict) -> list:
        """Reconstruct ADK Invocation objects from saved events."""
        from google.adk.evaluation.eval_case import Invocation, IntermediateData
        from google.genai import types

        invocations = []
        for inv_data in case_result["invocations"]:
            tool_uses, tool_responses, intermediate_responses = [], [], []
            final_response = None

            for event in inv_data["actual_events"]:
                # Extract tool calls from function_call events
                if "function_call" in event.get("actions", {}).get("parts", [{}])[0]:
                    fc = event["actions"]["parts"][0]["function_call"]
                    tool_uses.append(types.FunctionCall(
                        name=fc["name"], args=fc.get("args", {})
                    ))
                # Extract tool responses
                if "function_response" in event.get("actions", {}).get("parts", [{}])[0]:
                    fr = event["actions"]["parts"][0]["function_response"]
                    tool_responses.append(types.FunctionResponse(
                        name=fr["name"], response=fr.get("response", {})
                    ))
                # Extract final response
                if event.get("is_final_response"):
                    parts = event.get("content", {}).get("parts", [])
                    final_response = types.Content(parts=[
                        types.Part(text=p.get("text", "")) for p in parts
                    ])

            invocations.append(Invocation(
                invocation_id=inv_data.get("invocation_id", ""),
                user_content=types.Content(**inv_data["user_content"]),
                final_response=final_response,
                intermediate_data=IntermediateData(
                    tool_uses=tool_uses,
                    tool_responses=tool_responses,
                    intermediate_responses=intermediate_responses,
                ),
            ))
        return invocations

    async def evaluate(
        self,
        eval_set_path: str,
        config_path: Optional[str] = None,
        custom_metrics: Optional[dict] = None,
    ) -> dict:
        """Run evaluation metrics on saved inference results."""
        from google.adk.evaluation import EvalSet, EvalConfig
        from google.adk.evaluation.evaluator import MetricEvaluatorRegistry

        with open(eval_set_path) as f:
            eval_set = EvalSet.model_validate_json(f.read())

        eval_config = None
        if config_path:
            with open(config_path) as f:
                eval_config = EvalConfig.model_validate_json(f.read())

        # Register custom metrics if provided
        if custom_metrics:
            for name, evaluator_cls in custom_metrics.items():
                MetricEvaluatorRegistry.register(name, evaluator_cls)

        all_results = {}
        for case in eval_set.eval_cases:
            # Find matching inference result
            case_result = next(
                (r for r in self.results if r["eval_id"] == case.eval_id), None
            )
            if not case_result:
                all_results[case.eval_id] = {"status": "SKIPPED", "reason": "No inference data"}
                continue

            actual_invocations = self.reconstruct_invocations(case_result)
            expected_invocations = case.conversation or []

            case_evals = {}
            for metric_name, metric in (eval_config.criteria or {}).items():
                evaluator = MetricEvaluatorRegistry.get_evaluator(metric_name)
                result = await evaluator.evaluate_invocations(
                    actual_invocations=actual_invocations,
                    expected_invocations=expected_invocations,
                )
                case_evals[metric_name] = {
                    "score": result.overall_score,
                    "status": result.overall_eval_status.value,
                    "per_invocation": [
                        {"score": r.score, "status": r.eval_status.value}
                        for r in result.per_invocation_results
                    ],
                }
            all_results[case.eval_id] = case_evals

        return all_results
```

**CLI Extension:**

```bash
# Phase 1: Run inference only, save traces
adk-eval-ext infer \
    ./my_agent \
    ./my_agent/eval_set.evalset.json \
    --output-dir ./eval_runs/run_001 \
    --otel-endpoint localhost:4317  # optional: also send to Jaeger

# Phase 2: Evaluate saved traces
adk-eval-ext evaluate \
    ./eval_runs/run_001 \
    ./my_agent/eval_set.evalset.json \
    --config ./test_config.json \
    --custom-metrics ./custom_metrics.py \
    --print-detailed-results

# Phase 2 (re-run with different metrics, no re-inference):
adk-eval-ext evaluate \
    ./eval_runs/run_001 \
    ./my_agent/eval_set.evalset.json \
    --config ./test_config_v2.json
```

**Pros:**
- Full separation: inference runs once, evaluation runs many times with different configs
- OTel traces provide rich debugging data beyond structured results
- Dual export (file + remote) enables both offline eval and live monitoring
- Custom metrics can be iterated without re-running expensive inference
- Traces are standard OTLP -- viewable in any compatible tool

**Cons:**
- Reconstruction of Invocations from events requires careful parsing
- Two-file output (traces.jsonl + inference_results.json) -- could be unified
- Requires new CLI commands or wrapper script

---

### Option B: OTel Collector Sidecar with File Exporter

Use the OpenTelemetry Collector as a sidecar process that captures all traces to disk with rotation and batching.

**Architecture:**

```
┌────────────────┐     OTLP      ┌──────────────────────┐
│  ADK Agent     │───gRPC:4317──►│  OTel Collector      │
│  (inference)   │               │  ├─ otlpreceiver     │
│                │               │  ├─ batch processor   │
└────────────────┘               │  └─ file exporter     │──► traces/
                                 │     └─ json format    │    ├─ traces_001.json
                                 │     └─ rotation: 10MB │    ├─ traces_002.json
                                 │                       │    └─ ...
                                 │  └─ otlp exporter     │──► Jaeger (optional)
                                 └──────────────────────┘
```

**Collector Configuration (`otel-collector-config.yaml`):**

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: "0.0.0.0:4317"
      http:
        endpoint: "0.0.0.0:4318"

processors:
  batch:
    timeout: 5s
    send_batch_size: 100

  filter:
    traces:
      include:
        match_type: regexp
        span_names:
          - "invocation.*"
          - "agent_run.*"
          - "call_llm.*"
          - "execute_tool.*"

exporters:
  file:
    path: "./eval_traces/traces.jsonl"
    format: json
    rotation:
      max_megabytes: 50
      max_days: 7
      max_backups: 10

  # Optional: fan-out to Jaeger for live viewing
  otlp/jaeger:
    endpoint: "localhost:14250"
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, filter]
      exporters: [file, otlp/jaeger]
```

**Inference Script:**

```python
# Uses standard ADK with OTEL_EXPORTER_OTLP_ENDPOINT pointing to Collector

import os
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
os.environ["OTEL_SERVICE_NAME"] = "adk-eval-inference"

# Standard ADK agent execution
from google.adk.runners import InMemoryRunner
# ... run agent, traces automatically sent to Collector ...
```

**Evaluation Script:**

```python
# eval_from_traces.py -- Parse Collector output and evaluate

class CollectorTraceEvaluator:
    def __init__(self, traces_dir: str):
        self.traces_dir = Path(traces_dir)

    def load_all_traces(self) -> list:
        """Load all trace files from Collector output."""
        all_spans = []
        for trace_file in sorted(self.traces_dir.glob("traces*.jsonl")):
            with open(trace_file) as f:
                for line in f:
                    data = json.loads(line)
                    # Collector writes resourceSpans format
                    for rs in data.get("resourceSpans", []):
                        for ss in rs.get("scopeSpans", []):
                            all_spans.extend(ss.get("spans", []))
        return all_spans

    def group_by_trace_id(self, spans: list) -> dict:
        """Group spans by trace ID to reconstruct conversations."""
        traces = {}
        for span in spans:
            tid = span.get("traceId", "unknown")
            traces.setdefault(tid, []).append(span)
        return traces

    def spans_to_invocations(self, trace_spans: list) -> list:
        """Convert OTel spans to ADK Invocation objects."""
        # Sort by start time
        trace_spans.sort(key=lambda s: s.get("startTimeUnixNano", 0))

        invocations = []
        current_tool_uses = []
        current_tool_responses = []

        for span in trace_spans:
            name = span.get("name", "")
            attrs = {a["key"]: a["value"] for a in span.get("attributes", [])}

            if name.startswith("execute_tool"):
                tool_name = attrs.get("tool.name", {}).get("stringValue", name)
                tool_args = json.loads(
                    attrs.get("tool.args", {}).get("stringValue", "{}")
                )
                tool_result = json.loads(
                    attrs.get("tool.result", {}).get("stringValue", "{}")
                )
                current_tool_uses.append({"name": tool_name, "args": tool_args})
                current_tool_responses.append({"name": tool_name, "response": tool_result})

            elif name == "invocation":
                # Build invocation from accumulated data
                # ... reconstruct Invocation object ...
                pass

        return invocations
```

**Pros:**
- Enterprise-grade: rotation, batching, filtering, multi-destination fan-out
- Collector handles backpressure and reliability
- Traces viewable in real-time via Jaeger while also persisted to disk
- Standard OTLP format -- maximum interoperability
- Filter out noise (only keep agent-relevant spans)

**Cons:**
- Requires running a separate Collector process
- Heavier operational footprint than in-process file export
- Collector output format (resourceSpans) is verbose and harder to parse than ADK events
- Reconstructing ADK Invocations from raw OTel spans is complex (lossy mapping)

---

### Option C: ADK InferenceResult Serialization (Minimal Approach)

Serialize ADK's native `InferenceResult` objects to JSON after inference, then deserialize for evaluation.

**Architecture:**

```
Phase 1: INFERENCE                     Phase 2: EVALUATION
┌──────────────────────┐               ┌──────────────────────┐
│  LocalEvalService    │               │  LocalEvalService    │
│  .perform_inference()│               │  .evaluate()         │
│        │             │               │        ▲             │
│        ▼             │               │        │             │
│  InferenceResult[]   │──serialize──► │  InferenceResult[]   │
│  (Pydantic models)   │   .json       │  (deserialized)      │
└──────────────────────┘               └──────────────────────┘
```

**Prototype:**

```python
# serialize_inference.py

from google.adk.evaluation.local_eval_service import LocalEvalService
import json

class PersistableEvalService(LocalEvalService):
    """Extends LocalEvalService with inference result persistence."""

    async def run_inference_and_save(
        self, eval_set, agent, eval_config, output_path: str
    ):
        """Run inference and serialize results to JSON."""
        inference_results = await self.perform_inference(
            eval_set=eval_set, agent=agent, eval_config=eval_config
        )

        # Serialize using Pydantic
        serialized = []
        for result in inference_results:
            serialized.append(result.model_dump(mode="json"))

        with open(output_path, "w") as f:
            json.dump(serialized, f, indent=2)

        return output_path

    async def evaluate_from_saved(
        self, eval_set, eval_config, inference_path: str
    ):
        """Load saved inference results and evaluate."""
        with open(inference_path) as f:
            data = json.load(f)

        from google.adk.evaluation.local_eval_service import InferenceResult
        inference_results = [InferenceResult.model_validate(d) for d in data]

        return await self.evaluate(
            eval_set=eval_set,
            eval_config=eval_config,
            inference_results=inference_results,
        )
```

**Pros:**
- Minimal code -- extends existing ADK service
- Uses Pydantic serialization -- lossless round-trip
- No dependency on OTel for the separation itself
- Easiest to integrate with existing `adk eval` workflow

**Cons:**
- No OTel trace data in the output (just structured results)
- Loses detailed LLM request/response metadata
- Does not support trace-level debugging or viewing in Jaeger/Grafana
- Only works for ADK agents (no cross-framework benefit)

---

### Option D: Hybrid -- InferenceResult + OTel Traces

Combine Options A and C: serialize ADK's native InferenceResult for evaluation, and simultaneously export OTel traces for debugging.

**Architecture:**

```
Phase 1: INFERENCE
┌──────────────────────────────────────┐
│  HybridInferenceRunner               │
│  ├─ LocalEvalService.perform_inference()
│  │     └─ InferenceResult[] ──► inference_results.json (for eval)
│  └─ OTel TracerProvider
│        ├─ FileExporter ──► traces.jsonl (for debugging)
│        └─ OTLPExporter ──► Jaeger (for live viewing)
└──────────────────────────────────────┘

Phase 2: EVALUATION
┌──────────────────────────────────────┐
│  OfflineEvaluator                     │
│  ├─ Load inference_results.json       │
│  ├─ Deserialize InferenceResult[]     │
│  ├─ Run metrics (built-in + custom)   │
│  └─ Optionally enrich with trace data │
│     for custom metrics that need      │
│     latency, token counts, etc.       │
└──────────────────────────────────────┘
```

**Custom Metrics Example (using trace data):**

```python
# custom_metrics.py -- Custom evaluators that use OTel trace data

from google.adk.evaluation.evaluator import Evaluator, EvaluationResult, EvalStatus
import json

class LatencyEvaluator(Evaluator):
    """Evaluates agent response latency from OTel trace data."""

    def __init__(self, traces_path: str, max_latency_ms: float = 5000):
        self.max_latency_ms = max_latency_ms
        self.traces = self._load_traces(traces_path)

    def _load_traces(self, path):
        spans = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    spans.append(json.loads(line))
        return spans

    async def evaluate_invocations(self, actual, expected):
        # Calculate latency from trace spans
        invocation_spans = [s for s in self.traces if s.get("name") == "invocation"]
        latencies = []
        for span in invocation_spans:
            duration_ns = span["endTimeUnixNano"] - span["startTimeUnixNano"]
            latencies.append(duration_ns / 1e6)  # Convert to ms

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        score = 1.0 if avg_latency <= self.max_latency_ms else 0.0

        return EvaluationResult(
            overall_score=score,
            overall_eval_status=EvalStatus.PASSED if score >= 0.5 else EvalStatus.FAILED,
        )


class TokenEfficiencyEvaluator(Evaluator):
    """Evaluates token usage efficiency from OTel trace data."""

    def __init__(self, traces_path: str, max_tokens_per_turn: int = 4000):
        self.max_tokens = max_tokens_per_turn
        self.traces = self._load_traces(traces_path)

    def _load_traces(self, path):
        spans = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    spans.append(json.loads(line))
        return spans

    async def evaluate_invocations(self, actual, expected):
        llm_spans = [s for s in self.traces if s.get("name") == "call_llm"]
        total_tokens = 0
        for span in llm_spans:
            attrs = {a["key"]: a["value"] for a in span.get("attributes", [])}
            input_tokens = attrs.get("gen_ai.usage.input_tokens", {}).get("intValue", 0)
            output_tokens = attrs.get("gen_ai.usage.output_tokens", {}).get("intValue", 0)
            total_tokens += input_tokens + output_tokens

        avg_per_turn = total_tokens / len(actual) if actual else 0
        score = min(1.0, self.max_tokens / max(avg_per_turn, 1))

        return EvaluationResult(
            overall_score=score,
            overall_eval_status=EvalStatus.PASSED if score >= 0.5 else EvalStatus.FAILED,
        )
```

**Test Config with Custom Metrics:**

```json
{
  "criteria": {
    "tool_trajectory_avg_score": {"threshold": 1.0},
    "final_response_match_v2": {"threshold": 0.8},
    "response_latency": {
      "threshold": 0.8,
      "custom_metric": {
        "module": "custom_metrics",
        "class": "LatencyEvaluator",
        "params": {"traces_path": "./eval_runs/run_001/traces.jsonl", "max_latency_ms": 3000}
      }
    },
    "token_efficiency": {
      "threshold": 0.7,
      "custom_metric": {
        "module": "custom_metrics",
        "class": "TokenEfficiencyEvaluator",
        "params": {"traces_path": "./eval_runs/run_001/traces.jsonl", "max_tokens_per_turn": 4000}
      }
    }
  }
}
```

**Pros:**
- Best of both worlds: structured results for standard eval, traces for advanced metrics
- Custom metrics can access latency, token counts, span hierarchy -- impossible with results alone
- Lossless evaluation via Pydantic serialization
- OTel traces enable debugging failed evals

**Cons:**
- Two output files to manage
- Slightly more setup than Option C alone

---

## Trace Management Tools Comparison

| Tool | Best For | Setup Effort | AI Agent Support | File Export |
|------|----------|-------------|------------------|------------|
| **OTel Collector + File Exporter** | Production, CI/CD | Medium | Via filtering | Native (JSON/proto) |
| **Jaeger** | Rich querying, dependency graphs | Low (Docker) | Generic spans | Via Collector |
| **otel-desktop-viewer** | Local dev, quick inspection | Minimal (single binary) | Generic spans | N/A (viewer only) |
| **Arize Phoenix** | AI-native observability | Low (pip install) | ADK, LangChain, LlamaIndex | Export to Parquet |
| **Langfuse** | LLM evaluation platform | Medium (self-host or cloud) | ADK, LangChain | API export |
| **otel-tui** | Terminal-based debugging | Minimal | Generic spans | N/A (viewer only) |

**Recommendation for this task:** Use **OTel Collector with File Exporter** for CI/CD and batch evaluation. Use **Jaeger** or **otel-desktop-viewer** for interactive debugging during development. Consider **Arize Phoenix** if AI-specific trace visualization (LLM calls, tool calls) is important.

---

## ADK Web UI Compatibility

### Displaying Saved Inference Results

The ADK Web UI already has endpoints for eval results. To display offline evaluation results:

**Option 1 -- Save results via existing API:**

```python
# After offline evaluation, push results to ADK's eval results manager
from google.adk.evaluation import LocalEvalSetResultsManager

results_manager = LocalEvalSetResultsManager(agents_dir="./my_agent")
results_manager.save_eval_set_result(
    app_name="my_agent",
    eval_set_id="my_eval_set",
    eval_case_results=eval_results,
)
# Results now visible in `adk web` Eval tab
```

**Option 2 -- Add trace viewer endpoint:**

```python
# Extend adk_web_server.py with trace file viewer

@app.get("/apps/{app_name}/traces/{run_id}")
async def get_traces(app_name: str, run_id: str):
    """Serve saved OTel traces for a specific inference run."""
    traces_path = Path(f"./eval_runs/{run_id}/traces.jsonl")
    spans = []
    with open(traces_path) as f:
        for line in f:
            if line.strip():
                spans.append(json.loads(line))
    return {"spans": spans, "run_id": run_id}

@app.get("/apps/{app_name}/inference_runs")
async def list_inference_runs(app_name: str):
    """List available inference runs."""
    runs_dir = Path("./eval_runs")
    runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if (run_dir / "inference_results.json").exists():
            runs.append({
                "run_id": run_dir.name,
                "has_traces": (run_dir / "traces.jsonl").exists(),
                "timestamp": run_dir.stat().st_mtime,
            })
    return runs
```

**Option 3 -- Embed Jaeger or otel-desktop-viewer:**

Add an iframe or link in the ADK web UI that opens the external trace viewer:

```typescript
// In adk-web Angular component
@Component({
  template: `
    <div class="trace-viewer">
      <iframe *ngIf="traceUrl" [src]="traceUrl" width="100%" height="600px"></iframe>
      <a *ngIf="jaegerUrl" [href]="jaegerUrl" target="_blank">Open in Jaeger</a>
    </div>
  `
})
export class TraceViewerComponent {
  traceUrl: string;
  jaegerUrl: string;

  openTrace(traceId: string) {
    this.jaegerUrl = `http://localhost:16686/trace/${traceId}`;
  }
}
```

---

## Recommendation

**Option D (Hybrid)** is the strongest approach:

1. **For inference**: Use `PersistableEvalService` to serialize `InferenceResult` (lossless, fast evaluation) + OTel `FileExporter` for traces (debugging, advanced metrics)
2. **For evaluation**: Load `InferenceResult` for standard metrics, enrich with trace data for custom metrics (latency, token efficiency, cost)
3. **For trace management**: OTel Collector in CI/CD; direct `FileExporter` in local dev; Jaeger for interactive debugging
4. **For ADK Web UI**: Push evaluation results via `LocalEvalSetResultsManager`; link to Jaeger for trace details

This gives the maximum flexibility: re-run evaluations with different metrics without re-inference, debug failures with rich trace data, and maintain full compatibility with ADK's existing tooling.
