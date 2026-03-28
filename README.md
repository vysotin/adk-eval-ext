# ADK Eval Tool

Agentic evaluation and testing toolkit for [Google ADK](https://github.com/google/adk-python) agents. Parses agent structure, generates test intents/scenarios via LLM, produces ADK-compatible `.evalset.json` golden datasets, and runs evaluations with trace collection.

## Features

- **Agent Parser** — Introspect any live ADK Agent object (including sub-agents, MCP tools) into a structured metadata tree
- **Intent Generator** — ADK-based agent that analyzes agent capabilities and generates comprehensive test intents with scenarios (happy paths, edge cases, errors)
- **Test Case Generator** — ADK-based agent that produces `.evalset.json` files in ADK's native format, supporting trajectory-based and conversation_scenario eval types
- **Eval Runner** — Execute evaluations with OpenTelemetry trace collection, structured result capture, and basic metrics (token counts, call counts, latency)
- **Streamlit UI** — 7-page app for managing the full workflow: metadata viewing, intent editing, test case editing (with add/delete), eval config (all 9 ADK metrics), eval launching, result exploration with trace trees, and dataset versioning

## Prerequisites

- Python 3.10+
- Google API key for Gemini (set in `.env`)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd adk-eval-ext

# Install dependencies
pip install -e ".[dev]"

# Set up your API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

## Quick Start

### Run an example

```bash
# Single agent with 2 tools
python examples/weather_agent/run_pipeline.py

# Multi-agent with sub-agents
python examples/travel_multi_agent/run_pipeline.py
```

### Launch UI with a pre-parsed agent (recommended)

```bash
# Parse an agent and launch Streamlit with metadata pre-loaded
python -m adk_eval_tool examples.weather_agent.agent root_agent

# Multi-agent example
python -m adk_eval_tool examples.travel_multi_agent.agent root_agent

# If installed via pip install -e .
adk-eval-tool examples.weather_agent.agent root_agent

# Custom port
python -m adk_eval_tool examples.weather_agent.agent root_agent --port 8502
```

The CLI:
1. Imports the agent module and parses the agent object
2. On success: launches Streamlit UI with metadata already loaded, agent module pre-filled in Eval Config
3. On failure: prints the error to stderr and exits with code 1

### Use the Python API

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from adk_eval_tool.agent_parser import parse_agent
from adk_eval_tool.intent_generator import generate_intents
from adk_eval_tool.testcase_generator.agent import generate_all_test_cases

# 1. Define your agent
def my_tool(query: str) -> str:
    """Search for information."""
    return f"Results for {query}"

agent = Agent(
    name="my_agent",
    model="gemini-2.0-flash",
    instruction="You help users search.",
    tools=[my_tool],
)

# 2. Parse metadata
metadata = parse_agent(agent, save_path="output/metadata.json")

# 3. Generate intents and scenarios
intent_set = asyncio.run(generate_intents(
    metadata=metadata,
    save_path="output/intents.json",
))

# 4. Generate eval test cases
eval_sets = asyncio.run(generate_all_test_cases(
    metadata=metadata,
    intent_set=intent_set,
    save_dir="output/eval_datasets",
))
```

### Run the Streamlit UI

```bash
# With pre-parsed agent (recommended)
python -m adk_eval_tool examples.weather_agent.agent root_agent

# Or standalone (load agent manually in the UI)
streamlit run adk_eval_tool/ui/app.py
```

The `.env` file is loaded automatically — `GOOGLE_API_KEY` is available on all pages.

The UI provides 7 pages:

| Page | What it does |
|------|-------------|
| **Agent Metadata** | Upload metadata JSON or parse a live agent module. View/edit the agent tree. |
| **Intents & Scenarios** | Generate intents via LLM or upload existing ones. Edit individual intents. Selective regeneration. |
| **Test Cases** | Generate `.evalset.json` files. Trajectory-aware editor with per-invocation tool calls, tool responses, rubrics. Add/delete eval cases. |
| **Eval Config** | Configure all 9 ADK built-in metrics with per-metric thresholds, match types, judge model. |
| **Run Evaluation** | Select eval sets, launch evaluation runs against a live agent. |
| **Eval Results** | Explore per-case scores, per-invocation tool trajectory diffs, basic metrics (tokens, latency), trace trees. Historical result browsing. |
| **Dataset Versions** | Create/browse/load/delete versioned snapshots of metadata + intents + eval sets. |

### Run ADK eval directly

After generating eval datasets, you can run them with ADK's built-in eval command:

```bash
# Run eval on a single file
adk eval examples.weather_agent.agent examples/weather_agent/output/eval_datasets/weather_agent__get_current_weather_happy_path.evalset.json

# Run eval on all files in a directory
adk eval examples.weather_agent.agent examples/weather_agent/output/eval_datasets/
```

## Project Structure

```
adk_eval_tool/
├── schemas.py                  # Pydantic models (AgentMetadata, Intent, Scenario, EvalRunResult, etc.)
├── agent_parser/               # parse_agent() and parse_agent_async() for MCP
├── intent_generator/           # ADK agent that generates intents + scenarios
├── testcase_generator/         # ADK agent that generates .evalset.json files
├── eval_runner/                # Run evals with OTel traces + result capture
│   ├── runner.py               # run_evaluation() wrapping LocalEvalService
│   ├── trace_collector.py      # SqliteSpanExporter setup, span→tree builder
│   └── result_store.py         # File-based result persistence + averages
└── ui/                         # Streamlit app (7 pages)

examples/
├── weather_agent/              # Single agent, 2 tools
└── travel_multi_agent/         # Coordinator + 2 sub-agents with tools

tests/                          # 45 tests
```

## Pipeline Flow

```
ADK Agent object
      │
      ▼
  parse_agent()  ──►  AgentMetadata (JSON)
      │
      ▼
  generate_intents()  ──►  IntentScenarioSet (JSON)
      │
      ▼
  generate_test_cases()  ──►  .evalset.json (ADK format)
      │
      ▼
  run_evaluation() or adk eval  ──►  EvalRunResult with traces + scores
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Examples

- [`examples/weather_agent/`](examples/weather_agent/) — Single agent with `get_weather` and `get_forecast` tools
- [`examples/travel_multi_agent/`](examples/travel_multi_agent/) — Multi-agent: coordinator delegates to `flight_agent` and `hotel_agent`
