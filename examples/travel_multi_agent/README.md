# Travel Multi-Agent Example

A multi-agent ADK application with a coordinator and two specialist sub-agents — demonstrates how adk-eval-tool handles agent hierarchies and tool delegation.

## Agent Structure

```
travel_coordinator (LlmAgent)
├── flight_agent (LlmAgent)
│   └── search_flights(origin, destination, date)
└── hotel_agent (LlmAgent)
    └── search_hotels(city, check_in, check_out)
```

The coordinator decides which specialist to delegate to based on user intent. For full trip planning, it uses both sub-agents.

## Running the Pipeline

```bash
# From the project root — run the full pipeline
python examples/travel_multi_agent/run_pipeline.py

# Or launch the UI with agent pre-loaded
python -m adk_eval_tool examples.travel_multi_agent.agent root_agent
```

The pipeline script will:

1. **Parse** the multi-agent tree into `output/metadata.json` (recursively captures all sub-agents and their tools)
2. **Generate intents & scenarios** via Gemini into `output/intents.json`
3. **Generate ADK eval test cases** into `output/eval_datasets/*.evalset.json`

### Sample output

```
Step 1: Parsing multi-agent metadata...
  Root agent: travel_coordinator (LlmAgent)
  Sub-agents:
    - flight_agent: tools=['search_flights']
    - hotel_agent: tools=['search_hotels']

Step 2: Generating intents and scenarios...
  Generated 4 intents:
    - Book Flight Only: 2 scenarios
    - Book Hotel Only: 2 scenarios
    - Book Flight and Hotel: 2 scenarios
    - Ambiguous Request: 2 scenarios

Step 3: Generating ADK eval test cases...
  Generated 4 eval set(s), 8 test cases total
```

## Generated Files

```
output/
├── metadata.json                                               # Full agent tree
├── intents.json                                                # Intents + scenarios
└── eval_datasets/
    ├── travel_coordinator__book_flight_only.evalset.json
    ├── travel_coordinator__book_hotel_only.evalset.json
    ├── travel_coordinator__book_flight_and_hotel.evalset.json
    └── travel_coordinator__ambiguous_request.evalset.json
```

## Running ADK Eval

```bash
# Single eval set
adk eval examples.travel_multi_agent.agent \
  examples/travel_multi_agent/output/eval_datasets/travel_coordinator__book_flight_only.evalset.json

# All eval sets
adk eval examples.travel_multi_agent.agent \
  examples/travel_multi_agent/output/eval_datasets/
```

## Using the Streamlit UI

```bash
# Launch UI with the multi-agent pre-loaded (recommended)
python -m adk_eval_tool examples.travel_multi_agent.agent root_agent

# Or launch standalone
streamlit run adk_eval_tool/ui/app.py
```

With the CLI launcher, the full agent tree is pre-loaded. Then:

1. **Agent Metadata** — already shows the coordinator + sub-agents tree
2. **Intents & Scenarios** → Generate or upload `output/intents.json`
3. **Test Cases** → Generate trajectory-based eval sets covering sub-agent delegation
4. **Eval Config** → Agent module is pre-filled; configure `tool_trajectory_avg_score` with `IN_ORDER` match
5. **Run Evaluation** → Launch eval; coordinator routes to sub-agents which call their tools
6. **Eval Results** → Per-invocation tool trajectory diffs (expected vs actual)

## Using the Python API Directly

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()

from examples.travel_multi_agent.agent import root_agent
from adk_eval_tool.agent_parser import parse_agent
from adk_eval_tool.intent_generator import generate_intents

# Parse the full agent tree
metadata = parse_agent(root_agent)
print(f"Root: {metadata.name}")
for sub in metadata.sub_agents:
    print(f"  {sub.name}: tools={[t.name for t in sub.tools]}")

# Generate intents covering all sub-agents
intent_set = asyncio.run(generate_intents(
    metadata,
    user_constraints="Ensure both flight_agent and hotel_agent are tested",
))
for intent in intent_set.intents:
    print(f"  {intent.name}: {len(intent.scenarios)} scenarios")
```

## What Gets Tested

The generated eval sets cover multi-agent delegation patterns:

| Intent | Sub-agents Exercised | Scenario Types |
|--------|---------------------|----------------|
| Flight booking | `flight_agent` → `search_flights` | happy_path, edge_case (missing dates) |
| Hotel booking | `hotel_agent` → `search_hotels` | happy_path, error (invalid dates) |
| Complete trip | `flight_agent` + `hotel_agent` | happy_path, error (unknown city) |
| Ambiguous request | Coordinator only (clarification) | edge_case |

## Key Differences from Single-Agent Example

- **Recursive metadata parsing**: The parser walks the full agent tree, capturing sub-agent instructions, descriptions, and tools
- **Sub-agent delegation in trajectories**: Test scenarios include expected delegation to specific sub-agents
- **Tool trajectory match type**: Use `IN_ORDER` for multi-agent since the coordinator may interleave delegation calls
- **Cross-agent coverage**: The intent generator ensures every sub-agent and tool appears in at least one scenario
