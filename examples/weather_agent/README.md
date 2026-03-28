# Weather Agent Example

A single ADK agent with two tools — demonstrates the simplest use case for adk-eval-tool.

## Agent Structure

```
weather_agent (LlmAgent)
├── get_weather(city)      — current temperature, condition, humidity
└── get_forecast(city, days) — multi-day forecast (1-7 days)
```

The agent answers weather questions by choosing the right tool (or both) based on user intent.

## Running the Pipeline

```bash
# From the project root — run the full pipeline (parse + generate intents + generate test cases)
python examples/weather_agent/run_pipeline.py

# Or launch the UI with agent pre-loaded
python -m adk_eval_tool examples.weather_agent.agent root_agent
```

The pipeline script will:

1. **Parse** the agent into `output/metadata.json`
2. **Generate intents & scenarios** via Gemini into `output/intents.json`
3. **Generate ADK eval test cases** into `output/eval_datasets/*.evalset.json`

### Sample output

```
Step 1: Parsing agent metadata...
  Agent: weather_agent (LlmAgent)
  Tools: ['get_weather', 'get_forecast']

Step 2: Generating intents and scenarios...
  Generated 5 intents:
    - Get Current Weather - Happy Path: 2 scenarios
    - Get Current Weather - Edge Cases: 2 scenarios
    - Get Forecast - Happy Path: 2 scenarios
    - Get Forecast - Edge Cases: 2 scenarios
    - Combined Weather Requests: 2 scenarios

Step 3: Generating ADK eval test cases...
  Generated 5 eval set(s), 10 test cases total
```

## Generated Files

```
output/
├── metadata.json                                          # Agent structure
├── intents.json                                           # Intents + scenarios
└── eval_datasets/
    ├── weather_agent__get_current_weather_happy_path.evalset.json
    ├── weather_agent__get_current_weather_edge_cases.evalset.json
    ├── weather_agent__get_forecast_happy_path.evalset.json
    ├── weather_agent__get_forecast_edge_cases.evalset.json
    └── weather_agent__combined_weather_requests.evalset.json
```

## Running ADK Eval

After generating the datasets, run the agent against them:

```bash
# Single eval set
adk eval examples.weather_agent.agent \
  examples/weather_agent/output/eval_datasets/weather_agent__get_current_weather_happy_path.evalset.json

# All eval sets
adk eval examples.weather_agent.agent \
  examples/weather_agent/output/eval_datasets/
```

## Using the Streamlit UI

```bash
# Launch UI with agent pre-loaded (recommended)
python -m adk_eval_tool examples.weather_agent.agent root_agent

# Or launch standalone and load files manually
streamlit run adk_eval_tool/ui/app.py
```

With the CLI launcher, the agent metadata is already loaded when the UI opens. Then:

1. Go to **Intents & Scenarios** → Generate intents or upload `output/intents.json`
2. Go to **Test Cases** → Generate eval sets or upload `.evalset.json` files
3. Go to **Eval Config** → Agent module is pre-filled; set metrics and thresholds
4. Go to **Run Evaluation** → Launch eval against the live agent
5. Go to **Eval Results** → Explore scores and traces

## Using the Python API Directly

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()

from examples.weather_agent.agent import root_agent
from adk_eval_tool.agent_parser import parse_agent
from adk_eval_tool.intent_generator import generate_intents

# Parse
metadata = parse_agent(root_agent)
print(f"Tools: {[t.name for t in metadata.tools]}")

# Generate intents
intent_set = asyncio.run(generate_intents(metadata))
for intent in intent_set.intents:
    print(f"  {intent.name}: {len(intent.scenarios)} scenarios")
```

## What Gets Tested

The generated eval sets cover:

| Intent | Tools Exercised | Scenario Types |
|--------|----------------|----------------|
| Current weather (known city) | `get_weather` | happy_path |
| Current weather (unknown city) | `get_weather` | edge_case |
| Forecast (default days) | `get_forecast` | happy_path |
| Forecast (custom days / invalid) | `get_forecast` | edge_case |
| General weather question | `get_weather` + `get_forecast` | happy_path |
