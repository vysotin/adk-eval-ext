# Task 1: Sub-Agent Response Mocking in ADK Evaluation Datasets

## Problem Statement

Google ADK's evaluation framework supports tool response mocking in the legacy `.test.json` format via `mock_tool_output`, but has **no built-in mechanism** for mocking sub-agent responses in the newer `.evalset.json` format. This report analyzes options to add JSON-configurable mocking for both tools and sub-agents.

---

## Current State of ADK Mocking

### Legacy Format (`.test.json`) -- Tool Mocking Only

The legacy format supports `mock_tool_output` inside `expected_tool_use`:

```json
[
  {
    "name": "weather_test",
    "data": [
      {
        "query": "What's the weather in SF?",
        "expected_tool_use": [
          {
            "tool_name": "get_weather",
            "tool_input": {"city": "San Francisco"},
            "mock_tool_output": {"temp": 65, "condition": "sunny"}
          }
        ],
        "reference": "It's 65F and sunny in San Francisco."
      }
    ]
  }
]
```

The `EvaluationGenerator` intercepts tool calls and returns `mock_tool_output` instead of executing the real tool when a match is found.

### Current EvalSet Format (`.evalset.json`) -- No Mocking

In the newer format, `intermediateData.toolResponses` and `intermediateData.intermediateResponses` are **golden/expected data for comparison only** -- they are never replayed during execution:

```json
{
  "intermediateData": {
    "toolUses": [
      {"name": "get_weather", "args": {"city": "SF"}}
    ],
    "toolResponses": [
      {"name": "get_weather", "id": "c1", "response": {"temp": 65}}
    ],
    "intermediateResponses": [
      ["forecast_agent", [{"text": "Tomorrow: 62F"}]]
    ]
  }
}
```

### Sub-Agent Handling

ADK captures sub-agent responses in `intermediate_responses` as tuples of `(agent_name, response_parts)`. Sub-agents are delegated to via the `sub_agents` parameter on `Agent`. There is no interception hook analogous to `before_tool_callback` for sub-agent delegation.

---

## Options Analysis

### Option A: Extend EvalSet JSON with `_mock_responses` Section (Recommended)

Add a new top-level field to `EvalCase` that defines mock responses for both tools and sub-agents, processed by a custom evaluation runner.

**Extended JSON Schema:**

```json
{
  "evalSetId": "travel_agent_tests",
  "evalCases": [
    {
      "evalId": "flight_booking_test",
      "mockResponses": {
        "tools": {
          "search_flights": [
            {
              "matchArgs": {"origin": "SFO", "destination": "JFK"},
              "response": {"flights": [{"id": "AA100", "price": 350}]}
            },
            {
              "matchArgs": {},
              "response": {"flights": [], "error": "No flights found"}
            }
          ],
          "get_weather": [
            {
              "matchArgs": {},
              "response": {"temp": 72, "condition": "clear"}
            }
          ]
        },
        "subAgents": {
          "research_agent": {
            "responses": [
              {
                "matchInput": ".*flight.*",
                "response": [{"text": "Based on research, AA100 is the best option at $350."}]
              }
            ],
            "defaultResponse": [{"text": "I found relevant information on that topic."}]
          },
          "booking_agent": {
            "responses": [
              {
                "matchInput": ".*book.*AA100.*",
                "response": [{"text": "Booking confirmed. Confirmation: ABC123"}]
              }
            ],
            "defaultResponse": [{"text": "Booking processed."}]
          }
        }
      },
      "conversation": [
        {
          "userContent": {"role": "user", "parts": [{"text": "Book me a flight from SFO to JFK"}]},
          "finalResponse": {"parts": [{"text": "I've booked flight AA100 for $350. Confirmation: ABC123"}]},
          "intermediateData": {
            "toolUses": [
              {"name": "search_flights", "args": {"origin": "SFO", "destination": "JFK"}}
            ],
            "intermediateResponses": [
              ["research_agent", [{"text": "Based on research, AA100 is the best option at $350."}]],
              ["booking_agent", [{"text": "Booking confirmed. Confirmation: ABC123"}]]
            ]
          }
        }
      ]
    }
  ]
}
```

**Implementation Approach -- Prototype:**

```python
# mock_interceptor.py -- Loads mockResponses from EvalCase and wires into agent

from google.adk.agents import Agent
from google.genai import types
import re, json, copy

class MockInterceptor:
    """Loads mock definitions from EvalCase JSON and intercepts tool/sub-agent calls."""

    def __init__(self, mock_responses: dict):
        self.tool_mocks = mock_responses.get("tools", {})
        self.sub_agent_mocks = mock_responses.get("subAgents", {})

    def before_tool_callback(self, tool, args: dict, tool_context):
        """Intercept tool calls -- return mock if match found, else None (run real)."""
        tool_rules = self.tool_mocks.get(tool.name, [])
        for rule in tool_rules:
            match_args = rule.get("matchArgs", {})
            if not match_args:  # Empty matchArgs = wildcard
                return rule["response"]
            if all(args.get(k) == v for k, v in match_args.items()):
                return rule["response"]
        return None  # No match -- execute real tool

    def build_mocked_agent(self, real_agent: Agent) -> Agent:
        """Clone agent tree, replacing specified sub-agents with mock stubs."""
        new_sub_agents = []
        for sub in (real_agent.sub_agents or []):
            if sub.name in self.sub_agent_mocks:
                mock_def = self.sub_agent_mocks[sub.name]
                # Create a minimal agent that returns canned responses
                stub = Agent(
                    name=sub.name,
                    model="gemini-2.0-flash",
                    instruction=self._build_stub_instruction(sub.name, mock_def),
                    description=sub.description,
                )
                new_sub_agents.append(stub)
            else:
                # Recursively process non-mocked sub-agents
                new_sub_agents.append(self.build_mocked_agent(sub))

        # Clone the parent agent with new sub-agents and tool mock callback
        return Agent(
            name=real_agent.name,
            model=real_agent.model,
            instruction=real_agent.instruction,
            tools=real_agent.tools,
            sub_agents=new_sub_agents,
            before_tool_callback=self.before_tool_callback,
            description=real_agent.description,
        )

    def _build_stub_instruction(self, agent_name: str, mock_def: dict) -> str:
        """Build instruction for stub agent that returns canned responses."""
        default = mock_def.get("defaultResponse", [{"text": "OK"}])
        default_text = " ".join(p.get("text", "") for p in default)

        responses = mock_def.get("responses", [])
        lines = [f"You are a test stub for '{agent_name}'.", ""]
        for r in responses:
            pattern = r.get("matchInput", ".*")
            text = " ".join(p.get("text", "") for p in r["response"])
            lines.append(f"If the user's message matches '{pattern}', respond exactly: {text}")
        lines.append(f"\nFor all other inputs, respond exactly: {default_text}")
        return "\n".join(lines)


# --- Usage in extended evaluation runner ---

def run_eval_with_mocks(agent_module, eval_set_path, config_path=None):
    """Extended adk eval that processes mockResponses from EvalCase."""
    import importlib
    from google.adk.evaluation import AgentEvaluator, EvalSet

    with open(eval_set_path) as f:
        eval_data = json.load(f)

    module = importlib.import_module(agent_module)
    real_agent = module.root_agent

    for case in eval_data.get("evalCases", []):
        mock_responses = case.get("mockResponses")
        if mock_responses:
            interceptor = MockInterceptor(mock_responses)
            test_agent = interceptor.build_mocked_agent(real_agent)
        else:
            test_agent = real_agent

        # Run evaluation with the (possibly mocked) agent
        # ... invoke EvaluationGenerator with test_agent ...
```

**Pros:**
- Single JSON file defines both test data and mocks
- Tool mocking uses `before_tool_callback` (ADK-native pattern)
- Sub-agent mocking via agent tree reconstruction preserves ADK's delegation model
- Argument-based matching with wildcard fallback
- Regex-based input matching for sub-agents
- Compatible with existing `EvalCase` fields (conversation, intermediateData, rubrics)

**Cons:**
- Sub-agent stubs still use an LLM call (lightweight model) -- not purely deterministic
- Requires custom evaluation runner (cannot use stock `adk eval` CLI without modification)
- `matchArgs` uses exact equality; complex matching (ranges, contains) would need extension

---

### Option B: `before_tool_callback` + `before_agent_callback` Dual Hook

Use ADK's `before_tool_callback` for tools and propose a new `before_agent_callback` hook for sub-agents.

**JSON Schema:**

```json
{
  "evalId": "test_1",
  "mockConfig": {
    "toolCallbacks": {
      "get_weather": {"default": {"temp": 72}},
      "search": {"byArgs": [{"q": "flights", "result": []}]}
    },
    "agentCallbacks": {
      "research_agent": {"default": "Research complete."},
      "booking_agent": {"default": "Booking confirmed."}
    }
  }
}
```

**Prototype:**

```python
# This approach requires ADK to support before_agent_callback (does not exist today)
# Proposed API:

class Agent:
    def __init__(
        self,
        ...,
        before_agent_callback: Optional[Callable] = None,  # NEW
        after_agent_callback: Optional[Callable] = None,    # NEW
    ):
        ...

# The callback signature:
def before_agent_callback(
    sub_agent: Agent,
    user_message: Content,
    agent_context: AgentContext,
) -> Optional[Content]:
    """Return Content to short-circuit sub-agent, or None to proceed."""
    ...

# Mock loader using both callbacks:
class DualMockLoader:
    def __init__(self, mock_config: dict):
        self.tool_mocks = mock_config.get("toolCallbacks", {})
        self.agent_mocks = mock_config.get("agentCallbacks", {})

    def tool_callback(self, tool, args, ctx):
        config = self.tool_mocks.get(tool.name)
        if not config:
            return None
        for rule in config.get("byArgs", []):
            if all(args.get(k) == v for k, v in rule.items() if k != "result"):
                return rule["result"]
        return config.get("default")

    def agent_callback(self, sub_agent, user_message, ctx):
        config = self.agent_mocks.get(sub_agent.name)
        if not config:
            return None
        return types.Content(parts=[types.Part(text=config["default"])])
```

**Pros:**
- Clean separation: tool mocking via existing hook, sub-agent mocking via new hook
- Fully deterministic sub-agent mocking (no LLM call)
- Callback-based approach is consistent with ADK patterns

**Cons:**
- `before_agent_callback` does not exist in ADK today -- requires upstream contribution
- Until ADK adopts this, it's not usable without forking
- Callback registration is per-agent; nested sub-agents need recursive wiring

---

### Option C: Replay-Based Mocking from IntermediateData

Instead of intercepting live calls, replay recorded tool and sub-agent responses from the `intermediateData` field directly.

**JSON (uses existing schema, no new fields):**

```json
{
  "evalId": "replay_test",
  "replayMode": true,
  "conversation": [
    {
      "userContent": {"role": "user", "parts": [{"text": "Book a flight"}]},
      "intermediateData": {
        "toolResponses": [
          {"name": "search_flights", "id": "c1", "response": {"flights": [{"id": "AA100"}]}}
        ],
        "intermediateResponses": [
          ["booking_agent", [{"text": "Booked AA100. Conf: ABC123"}]]
        ]
      },
      "finalResponse": {"parts": [{"text": "Flight AA100 booked."}]}
    }
  ]
}
```

**Prototype:**

```python
class ReplayInterceptor:
    """Replays tool and sub-agent responses from intermediateData in order."""

    def __init__(self, invocations: list):
        self._tool_queue = {}  # {tool_name: deque of responses}
        self._agent_queue = {}  # {agent_name: deque of responses}
        for inv in invocations:
            idata = inv.get("intermediateData", {})
            for tr in idata.get("toolResponses", []):
                self._tool_queue.setdefault(tr["name"], []).append(tr["response"])
            for ar in idata.get("intermediateResponses", []):
                self._agent_queue.setdefault(ar[0], []).append(ar[1])

    def before_tool_callback(self, tool, args, ctx):
        queue = self._tool_queue.get(tool.name, [])
        if queue:
            return queue.pop(0)
        return None  # Fallback to real execution

    def get_agent_response(self, agent_name):
        queue = self._agent_queue.get(agent_name, [])
        if queue:
            return queue.pop(0)
        return None
```

**Pros:**
- Uses existing `intermediateData` schema -- no new JSON fields
- Golden data doubles as mock data (single source of truth)
- Order-based replay matches multi-turn conversation flow

**Cons:**
- Order-dependent: if agent makes calls in different order, wrong response is returned
- Cannot handle dynamic/branching agent behavior (different paths per run)
- No argument-based matching -- purely sequential replay
- Conflates expected data with mock data (editing one changes the other)

---

### Option D: External Mock Service via MCP

Run a lightweight MCP server that serves mock responses, referenced from the eval dataset.

**JSON:**

```json
{
  "evalId": "mcp_mock_test",
  "mockServer": {
    "type": "mcp",
    "configPath": "./mock_server_config.json"
  }
}
```

**`mock_server_config.json`:**

```json
{
  "tools": {
    "get_weather": {
      "responses": [
        {"args": {"city": "SF"}, "result": {"temp": 65}},
        {"args": {}, "result": {"temp": 70}}
      ]
    }
  },
  "agents": {
    "research_agent": {
      "endpoint": "http://localhost:9090/mock/research_agent",
      "responses": [
        {"input": ".*", "output": "Mock research result."}
      ]
    }
  }
}
```

**Pros:**
- Framework-agnostic -- any agent framework can call the mock MCP server
- Sub-agent mocking via A2A protocol mock endpoint
- Reusable across tests
- Can simulate latency, errors, and complex behaviors

**Cons:**
- Requires running a separate service -- infrastructure overhead
- More complex setup than callback-based approaches
- Latency from network calls even on localhost
- Not integrated with ADK's evaluation runner

---

## Comparison Matrix

| Criterion | Option A (JSON + Interceptor) | Option B (Dual Callback) | Option C (Replay) | Option D (MCP Mock) |
|-----------|------|------|------|------|
| Works today (no ADK changes) | Yes | No (needs new callback) | Partially | Yes |
| Tool mocking | Full (arg matching) | Full (arg matching) | Sequential only | Full |
| Sub-agent mocking | Via stub agents (LLM) | Deterministic (callback) | Sequential only | Via mock endpoints |
| JSON-configurable | Yes | Yes | Uses existing schema | Separate config |
| Deterministic | Mostly (stubs use LLM) | Fully | Fully | Fully |
| ADK Web UI compatible | With custom runner | With ADK changes | Compatible | Needs UI extension |
| Complexity | Medium | Low (if ADK supports) | Low | High |

---

## Recommendation

**Option A** is the best near-term choice: it works without ADK changes, supports both tool and sub-agent mocking from a single JSON file, and integrates naturally with ADK's `before_tool_callback`.

For long-term, **Option B** (proposing `before_agent_callback` upstream to ADK) would provide the cleanest, most deterministic sub-agent mocking. The two options are complementary -- Option A can be implemented now while Option B is proposed as an ADK enhancement.

### ADK Web UI Compatibility

To make mocked evaluations visible in `adk web`:

1. **Custom FastAPI endpoint**: Add a `/apps/{app_name}/eval_sets/{id}/run_eval_mocked` endpoint to `get_fast_api_app()` that loads `mockResponses` and runs with the `MockInterceptor`
2. **Trace preservation**: The mocked agent still produces standard ADK events, so traces display normally in the UI
3. **Mock indicator**: Add a `_mocked: true` flag to `Invocation.app_details` so the UI can highlight mocked responses (requires minor Angular component change in `adk-web`)

```python
# Extending adk_web_server.py
@app.post("/apps/{app_name}/eval_sets/{eval_set_id}/run_eval_mocked")
async def run_mocked_eval(app_name: str, eval_set_id: str, config: EvalRunConfig):
    eval_set = eval_sets_manager.get_eval_set(app_name, eval_set_id)
    agent = load_agent(app_name)

    results = []
    for case in eval_set.eval_cases:
        mock_data = case.model_extra.get("mockResponses")  # Pydantic extra fields
        if mock_data:
            interceptor = MockInterceptor(mock_data)
            test_agent = interceptor.build_mocked_agent(agent)
        else:
            test_agent = agent
        result = await run_single_eval(test_agent, case, config)
        results.append(result)
    return results
```
