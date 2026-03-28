"""Run the full adk-eval-tool pipeline on the travel multi-agent.

Steps:
  1. Parse the multi-agent into metadata (recursive tree)
  2. Generate intents and scenarios using LLM
  3. Generate ADK-compatible eval test cases
  4. Save all outputs to examples/travel_multi_agent/output/
"""

import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from examples.travel_multi_agent.agent import root_agent
from adk_eval_tool.agent_parser import parse_agent
from adk_eval_tool.intent_generator import generate_intents
from adk_eval_tool.testcase_generator.agent import generate_all_test_cases
from adk_eval_tool.schemas import TestCaseConfig


OUTPUT_DIR = Path(__file__).parent / "output"


async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Step 1: Parse agent metadata ---
    print("=" * 60)
    print("Step 1: Parsing multi-agent metadata...")
    print("=" * 60)

    metadata = parse_agent(
        root_agent,
        save_path=str(OUTPUT_DIR / "metadata.json"),
    )

    print(f"  Root agent: {metadata.name} ({metadata.agent_type})")
    print(f"  Tools: {[t.name for t in metadata.tools]}")
    print(f"  Sub-agents:")
    for sub in metadata.sub_agents:
        print(f"    - {sub.name}: tools={[t.name for t in sub.tools]}")
    print(f"  Saved to: {OUTPUT_DIR / 'metadata.json'}")
    print()

    # --- Step 2: Generate intents and scenarios ---
    print("=" * 60)
    print("Step 2: Generating intents and scenarios...")
    print("=" * 60)

    intent_set = await generate_intents(
        metadata=metadata,
        user_constraints=(
            "Cover these key scenarios:\n"
            "- User wants flights only\n"
            "- User wants hotels only\n"
            "- User wants a complete trip (flights + hotels)\n"
            "- Edge cases: missing dates, unknown cities\n"
            "- Ensure both sub-agents (flight_agent, hotel_agent) are exercised"
        ),
        num_scenarios_per_intent=2,
        save_path=str(OUTPUT_DIR / "intents.json"),
        model="gemini-2.0-flash",
    )

    print(f"  Generated {len(intent_set.intents)} intents:")
    for intent in intent_set.intents:
        print(f"    - {intent.name} ({intent.intent_id}): {len(intent.scenarios)} scenarios")
        for scenario in intent.scenarios:
            tools_used = set()
            for step in scenario.steps:
                tools_used.update(step.expected_tool_calls)
            print(f"      [{', '.join(scenario.tags)}] {scenario.name} -> tools: {tools_used or 'none'}")
    print(f"  Saved to: {OUTPUT_DIR / 'intents.json'}")
    print()

    # --- Step 3: Generate eval test cases ---
    print("=" * 60)
    print("Step 3: Generating ADK eval test cases...")
    print("=" * 60)

    config = TestCaseConfig(
        eval_metrics={
            "tool_trajectory_avg_score": 0.7,
            "safety_v1": 1.0,
        },
        judge_model="gemini-2.0-flash",
        tool_trajectory_match_type="IN_ORDER",
    )

    eval_sets = await generate_all_test_cases(
        metadata=metadata,
        intent_set=intent_set,
        config=config,
        save_dir=str(OUTPUT_DIR / "eval_datasets"),
    )

    print(f"  Generated {len(eval_sets)} eval set(s):")
    for es in eval_sets:
        eval_set_id = es.get("evalSetId", "unknown")
        num_cases = len(es.get("evalCases", []))
        print(f"    - {eval_set_id}: {num_cases} test case(s)")

        filepath = OUTPUT_DIR / "eval_datasets" / f"{eval_set_id}.evalset.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(es, indent=2))

    print(f"  Saved to: {OUTPUT_DIR / 'eval_datasets/'}")
    print()

    # --- Summary ---
    print("=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Metadata:    {OUTPUT_DIR / 'metadata.json'}")
    print(f"  Intents:     {OUTPUT_DIR / 'intents.json'}")
    print(f"  Eval sets:   {OUTPUT_DIR / 'eval_datasets/'}")
    print()
    print("Agent tree:")
    _print_agent_tree(metadata)
    print()
    print("To run the Streamlit UI:")
    print("  streamlit run adk_eval_tool/ui/app.py")
    print()
    print("To run ADK eval on the generated datasets:")
    print("  adk eval examples.travel_multi_agent.agent <eval_dataset_path>")


def _print_agent_tree(meta, indent=0):
    prefix = "  " * indent
    tools_str = ", ".join(t.name for t in meta.tools) if meta.tools else "(none)"
    print(f"{prefix}{meta.name} ({meta.agent_type}) [tools: {tools_str}]")
    for sub in meta.sub_agents:
        _print_agent_tree(sub, indent + 1)


if __name__ == "__main__":
    asyncio.run(main())
