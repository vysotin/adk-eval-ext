"""Run the full adk-eval-tool pipeline on the weather agent.

Steps:
  1. Parse the agent into metadata
  2. Generate tasks and base trajectories using LLM
  3. Generate ADK-compatible eval test cases
  4. Save all outputs to examples/weather_agent/output/
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from examples.weather_agent.agent import root_agent
from adk_eval_tool.agent_parser import parse_agent
from adk_eval_tool.task_generator import generate_tasks
from adk_eval_tool.testcase_generator.agent import generate_all_test_cases
from adk_eval_tool.schemas import TestCaseConfig


OUTPUT_DIR = Path(__file__).parent / "output"


async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Step 1: Parse agent metadata ---
    print("=" * 60)
    print("Step 1: Parsing agent metadata...")
    print("=" * 60)

    metadata = parse_agent(
        root_agent,
        save_path=str(OUTPUT_DIR / "metadata.json"),
    )

    print(f"  Agent: {metadata.name} ({metadata.agent_type})")
    print(f"  Tools: {[t.name for t in metadata.tools]}")
    print(f"  Sub-agents: {[a.name for a in metadata.sub_agents]}")
    print(f"  Saved to: {OUTPUT_DIR / 'metadata.json'}")
    print()

    # --- Step 2: Generate tasks and base trajectories ---
    print("=" * 60)
    print("Step 2: Generating tasks and base trajectories...")
    print("=" * 60)

    task_set = await generate_tasks(
        metadata=metadata,
        user_constraints="Focus on both tools being used. Cover all distinct user tasks.",
        save_path=str(OUTPUT_DIR / "tasks.json"),
        model="gemini-2.0-flash",
    )

    print(f"  Generated {len(task_set.tasks)} tasks:")
    for task in task_set.tasks:
        print(f"    - {task.name} ({task.task_id}): {len(task.trajectories)} trajectories")
    print(f"  Saved to: {OUTPUT_DIR / 'tasks.json'}")
    print()

    # --- Step 3: Generate eval test cases ---
    print("=" * 60)
    print("Step 3: Generating ADK eval test cases...")
    print("=" * 60)

    config = TestCaseConfig(
        eval_metrics={
            "tool_trajectory_avg_score": 0.8,
            "safety_v1": 1.0,
        },
        judge_model="gemini-2.0-flash",
        tool_trajectory_match_type="IN_ORDER",
    )

    eval_sets = await generate_all_test_cases(
        metadata=metadata,
        task_set=task_set,
        config=config,
        save_dir=str(OUTPUT_DIR / "eval_datasets"),
    )

    print(f"  Generated {len(eval_sets)} eval set(s):")
    for es in eval_sets:
        eval_set_id = es.get("evalSetId", "unknown")
        num_cases = len(es.get("evalCases", []))
        print(f"    - {eval_set_id}: {num_cases} test case(s)")

    print(f"  Saved to: {OUTPUT_DIR / 'eval_datasets/'}")
    print()

    print("=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
