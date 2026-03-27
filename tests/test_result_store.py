"""Tests for eval result storage."""

from adk_eval_tool.eval_runner.result_store import ResultStore
from adk_eval_tool.schemas import EvalRunResult


def test_result_store_save_and_load(tmp_path):
    store = ResultStore(base_dir=str(tmp_path))
    result = EvalRunResult(
        run_id="run-1",
        eval_set_id="agent__book_flight",
        eval_id="book_flight__happy_path",
        status="PASSED",
        overall_scores={"tool_trajectory_avg_score": 0.95, "safety_v1": 1.0},
        per_invocation_scores=[
            {"invocation_id": "inv-1", "scores": {"tool_trajectory_avg_score": 0.95}},
        ],
        session_id="sess-1",
        timestamp=1000.0,
    )
    store.save_result(result)

    loaded = store.load_results()
    assert len(loaded) == 1
    assert loaded[0].run_id == "run-1"
    assert loaded[0].status == "PASSED"


def test_result_store_multiple_results(tmp_path):
    store = ResultStore(base_dir=str(tmp_path))
    for i in range(3):
        store.save_result(EvalRunResult(
            run_id=f"run-{i}",
            eval_set_id="agent__intent",
            eval_id=f"intent__scenario_{i}",
            status="PASSED" if i < 2 else "FAILED",
            overall_scores={"safety_v1": 1.0 if i < 2 else 0.5},
        ))

    loaded = store.load_results()
    assert len(loaded) == 3
    assert sum(1 for r in loaded if r.status == "PASSED") == 2


def test_result_store_load_by_eval_set(tmp_path):
    store = ResultStore(base_dir=str(tmp_path))
    store.save_result(EvalRunResult(
        run_id="run-1", eval_set_id="set_a", eval_id="case_1", status="PASSED",
    ))
    store.save_result(EvalRunResult(
        run_id="run-2", eval_set_id="set_b", eval_id="case_2", status="FAILED",
    ))

    results_a = store.load_results(eval_set_id="set_a")
    assert len(results_a) == 1
    assert results_a[0].eval_set_id == "set_a"


def test_result_store_compute_averages(tmp_path):
    store = ResultStore(base_dir=str(tmp_path))
    for score in [0.8, 0.9, 1.0]:
        store.save_result(EvalRunResult(
            run_id=f"run-{score}",
            eval_set_id="set_a",
            eval_id="case_1",
            status="PASSED",
            overall_scores={"tool_trajectory_avg_score": score, "safety_v1": 1.0},
        ))

    averages = store.compute_averages(eval_set_id="set_a")
    assert abs(averages["tool_trajectory_avg_score"] - 0.9) < 0.01
    assert averages["safety_v1"] == 1.0
