"""Persist and query evaluation results."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Optional

from adk_eval_tool.schemas import EvalRunResult


class ResultStore:
    """File-based store for evaluation results."""

    def __init__(self, base_dir: str = "eval_results"):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, result: EvalRunResult) -> Path:
        """Save an eval run result to disk."""
        filename = f"{result.run_id}.json"
        path = self._base_dir / filename
        path.write_text(result.model_dump_json(indent=2))
        return path

    def load_results(
        self,
        eval_set_id: Optional[str] = None,
    ) -> list[EvalRunResult]:
        """Load all results, optionally filtered by eval_set_id."""
        results = []
        for path in sorted(self._base_dir.glob("*.json")):
            try:
                result = EvalRunResult.model_validate_json(path.read_text())
                if eval_set_id is None or result.eval_set_id == eval_set_id:
                    results.append(result)
            except Exception:
                continue
        return results

    def compute_averages(
        self,
        eval_set_id: Optional[str] = None,
    ) -> dict[str, float]:
        """Compute average scores across all results for an eval set."""
        results = self.load_results(eval_set_id=eval_set_id)
        if not results:
            return {}

        all_scores: dict[str, list[float]] = {}
        for result in results:
            for metric, score in result.overall_scores.items():
                if score is not None:
                    all_scores.setdefault(metric, []).append(score)

        return {
            metric: statistics.mean(scores)
            for metric, scores in all_scores.items()
        }
