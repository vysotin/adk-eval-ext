"""Tests for POC split-eval schemas."""

import json

from poc_split_eval.schemas import InferenceArtifact, InferenceBundle


def test_inference_artifact_roundtrip():
    artifact = InferenceArtifact(
        eval_set_json={"eval_set_id": "set1", "eval_cases": []},
        inference_result_json={
            "app_name": "app",
            "eval_set_id": "set1",
            "eval_case_id": "case1",
            "session_id": "sess-1",
            "status": 1,
            "inferences": [],
        },
        session_id="sess-1",
    )
    j = artifact.model_dump_json()
    loaded = InferenceArtifact.model_validate_json(j)
    assert loaded.session_id == "sess-1"
    assert loaded.eval_set_json["eval_set_id"] == "set1"


def test_inference_bundle_roundtrip():
    bundle = InferenceBundle(
        app_name="test_app",
        agent_module="examples.weather_agent.agent",
        artifacts=[
            InferenceArtifact(
                eval_set_json={"eval_set_id": "s1", "eval_cases": []},
                inference_result_json={"app_name": "a", "eval_set_id": "s1",
                                       "eval_case_id": "c1", "session_id": "x",
                                       "status": 1, "inferences": []},
                session_id="x",
            ),
        ],
        metadata={"num_runs": 2},
    )
    j = bundle.model_dump_json(indent=2)
    loaded = InferenceBundle.model_validate_json(j)
    assert loaded.app_name == "test_app"
    assert len(loaded.artifacts) == 1
    assert loaded.metadata["num_runs"] == 2


def test_bundle_to_file(tmp_path):
    bundle = InferenceBundle(app_name="x", artifacts=[])
    path = tmp_path / "bundle.json"
    path.write_text(bundle.model_dump_json(indent=2))
    loaded = InferenceBundle.model_validate_json(path.read_text())
    assert loaded.app_name == "x"
