"""POC: Split ADK evaluation into inference and evaluation stages.

Stage 1 — Inference: runs the agent against eval cases, captures
InferenceResult objects (with Invocations) and OTel traces, serialises
everything to disk. No scoring happens.

Stage 2 — Evaluation: loads persisted InferenceResults and eval sets,
scores them against configurable metrics using ADK's LocalEvalService.
No agent or LLM calls needed (except LLM-as-a-judge metrics).
"""
