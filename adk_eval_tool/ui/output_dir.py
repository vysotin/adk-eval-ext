"""Helpers for resolving output paths relative to the configured output directory."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def get_output_dir() -> str:
    """Return the current output directory from session state."""
    try:
        return st.session_state["output_dir"]
    except (KeyError, AttributeError):
        return "./output"


def get_output_path(*parts: str) -> str:
    """Join sub-paths under the output directory.

    Example::

        get_output_path("metadata", "agent.json")
        # → "./output/metadata/agent.json"
    """
    return str(Path(get_output_dir()) / Path(*parts))
