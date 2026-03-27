"""Generation progress/status display component."""

from __future__ import annotations

from typing import Optional

import streamlit as st


def run_status_display(
    status: str,
    progress: float = 0.0,
    message: str = "",
    details: Optional[dict] = None,
):
    """Display generation run status.

    Args:
        status: One of "idle", "running", "completed", "failed".
        progress: Progress fraction 0.0 to 1.0.
        message: Status message.
        details: Optional details dict.
    """
    if status == "idle":
        st.info("Ready to run.")
    elif status == "running":
        st.progress(progress, text=message)
    elif status == "completed":
        st.success(message or "Completed successfully.")
    elif status == "failed":
        st.error(message or "Generation failed.")

    if details:
        with st.expander("Details"):
            st.json(details)
