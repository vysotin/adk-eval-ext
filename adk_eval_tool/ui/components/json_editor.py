"""JSON editor widget for Streamlit with auto-format."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st


def json_editor(
    data: dict[str, Any],
    key: str,
    readonly: bool = False,
    height: int = 400,
) -> dict[str, Any]:
    """Render an editable JSON text area with format-on-apply.

    The editor stores raw text in session state. A "Format & Apply" button
    pretty-prints the JSON and validates it. This avoids the Streamlit
    rerun-on-every-keystroke problem while keeping JSON tidy.

    Args:
        data: The JSON data to display/edit.
        key: Unique Streamlit key prefix.
        readonly: If True, display only.
        height: Text area height in pixels.

    Returns:
        The edited (or original) data.
    """
    if readonly:
        st.code(json.dumps(data, indent=2), language="json")
        return data

    # Use session state to persist the raw text between reruns.
    # Initialize from data if this is the first render for this key.
    state_key = f"_json_raw_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = json.dumps(data, indent=2)

    # If upstream data changed (e.g. after add/delete), resync
    current_canonical = _try_canonical(st.session_state[state_key])
    data_canonical = json.dumps(data, sort_keys=True)
    if current_canonical is not None and current_canonical != data_canonical:
        # Data was changed externally — reset editor text
        st.session_state[state_key] = json.dumps(data, indent=2)

    edited_str = st.text_area(
        "JSON",
        value=st.session_state[state_key],
        height=height,
        key=f"{key}_editor",
        label_visibility="collapsed",
    )

    # Always store the latest typed text
    st.session_state[state_key] = edited_str

    col_fmt, col_status = st.columns([1, 3])
    with col_fmt:
        format_clicked = st.button(
            "Format & Apply",
            key=f"{key}_format",
            type="secondary",
            use_container_width=True,
        )

    if format_clicked:
        try:
            parsed = json.loads(edited_str)
            formatted = json.dumps(parsed, indent=2)
            st.session_state[state_key] = formatted
            with col_status:
                st.success("JSON formatted and applied.", icon="\u2705")
            return parsed
        except json.JSONDecodeError as e:
            with col_status:
                st.error(f"Invalid JSON: {e}")
            return data

    # Even without clicking format, try to parse for the return value
    try:
        return json.loads(edited_str)
    except json.JSONDecodeError:
        return data


def _try_canonical(text: str) -> str | None:
    """Try to produce a canonical JSON string for comparison."""
    try:
        return json.dumps(json.loads(text), sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        return None
