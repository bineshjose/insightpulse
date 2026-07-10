"""Navigation helpers shared by dashboard pages."""

from __future__ import annotations

import streamlit as st
from streamlit.errors import StreamlitAPIException


def page_link(page: str, label: str) -> None:
    """Render a page link, degrading to plain text outside a multipage app.

    ``st.page_link`` requires the target page to be registered in the
    multipage registry; when a page runs standalone (AppTest, or
    ``streamlit run`` on a single page file) the lookup fails. Fall back
    to a plain label so the page still renders.

    Args:
        page: Path to the target page, relative to the main script.
        label: Link label.
    """
    try:
        st.page_link(page, label=label)
    except (KeyError, StreamlitAPIException):
        st.markdown(label)
