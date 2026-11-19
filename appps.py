                }
            )
        st.dataframe(pd.DataFrame(profile), width="stretch", height=320, hide_index=True)


def render_activity_tab() -> None:
    ss = st.session_state
    st.markdown(
        '<div class="dfx-sec-title">Activity log</div>'
        '<div class="dfx-sec-sub">A running record of this cleaning session.</div>',
        unsafe_allow_html=True,
    )
    if not ss["log"]:
        st.markdown(
            f'<div class="dfx-empty"><div class="dfx-empty-icon">{SVG["check"]}</div>'
            f'<div class="dfx-empty-title">Nothing logged yet</div>'
            f'<div class="dfx-empty-text">Actions you apply will be captured here automatically.</div></div>',
            unsafe_allow_html=True,
        )
        return

    tones = {"load": "blue", "missing": "green", "dedupe": "green", "text": "blue",
             "columns": "amber", "reset": "red"}
    items = []
    for entry in reversed(ss["log"][-100:]):
        tone = tones.get(entry["kind"], "blue")
        impact = f'<div class="dfx-tl-impact">{esc(entry["impact"])}</div>' if entry["impact"] else ""
        items.append(
            f'<div class="dfx-tl-item"><span class="dfx-tl-dot {tone}"></span>'
            f'<div class="dfx-tl-head"><span class="dfx-tl-title">{esc(entry["action"])}</span>'
            f'<span class="dfx-tl-time">{esc(entry["timestamp"])}</span></div>'
            f'<div class="dfx-tl-detail">{esc(entry["detail"])}</div>{impact}</div>'
        )
    st.markdown('<div class="dfx-timeline">' + "".join(items) + "</div>", unsafe_allow_html=True)
    st.caption("The change log can be exported from the sidebar.")


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("df", None)
    ss.setdefault("original", None)
    ss.setdefault("rid", RID_BASE)
    ss.setdefault("file_name", None)
    ss.setdefault("file_size", 0)
    ss.setdefault("sheet", None)
    ss.setdefault("source", None)
    ss.setdefault("changes", {})
    ss.setdefault("removed", [])
    ss.setdefault("dropped", [])
    ss.setdefault("log", [])
    ss.setdefault("flash", None)
    ss.setdefault("epoch", 0)
    ss.setdefault("failed_sig", None)


def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()
    if st.session_state.get("df") is None:
        render_empty_state()
    else:
        render_dashboard()


main()
"""DataFix Studio — a premium data cleaning workspace."""

from __future__ import annotations

import csv
import html
import io
import numbers
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype, is_string_dtype

st.set_page_config(
    page_title="DataFix Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RID_BASE = "__dfx_rid__"
MAX_ROWS_EXCEL = 250_000

SVG = {
    "shield": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7 3v5.6c0 4.3-2.9 8-7 9.4-4.1-1.4-7-5.1-7-9.4V6l7-3z"/><path d="M8.8 11.9l2.2 2.2 4.2-4.4"/></svg>',
    "rows": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="3"/><path d="M4 10h16M4 15h16"/></svg>',
    "columns": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="3"/><path d="M10 4v16M15 4v16"/></svg>',
    "missing": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8.2"/><path d="M6.4 6.4l11.2 11.2"/></svg>',
    "duplicates": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="8.5" y="8.5" width="11" height="11" rx="2.4"/><path d="M16 8.5V6.2A2.2 2.2 0 0 0 13.8 4H6.2A2.2 2.2 0 0 0 4 6.2v7.6A2.2 2.2 0 0 0 6.2 16h2.3"/></svg>',
    "swap": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8.5h13.5"/><path d="M14.5 5.5l3 3-3 3"/><path d="M20 15.5H6.5"/><path d="M9.5 12.5l-3 3 3 3"/></svg>',
    "lock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="10.5" width="14" height="9.5" rx="2.5"/><path d="M8.2 10.5V7.8a3.8 3.8 0 0 1 7.6 0v2.7"/></svg>',
    "check": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8.5"/><path d="M8.4 12.2l2.5 2.5 4.7-4.9"/></svg>',
    "upload": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V4"/><path d="M7.8 8.2L12 4l4.2 4.2"/><path d="M4.5 15.5v2.7A2.3 2.3 0 0 0 6.8 20.5h10.4a2.3 2.3 0 0 0 2.3-2.3v-2.7"/></svg>',
    "text": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 6.5h14M12 6.5V19"/><path d="M8.5 19h7"/></svg>',
}


def spark_svg(key: str) -> str:
    uid = f"spk-{key}"
    glow = (
        f'<filter id="{uid}-glow" x="-50%" y="-50%" width="200%" height="200%">'
        f'<feGaussianBlur stdDeviation="1.7" result="b"/>'
        f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
    )
    if key in ("rows", "columns"):
        c1, c2 = ("#22D3EE", "#8B5CF6") if key == "rows" else ("#A78BFA", "#F472B6")