"""Shared chrome and small UI helpers for every page of the NetGuard app."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import de_pipeline as de

ASSETS = Path(__file__).parent / "assets"

CLASS_DOT = {"Normal": "dot-normal", "DoS": "dot-dos", "Probe": "dot-probe", "R2L": "dot-r2l", "U2R": "dot-u2r"}
CLASS_HEX = {"Normal": "#F2894E", "DoS": "#4C6EF5", "Probe": "#2BB673", "R2L": "#E0417B", "U2R": "#8B5CF6"}


def inject_css() -> None:
    st.markdown(f"<style>{(ASSETS / 'style.css').read_text()}</style>", unsafe_allow_html=True)


def render_logo() -> None:
    st.sidebar.markdown(
        """
        <div class="ng-logo">
            <div class="ng-logo-badge">NG</div>
            <div class="ng-logo-text">NetGuard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_settings_panel() -> None:
    st.sidebar.markdown('<div class="ng-section-label">SETTINGS</div>', unsafe_allow_html=True)
    with st.sidebar.expander("Data & Model", expanded=True):
        st.file_uploader("KDDTrain+.txt (leave empty to load automatically)",
                          type=["txt", "csv"], key="train_file")
        st.file_uploader("KDDTest+.txt (leave empty to load automatically)",
                          type=["txt", "csv"], key="test_file")
        st.slider("Random Forest trees", 20, 200, 60, 10, key="n_estimators")
        st.caption("200 MB per file · TXT, CSV")


def ensure_bundle() -> dict:
    """(Re)build the data-engineering bundle whenever the uploaded files change."""
    train_file = st.session_state.get("train_file")
    test_file = st.session_state.get("test_file")
    sig = (
        train_file.name if train_file else None, train_file.size if train_file else None,
        test_file.name if test_file else None, test_file.size if test_file else None,
    )
    if st.session_state.get("bundle_sig") != sig or "bundle" not in st.session_state:
        with st.spinner("Loading NSL-KDD data and running the cleaning pipeline..."):
            try:
                train_bytes = train_file.getvalue() if train_file else None
                test_bytes = test_file.getvalue() if test_file else None
                train_df, train_src = de.load_one(train_bytes, "KDDTrain+.txt")
                test_df, test_src = de.load_one(test_bytes, "KDDTest+.txt")
                bundle = de.prepare(train_df, test_df)
                bundle["train_src"], bundle["test_src"] = train_src, test_src
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not load the dataset: {exc}")
                st.stop()
        st.session_state["bundle"] = bundle
        st.session_state["bundle_sig"] = sig
        # invalidate anything computed from the previous bundle
        for k in ("ablation_result", "comparison_result", "final_model"):
            st.session_state.pop(k, None)
    return st.session_state["bundle"]


_ICON_SEARCH = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/>'
    '<line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'
)
_ICON_BELL = (
    '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M18 8a6 6 0 1 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/>'
    '<path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>'
)
_ICON_MAIL = (
    '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m2 7 10 6 10-6"/></svg>'
)


def render_topbar(subtitle: str = "204426 · Group Project") -> None:
    st.markdown(
        f"""
        <div class="ng-topbar">
            <div class="ng-search">{_ICON_SEARCH}&nbsp; Search</div>
            <div class="ng-topbar-icons">{_ICON_BELL}{_ICON_MAIL}</div>
            <div class="ng-avatar">DE</div>
            <div>
                <div class="ng-user-name">Data Engineer</div>
                <div class="ng-user-sub">{subtitle}</div>
            </div>
            <div class="ng-toggle"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, lede: str) -> None:
    st.markdown(f'<div class="ng-h1">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ng-lede">{lede}</div>', unsafe_allow_html=True)


def kpi_card(title: str, value: str, sub: str = "", foot_html: str = "", tag: str = "") -> str:
    return f"""
    <div class="ng-card">
        <div class="ng-card-head">
            <div class="ng-card-title">{title}</div>
            <div class="ng-card-tag">{tag}</div>
        </div>
        <div class="ng-card-value">{value}</div>
        <div class="ng-card-sub">{sub}</div>
        {f'<div class="ng-card-foot">{foot_html}</div>' if foot_html else ''}
    </div>
    """


def class_bar_html(counts: pd.Series) -> str:
    total = counts.sum()
    segs = "".join(
        f'<div style="width:{counts[c] / total * 100:.2f}%; background:{CLASS_HEX[c]};"></div>'
        for c in de.LABELS if counts[c] > 0
    )
    return f'<div class="ng-bar-track">{segs}</div>'


def class_legend_html(labels=de.LABELS) -> str:
    spans = "".join(f'<span><span class="ng-dot {CLASS_DOT[c]}"></span>{c}</span>' for c in labels)
    return f'<div class="ng-legend">{spans}</div>'


def badge(status: str) -> str:
    cls = {"PASS": "badge-pass", "WARN": "badge-warn", "FAIL": "badge-fail"}[status]
    return f'<span class="ng-badge {cls}">{status}</span>'


def checks_table_html(checks: list[dict]) -> str:
    rows = "".join(
        f"<tr><td style='padding:8px 10px;font-weight:600;color:#232634;'>{c['Check']}</td>"
        f"<td style='padding:8px 10px;'>{badge(c['Status'])}</td>"
        f"<td style='padding:8px 10px;color:#6B6F80;'>{c['Detail']}</td></tr>"
        for c in checks
    )
    return (
        "<table style='width:100%; border-collapse:collapse; font-size:13.5px;'>"
        f"{rows}</table>"
    )
