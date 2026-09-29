import altair as alt
import pandas as pd
import streamlit as st

import de_pipeline as de
from ui import CLASS_HEX, class_bar_html, class_legend_html, ensure_bundle, kpi_card, page_header


def _class_card(b, label):
    tr, te = b["counts"].loc[label, "train"], b["counts"].loc[label, "test"]
    share = tr / b["counts"]["train"].sum() * 100
    r2l = b["counts"].loc["R2L", "train"]
    u2r = b["counts"].loc["U2R", "train"]
    dot = f'<span class="ng-dot" style="background:{CLASS_HEX[label]}"></span>'
    return kpi_card(
        title=f"{dot}{label}", tag="Training",
        value=f"{tr:,}", sub=f"Test: {te:,}",
        foot_html=f"Share of training set<br><b>{share:.1f}%</b>"
                  f"&nbsp;&nbsp;<span style='color:#9A9EB0'>R2L {r2l:,} · U2R {u2r:,}</span>",
    )


def render():
    b = ensure_bundle()
    page_header(
        "Dashboard",
        "Objective: classify network connections into five categories (Normal, DoS, Probe, R2L, U2R) "
        "and quantify the contribution of data engineering to model performance on the NSL-KDD benchmark.",
    )

    total = int(b["counts"]["train"].sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card(
                title="Dataset", tag="NSL-KDD",
                value=f"{total:,} rows",
                sub=class_bar_html(b["counts"]["train"]) + class_legend_html(),
                foot_html=f"Test rows (KDDTest+)<br><b>{int(b['counts']['test'].sum()):,}</b>"
                          f"&nbsp;&nbsp;<span class='ng-flag'>&minus;{b['n_dup_train']} duplicates in training set</span>",
            ),
            unsafe_allow_html=True,
        )
    for col, label in zip((c2, c3, c4), ("Normal", "DoS", "Probe")):
        with col:
            st.markdown(_class_card(b, label), unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    for col, label in zip((c5, c6), ("R2L", "U2R")):
        with col:
            st.markdown(_class_card(b, label), unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    top_l, top_r = st.columns([3, 2])
    with top_l:
        st.markdown("#### Data Flow")
    with top_r:
        metric = st.radio("Data flow metric", ["Features", "Training rows", "Test rows"],
                           horizontal=True, label_visibility="collapsed")

    key = {"Features": "n_features", "Training rows": "train_rows", "Test rows": "test_rows"}[metric]
    log = b["step_log"].copy()
    log["step_short"] = [s.split(" ", 1)[0] + ": " + s.split(" ", 1)[1][:22] for s in log["step"]]
    chart = (
        alt.Chart(log)
        .mark_line(point=alt.OverlayMarkDef(size=70, filled=True), color="#F2894E", strokeWidth=3)
        .encode(
            x=alt.X("step_short:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelLimit=220)),
            y=alt.Y(f"{key}:Q", title=metric),
            tooltip=["step", key],
        )
        .properties(height=300)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("<hr class='ng-sep'>", unsafe_allow_html=True)
    st.markdown("#### At a glance")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Raw features", 41)
    g2.metric("Features after cleaning", int(log["n_features"].iloc[2]))
    g3.metric("Constant columns dropped", len(b["const_cols"]))
    g4.metric("Correlated columns dropped", len(b["corr_drop"]))
    st.caption(
        f"Train file loaded from {b['train_src']} · Test file loaded from {b['test_src']} · "
        f"{len(b['unseen'])} attack sub-types appear only in the test set."
    )
