import altair as alt
import streamlit as st

from de_pipeline import quality_report
from ui import CLASS_HEX, class_bar_html, class_legend_html, ensure_bundle, kpi_card, page_header


def _class_card(b, label):
    tr, te = b["counts"].loc[label, "train"], b["counts"].loc[label, "test"]
    share = tr / b["counts"]["train"].sum() * 100
    r2l, u2r = b["counts"].loc["R2L", "train"], b["counts"].loc["U2R", "train"]
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
        "Upload & Data Quality",
        "Load KDDTrain+ / KDDTest+ from the sidebar (or use the auto-downloaded defaults), and review the "
        "profiling checks that decide every cleaning step used later in the pipeline.",
    )

    total = int(b["counts"]["train"].sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card(
                title="Dataset", tag="NSL-KDD", value=f"{total:,} rows",
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

    st.caption(
        f"Train file loaded from {b['train_src']} · Test file loaded from {b['test_src']} · "
        f"{len(b['unseen'])} attack sub-types appear only in the test set."
    )

    st.markdown("<hr class='ng-sep'>", unsafe_allow_html=True)
    st.markdown("#### Data quality findings")
    st.dataframe(quality_report(b), hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Most skewed numeric features (train)")
        skew_df = b["skew"].head(12).reset_index()
        skew_df.columns = ["feature", "abs_skew"]
        chart = (
            alt.Chart(skew_df)
            .mark_bar(color="#F2894E", cornerRadiusEnd=4)
            .encode(x=alt.X("abs_skew:Q", title="|skewness|"), y=alt.Y("feature:N", sort="-x", title=None))
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    with right:
        st.markdown("#### Correlation heatmap")
        corr = b["corr"]
        long = corr.stack().reset_index()
        long.columns = ["a", "b", "corr"]
        heat = (
            alt.Chart(long)
            .mark_rect()
            .encode(
                x=alt.X("a:N", title=None, axis=alt.Axis(labelAngle=-60, labelFontSize=8)),
                y=alt.Y("b:N", title=None, axis=alt.Axis(labelFontSize=8)),
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="oranges", domain=[0, 1]), title="|corr|"),
                tooltip=["a", "b", alt.Tooltip("corr:Q", format=".2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(heat, use_container_width=True)

    with st.expander(f"Pairs above the drop threshold (0.95) — {len(b['pairs'])} found"):
        if len(b["pairs"]):
            st.dataframe(b["pairs"].round(3), hide_index=True, use_container_width=True)
        else:
            st.info("No feature pair exceeds the 0.95 correlation threshold in the current training data.")
