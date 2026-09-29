import altair as alt
import pandas as pd
import streamlit as st

from de_pipeline import quality_report
from ui import ensure_bundle, page_header


def render():
    b = ensure_bundle()
    page_header("Data Quality", "Findings from profiling the training set, and the action taken for each issue.")

    st.dataframe(quality_report(b), hide_index=True, use_container_width=True)

    st.markdown("<hr class='ng-sep'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Most skewed numeric features (train)")
        skew_df = b["skew"].head(12).reset_index()
        skew_df.columns = ["feature", "abs_skew"]
        chart = (
            alt.Chart(skew_df)
            .mark_bar(color="#F2894E", cornerRadiusEnd=4)
            .encode(x=alt.X("abs_skew:Q", title="|skewness|"),
                    y=alt.Y("feature:N", sort="-x", title=None))
            .properties(height=340)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("Right-skewed byte/duration counters are the reason src_bytes_log, dst_bytes_log and "
                   "duration_log are tested in the ablation study.")

    with c2:
        st.markdown(f"#### Correlation heatmap (|corr| > 0.6 highlighted)")
        corr = b["corr"]
        long = corr.where(~corr.isna()).stack().reset_index()
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
            .properties(height=340)
        )
        st.altair_chart(heat, use_container_width=True)

    st.markdown("#### Pairs above the drop threshold (0.95)")
    if len(b["pairs"]):
        st.dataframe(b["pairs"].round(3), hide_index=True, use_container_width=True)
    else:
        st.info("No feature pair exceeds the 0.95 correlation threshold in the current training data.")
