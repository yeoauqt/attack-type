import altair as alt
import pandas as pd
import streamlit as st

from de_pipeline import run_ablation
from ui import ensure_bundle, page_header


def render():
    b = ensure_bundle()
    page_header("Ablation Study", "Each data-engineering step is added one at a time and evaluated with the "
                                   "same Random Forest, so any change in score is caused by that step alone.")

    c1, c2 = st.columns([3, 1])
    with c1:
        cv = st.checkbox("Include 3-fold cross-validation on the training set (slower)", value=False)
    with c2:
        run = st.button("Run ablation study", type="primary", use_container_width=True)

    if run:
        with st.spinner("Training a Random Forest for every setup..."):
            st.session_state["ablation_result"] = run_ablation(b, st.session_state.get("n_estimators", 60), cv=cv)

    result = st.session_state.get("ablation_result")
    if result is None:
        st.info("Click **Run ablation study** to fit a Random Forest on each of the four setups below.")
        return

    st.dataframe(result.round(3), use_container_width=True)

    long = result.reset_index().melt(
        id_vars="setup",
        value_vars=[c for c in ["test_macro_f1", "R2L_recall", "U2R_recall"] if c in result.columns],
        var_name="metric", value_name="score",
    )
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("setup:N", title=None, sort=None, axis=alt.Axis(labelAngle=-15, labelLimit=200)),
            y=alt.Y("score:Q", title="score"),
            color=alt.Color("metric:N", scale=alt.Scale(range=["#F2894E", "#E0417B", "#8B5CF6"]), title=None),
            xOffset="metric:N",
            tooltip=["setup", "metric", alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    base, full = result["test_macro_f1"].iloc[0], result["test_macro_f1"].iloc[-1]
    delta = full - base
    direction = "improved" if delta > 0.005 else ("stayed about the same" if abs(delta) <= 0.005 else "dropped")
    st.markdown(
        f"**Reading the result:** macro F1 {direction} from **{base:.3f}** (raw features) to **{full:.3f}** "
        f"(fully engineered) — a difference of **{delta:+.3f}**. Random Forests split on raw thresholds, so "
        f"scaling and most log-transforms are not expected to help; the correlation filter mainly helps by "
        f"removing redundant, noisy splits rather than by adding information."
    )
