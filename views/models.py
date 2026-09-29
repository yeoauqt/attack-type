import altair as alt
import streamlit as st

from de_pipeline import SETUP_NAMES, run_comparison
from ui import ensure_bundle, page_header


def render():
    b = ensure_bundle()
    page_header("Models", "Compare a trivial baseline, a linear model and a Random Forest on the same, "
                           "fully engineered feature set.")

    setup = st.selectbox("Feature setup used for this comparison", SETUP_NAMES, index=len(SETUP_NAMES) - 1)
    if st.button("Compare models", type="primary"):
        with st.spinner("Fitting Dummy, Logistic Regression and Random Forest..."):
            st.session_state["comparison_result"] = run_comparison(b, setup, st.session_state.get("n_estimators", 60))
            st.session_state["comparison_setup"] = setup

    result = st.session_state.get("comparison_result")
    if result is None:
        st.info("Click **Compare models** to fit all four models on the selected feature setup.")
        return

    st.caption(f"Setup used: {st.session_state.get('comparison_setup')}")
    st.dataframe(result.round(3), use_container_width=True)

    long = result.reset_index().melt(id_vars="setup", value_vars=["test_acc", "test_macro_f1"],
                                      var_name="metric", value_name="score")
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("setup:N", title=None, axis=alt.Axis(labelAngle=-15, labelLimit=220)),
            y=alt.Y("score:Q"),
            color=alt.Color("metric:N", scale=alt.Scale(range=["#F2894E", "#4C6EF5"]), title=None),
            xOffset="metric:N",
            tooltip=["setup", "metric", alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    best = result["test_macro_f1"].idxmax()
    st.markdown(
        f"**Best on macro F1:** {best}. Random Forest is generally preferred here because NSL-KDD features "
        f"mix very different scales (byte counts vs. rates in [0, 1]) and a tree-based model splits on raw "
        f"thresholds without needing that scale to be uniform."
    )
