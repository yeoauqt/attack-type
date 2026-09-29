import altair as alt
import pandas as pd
import streamlit as st

from de_pipeline import LABELS, SETUP_NAMES, build_final, error_tables
from ui import ensure_bundle, page_header


def _build_final_if_needed(b, setup, n_est):
    key = (id(b), setup, n_est)
    if st.session_state.get("final_model_key") != key:
        with st.spinner("Fitting the final Random Forest..."):
            st.session_state["final_model"] = build_final(b, setup, n_est)
            st.session_state["final_model_key"] = key
    return st.session_state["final_model"]


def render():
    b = ensure_bundle()
    page_header("Error Analysis", "Where the final model succeeds and fails, with a focus on the rare "
                                   "R2L and U2R classes that matter most for security.")

    setup = st.selectbox("Feature setup for the final model", SETUP_NAMES, index=len(SETUP_NAMES) - 1)
    final = _build_final_if_needed(b, setup, st.session_state.get("n_estimators", 60))
    et = error_tables(b, final["pred"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Train accuracy", f"{final['train_acc']:.3f}")
    c2.metric("Test accuracy", f"{final['test_acc']:.3f}")
    c3.metric("Test macro F1", f"{et['report'].loc['macro avg', 'f1-score']:.3f}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Confusion matrix (row-normalised)")
        cm = pd.DataFrame(et["cm_norm"], index=LABELS, columns=LABELS).reset_index().melt(
            id_vars="index", var_name="predicted", value_name="rate")
        cm.columns = ["actual", "predicted", "rate"]
        heat = (
            alt.Chart(cm)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", sort=LABELS, title="Predicted"),
                y=alt.Y("actual:N", sort=LABELS, title="Actual"),
                color=alt.Color("rate:Q", scale=alt.Scale(scheme="oranges", domain=[0, 1]), title="Recall"),
                tooltip=["actual", "predicted", alt.Tooltip("rate:Q", format=".2f")],
            )
            .properties(height=320)
        )
        text = heat.mark_text(baseline="middle").encode(
            text=alt.Text("rate:Q", format=".2f"),
            color=alt.condition("datum.rate > 0.5", alt.value("white"), alt.value("#232634")),
        )
        st.altair_chart(heat + text, use_container_width=True)

    with right:
        st.markdown("#### Precision / recall / F1")
        st.dataframe(et["report"], use_container_width=True)

    st.markdown("<hr class='ng-sep'>", unsafe_allow_html=True)
    st.markdown("#### R2L and U2R: seen vs. unseen attack sub-types")
    st.dataframe(et["by_seen"], hide_index=True, use_container_width=True)
    st.caption(
        f"{len(b['unseen'])} attack sub-types in KDDTest+ never appear in KDDTrain+ (e.g. httptunnel, "
        f"sqlattack, snmpguess). Recall on those rows is expected to be lower — the model has no training "
        f"signal for them, which is a known property of NSL-KDD rather than a bug in the pipeline."
    )
    with st.expander("Breakdown by exact attack sub-type"):
        st.dataframe(et["sub"], hide_index=True, use_container_width=True)
    with st.expander("What R2L rows get misclassified as"):
        st.bar_chart(et["r2l_as"])
