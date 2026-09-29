import altair as alt
import streamlit as st

from de_pipeline import ENGINEERED_TABLE, run_ablation
from ui import ensure_bundle, page_header

STEP_INFO = [
    ("1. Dedup + drop constant", "Remove exact-duplicate training rows and columns with a single "
                                   "constant value (no information for any model)."),
    ("2. Drop correlated (>0.95)", "Compute correlation on the training set only, then drop one column "
                                     "of every pair above the threshold to avoid redundant, collinear inputs."),
    ("3. Feature engineering", "Add ratios and log-transforms, aggregate suspicious-activity counts, and "
                                 "group rare services into 'other' to control one-hot width."),
]


def render():
    b = ensure_bundle()
    page_header("Pipeline Explorer", "Before-and-after view of every cleaning step, fitted on the training "
                                      "set only and re-applied identically to test data and new traffic.")

    log = b["step_log"].rename(columns={
        "step": "Step", "train_rows": "Train rows", "test_rows": "Test rows", "n_features": "Features",
    })
    cols = st.columns(len(STEP_INFO))
    for col, (title, desc) in zip(cols, STEP_INFO):
        with col:
            st.markdown(
                f"<div class='ng-card' style='min-height:150px'>"
                f"<div class='ng-card-title'>{title}</div>"
                f"<div class='ng-card-sub' style='margin-top:8px; line-height:1.5;'>{desc}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Row and feature counts at every step")
        st.dataframe(log, hide_index=True, use_container_width=True)
        st.download_button("Download step log (CSV)", log.to_csv(index=False).encode(),
                            file_name="pipeline_step_log.csv", mime="text/csv")
    with right:
        chart = (
            alt.Chart(b["step_log"])
            .mark_line(point=alt.OverlayMarkDef(size=60, filled=True), color="#F2894E", strokeWidth=3)
            .encode(x=alt.X("step:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelLimit=0, labels=False)),
                    y=alt.Y("n_features:Q", title="Features remaining"), tooltip=["step", "n_features"])
            .properties(height=260, title="Feature count across steps")
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("<hr class='ng-sep'>", unsafe_allow_html=True)
    st.markdown("#### Engineered features")
    c1, c2 = st.columns(2)
    c1.metric("Features before engineering", b["X_train_sel"].shape[1])
    c2.metric("Features after engineering", b["X_train_fe"].shape[1])
    st.dataframe(ENGINEERED_TABLE, hide_index=True, use_container_width=True)

    st.markdown("<hr class='ng-sep'>", unsafe_allow_html=True)
    st.markdown("#### Ablation study — does each step actually help?")
    c1, c2 = st.columns([3, 1])
    with c1:
        cv = st.checkbox("Include 3-fold cross-validation (slower)", value=False)
    with c2:
        run = st.button("Run ablation study", type="primary", use_container_width=True)
    if run:
        with st.spinner("Training a Random Forest for every setup..."):
            st.session_state["ablation_result"] = run_ablation(b, st.session_state.get("n_estimators", 60), cv=cv)

    result = st.session_state.get("ablation_result")
    if result is None:
        st.info("Click **Run ablation study** to fit a Random Forest on each of the four setups above.")
        return

    st.dataframe(result.round(3), use_container_width=True)
    long = result.reset_index().melt(
        id_vars="setup", value_vars=[c for c in ["test_macro_f1", "R2L_recall", "U2R_recall"] if c in result.columns],
        var_name="metric", value_name="score",
    )
    bar = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("setup:N", title=None, sort=None, axis=alt.Axis(labelAngle=-15, labelLimit=200)),
            y=alt.Y("score:Q"),
            color=alt.Color("metric:N", scale=alt.Scale(range=["#F2894E", "#E0417B", "#8B5CF6"]), title=None),
            xOffset="metric:N", tooltip=["setup", "metric", alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=320)
    )
    st.altair_chart(bar, use_container_width=True)

    base, full = result["test_macro_f1"].iloc[0], result["test_macro_f1"].iloc[-1]
    delta = full - base
    direction = "improved" if delta > 0.005 else ("stayed about the same" if abs(delta) <= 0.005 else "dropped")
    st.markdown(
        f"**Reading the result:** macro F1 {direction} from **{base:.3f}** (raw features) to **{full:.3f}** "
        f"(fully engineered), a difference of **{delta:+.3f}**. Random Forests split on raw thresholds, so "
        f"scaling and most log-transforms are not expected to help; the correlation filter mainly helps by "
        f"removing redundant, noisy splits rather than by adding information."
    )
