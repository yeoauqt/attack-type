import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report

from de_pipeline import LABELS, SETUP_NAMES, build_final, error_tables, run_comparison, transform_for_setup, validate_upload
from ui import checks_table_html, ensure_bundle, page_header


def _get_final(b, setup, n_est):
    key = (id(b), setup, n_est)
    if st.session_state.get("final_model_key") != key:
        with st.spinner("Fitting the Random Forest used for prediction and evaluation..."):
            st.session_state["final_model"] = build_final(b, setup, n_est)
            st.session_state["final_model_key"] = key
    return st.session_state["final_model"]


def render():
    b = ensure_bundle()
    page_header("Predict & Visualize", "Run the trained pipeline end to end on new traffic, then inspect "
                                        "where the final model succeeds and fails.")

    setup = st.selectbox("Feature setup used for prediction and evaluation", SETUP_NAMES, index=len(SETUP_NAMES) - 1)
    n_est = st.session_state.get("n_estimators", 60)
    final = _get_final(b, setup, n_est)
    model = final["model"]

    tab_file, tab_single = st.tabs(["Upload traffic file", "Single record"])

    with tab_file:
        up = st.file_uploader("CSV/TXT with 41 NSL-KDD columns (label and difficulty columns are optional)",
                               type=["csv", "txt"], key="predict_file")
        if up is not None:
            X, y_true, checks = validate_upload(up.getvalue(), b)
            st.markdown("#### Schema validation")
            st.markdown(checks_table_html(checks), unsafe_allow_html=True)

            if X is None:
                st.error("The file did not pass validation, so no prediction was run.")
            else:
                Xf, steps = transform_for_setup(X, b, setup)
                pred = model.predict(Xf)
                st.markdown("#### Pipeline steps applied to this file")
                st.write(" → ".join(f"{name} ({n} cols)" for name, n in steps) + f" → predict ({len(pred)} rows)")

                dist = pd.Series(pred).value_counts().reindex(LABELS).fillna(0).astype(int)
                dist.name, dist.index.name = "count", "class"
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.markdown("#### Predicted class distribution")
                    st.dataframe(dist, use_container_width=True)
                with c2:
                    chart = (alt.Chart(dist.reset_index()).mark_bar(color="#F2894E", cornerRadiusEnd=4)
                             .encode(x=alt.X("class:N", sort=LABELS, title=None), y=alt.Y("count:Q"))
                             .properties(height=240))
                    st.altair_chart(chart, use_container_width=True)

                if y_true is not None:
                    acc = accuracy_score(y_true, pred)
                    st.metric("Accuracy on this file (labels were provided)", f"{acc:.3f}")
                    rep = pd.DataFrame(classification_report(
                        y_true, pred, labels=LABELS, output_dict=True, zero_division=0)).T.round(3)
                    st.dataframe(rep, use_container_width=True)

    with tab_single:
        st.caption("Start from a real NSL-KDD test record, then adjust a few key fields.")
        if st.button("Pick a random test record"):
            row = b["X_test_raw"].sample(1).iloc[0]
            st.session_state["single_row"] = row
            st.session_state["single_true"] = b["y_test"].loc[row.name]

        row = st.session_state.get("single_row")
        if row is None:
            st.info("Click **Pick a random test record** to load a starting point.")
        else:
            services = sorted(b["cats"]["service"])
            with st.form("single_predict_form"):
                fc1, fc2, fc3, fc4 = st.columns(4)
                service = fc1.selectbox("service", services,
                                         index=services.index(row["service"]) if row["service"] in services else 0)
                src_bytes = fc2.number_input("src_bytes", min_value=0, value=int(row["src_bytes"]))
                count = fc3.number_input("count", min_value=0, value=int(row["count"]))
                serror_rate = fc4.number_input("serror_rate", min_value=0.0, max_value=1.0,
                                                value=float(row["serror_rate"]), step=0.01)
                submitted = st.form_submit_button("Predict", type="primary")

            if submitted:
                edited = row.copy()
                edited["service"], edited["src_bytes"] = service, src_bytes
                edited["count"], edited["serror_rate"] = count, serror_rate
                X = pd.DataFrame([edited])[list(row.index)]
                Xf, _ = transform_for_setup(X, b, setup)
                classes = list(model.classes_)
                proba = model.predict_proba(Xf)[0]
                pred = classes[int(proba.argmax())]

                r1, r2 = st.columns([1, 2])
                with r1:
                    st.metric("Predicted class", pred)
                    st.metric("True label (original record)", str(st.session_state["single_true"]))
                with r2:
                    proba_df = pd.DataFrame({"class": classes, "probability": proba})
                    chart = (alt.Chart(proba_df).mark_bar(color="#F2894E", cornerRadiusEnd=4)
                             .encode(x=alt.X("class:N", sort=LABELS, title=None),
                                     y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1])))
                             .properties(height=240))
                    st.altair_chart(chart, use_container_width=True)

    st.markdown("<hr class='ng-sep'>", unsafe_allow_html=True)
    st.markdown("#### Model evaluation on KDDTest+")
    et = error_tables(b, final["pred"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Train accuracy", f"{final['train_acc']:.3f}")
    m2.metric("Test accuracy", f"{final['test_acc']:.3f}")
    m3.metric("Test macro F1", f"{et['report'].loc['macro avg', 'f1-score']:.3f}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Confusion matrix (row-normalised)")
        cm = pd.DataFrame(et["cm_norm"], index=LABELS, columns=LABELS).reset_index().melt(
            id_vars="index", var_name="predicted", value_name="rate")
        cm.columns = ["actual", "predicted", "rate"]
        heat = (alt.Chart(cm).mark_rect()
                .encode(x=alt.X("predicted:N", sort=LABELS, title="Predicted"),
                        y=alt.Y("actual:N", sort=LABELS, title="Actual"),
                        color=alt.Color("rate:Q", scale=alt.Scale(scheme="oranges", domain=[0, 1]), title="Recall"),
                        tooltip=["actual", "predicted", alt.Tooltip("rate:Q", format=".2f")])
                .properties(height=300))
        text = heat.mark_text(baseline="middle").encode(
            text=alt.Text("rate:Q", format=".2f"),
            color=alt.condition("datum.rate > 0.5", alt.value("white"), alt.value("#232634")))
        st.altair_chart(heat + text, use_container_width=True)
    with right:
        st.markdown("##### Top feature importances")
        imp = final["importance"].reset_index()
        imp.columns = ["feature", "importance"]
        bar = (alt.Chart(imp).mark_bar(color="#8B5CF6", cornerRadiusEnd=4)
               .encode(x=alt.X("importance:Q"), y=alt.Y("feature:N", sort="-x", title=None))
               .properties(height=300))
        st.altair_chart(bar, use_container_width=True)

    st.markdown("##### Precision / recall / F1 by class")
    st.dataframe(et["report"], use_container_width=True)

    with st.expander("R2L / U2R: seen vs. unseen attack sub-types"):
        st.dataframe(et["by_seen"], hide_index=True, use_container_width=True)
        st.caption(
            f"{len(b['unseen'])} attack sub-types in KDDTest+ never appear in KDDTrain+ (e.g. httptunnel, "
            f"sqlattack, snmpguess). Recall on those rows is expected to be lower — the model has no training "
            f"signal for them, which is a known property of NSL-KDD rather than a bug in the pipeline."
        )
        st.dataframe(et["sub"], hide_index=True, use_container_width=True)

    with st.expander("Compare against simpler baselines (Dummy / Logistic Regression)"):
        if st.button("Run comparison"):
            with st.spinner("Fitting Dummy and Logistic Regression..."):
                st.session_state["comparison_result"] = run_comparison(b, setup, n_est)
        cmp_ = st.session_state.get("comparison_result")
        if cmp_ is not None:
            st.dataframe(cmp_.round(3), use_container_width=True)
