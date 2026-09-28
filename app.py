"""
NSL-KDD Data Engineering Dashboard  (204426 - Semester 1/2026)
Streamlit application: presents the pipeline from attack_type_nslkdd_DE.ipynb  |  UI follows the 'Extej Wallet' design
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

import pipeline as P
from styles import CLASS_COLORS, CSS, LOGO_SVG, NAVY, ORANGE

st.set_page_config(page_title="NetGuard | NSL-KDD Data Engineering", page_icon=":material/shield:", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================ icons (inline SVG, replaces emoji)
_ICON_PATHS = {
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
    "mail": '<rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "file": '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "arrow": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
}


def icon(name: str, size: int = 16) -> str:
    """Return a single-line inline SVG (stroke = currentColor) for the given icon name."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align:-3px">{_ICON_PATHS[name]}</svg>')


def html(s: str):
    """Markdown treats lines indented by 4 spaces as code, so strip indentation and blank lines first."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


def fmt(n) -> str:
    return f"{int(n):,}"


# ============================================================ data (cached)
@st.cache_resource(show_spinner="Loading the NSL-KDD dataset and executing the data engineering pipeline ...")
def get_data(train_bytes=None, test_bytes=None, corr_threshold=0.95):
    import io
    tr_src = io.BytesIO(train_bytes) if train_bytes else None
    te_src = io.BytesIO(test_bytes) if test_bytes else None
    train, test = P.load_raw(tr_src, te_src)
    train, test = P.add_attack_type(train, test)
    de = P.run_de(train, test, corr_threshold)
    return train, test, de


@st.cache_data(show_spinner="Running ablation experiments (cross-validation). This may take a few minutes ...")
def get_experiments(n_est, _de, key):
    ab = P.run_ablation(_de, n_est=n_est)
    best = ab["cv_macro_f1"].idxmax()
    comp = P.run_comparison(_de, best, n_est=n_est)
    return ab, best, comp


@st.cache_resource(show_spinner="Training the final model ...")
def get_final(setup, n_est, _de, key):
    Xtr, ytr, Xte = _de["setups"][setup]
    model = P.make_model(Xtr, "rf", scale=False, n_est=n_est).fit(Xtr, ytr)
    return model, model.predict(Xte), model.predict_proba(Xte)


# ============================================================ sidebar
NAV = [":material/dashboard:  Dashboard",
       ":material/account_tree:  Pipeline",
       ":material/fact_check:  Data Quality",
       ":material/build:  Feature Engineering",
       ":material/science:  Ablation Study",
       ":material/model_training:  Models",
       ":material/troubleshoot:  Error Analysis",
       ":material/online_prediction:  Prediction"]

with st.sidebar:
    html(f'<div class="brand">{LOGO_SVG}<span class="name">NetGuard</span></div><div class="side-title">Pages</div>')
    page = st.radio("nav", NAV, label_visibility="collapsed", key="nav")
    st.markdown('<div class="side-title">Settings</div>', unsafe_allow_html=True)
    with st.expander(":material/tune:  Data & Model"):
        up_tr = st.file_uploader("KDDTrain+.txt (leave empty to load automatically)", type=["txt", "csv"])
        up_te = st.file_uploader("KDDTest+.txt", type=["txt", "csv"])
        corr_th = st.slider("Correlation threshold", 0.80, 0.99, 0.95, 0.01)
        n_est = st.select_slider("Random Forest: n_estimators", [20, 50, 100, 200], value=100)
    html(f"""<div class="side-title">Documentation & Support</div>
    <a class="side-link" href="https://github.com/defcom17/NSL_KDD" target="_blank">{icon('file')}&nbsp; NSL-KDD dataset</a>
    <a class="side-link" href="https://docs.streamlit.io" target="_blank">{icon('book')}&nbsp; Streamlit documentation</a>""")

# ------- load
try:
    train, test, de = get_data(up_tr.getvalue() if up_tr else None,
                               up_te.getvalue() if up_te else None, corr_th)
except Exception as e:  # no network connection or files missing
    st.error("Unable to load the dataset. Place `KDDTrain+.txt` and `KDDTest+.txt` in the `data/` folder, "
             "or upload them via Sidebar › Settings › Data & Model.")
    st.exception(e)
    st.stop()

data_key = f"{len(train)}-{len(test)}-{corr_th}"
if "exp" not in st.session_state or st.session_state.get("exp_key") != (data_key, n_est):
    st.session_state.pop("exp", None)

BEST_DEFAULT = "3 + engineered features"
best_setup = st.session_state["exp"][1] if "exp" in st.session_state else BEST_DEFAULT

# ------- topbar
html(f"""<div class="topbar"><div class="ico">{icon('search', 18)}</div><div class="search">Search</div>
<div class="ico">{icon('bell', 18)}</div><div class="ico">{icon('mail', 18)}</div><div class="avatar">DE</div>
<div class="who"><b>Data Engineer</b><span>204426 · Group Project</span></div><div class="toggle"></div><div class="ico">{icon('globe', 18)}</div></div>""")


def plotly_style(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins, Noto Sans Thai", color=NAVY),
                      legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f1f6")
    return fig


def run_experiments():
    st.session_state["exp"] = get_experiments(n_est, de, data_key)
    st.session_state["exp_key"] = (data_key, n_est)


# ============================================================ pages
def page_dashboard():
    st.markdown('<h1 class="page-title">Dashboard</h1>', unsafe_allow_html=True)
    st.caption("Objective: classify network connections into five categories (Normal, DoS, Probe, R2L, U2R) "
               "and quantify the contribution of data engineering to model performance on the NSL-KDD benchmark.")
    counts = de["counts"]
    total = counts["train"].sum()
    c0, c1, c2, c3 = st.columns([1.5, 1, 1, 1])

    with c0.container(key="card_dataset"):
        bar = "".join(f'<div style="width:{counts.loc[k, "train"] / total * 100:.2f}%;background:{CLASS_COLORS[k]}"></div>'
                      for k in P.LABELS)
        leg = "".join(f'<span><i class="dot" style="background:{CLASS_COLORS[k]}"></i>{k}</span>' for k in P.LABELS)
        n_dup = de["n_dup_train"]
        html(f"""<div class="card-head"><span style="display:inline-flex;align-items:center;gap:8px">{icon('database', 18)}Dataset</span><span class="sub">NSL-KDD ▾</span></div>
        <div class="big">{fmt(total)} rows</div>
        <div class="segbar">{bar}</div><div class="legend">{leg}</div>
        <div class="muted" style="margin-top:14px">Test rows (KDDTest+)</div>
        <div class="row-between"><b>{fmt(counts['test'].sum())}</b><span class="red">−{n_dup} duplicates in training set</span></div>""")

    for col, k in zip((c1, c2, c3), ("Normal", "DoS", "Probe")):
        with col.container(key=f"card_cls_{k}"):
            share = counts.loc[k, "train"] / total * 100
            html(f"""<div class="card-head"><span><i class="dot" style="background:{CLASS_COLORS[k]}"></i>{k}</span><span class="sub">Training ▾</span></div>
            <div class="big">{fmt(counts.loc[k, 'train'])}</div>
            <div class="muted">Test: {fmt(counts.loc[k, 'test'])}</div>
            <div class="muted" style="margin-top:14px">Share of training set</div>
            <div class="row-between"><b>{share:.1f}%</b><span class="green">R2L {fmt(counts.loc['R2L','train'])} · U2R {fmt(counts.loc['U2R','train'])}</span></div>""")

    with st.container(key="card_flow"):
        h1, h2 = st.columns([1, 1])
        h1.markdown('<div class="card-title">Data Flow</div>', unsafe_allow_html=True)
        with h2.container(key="pills"):
            metric = st.radio("metric", ["Features", "Training rows", "Test rows"], horizontal=True,
                              label_visibility="collapsed")
        sd = de["step_df"].reset_index()
        ycol = {"Features": "n_features", "Training rows": "train_rows", "Test rows": "test_rows"}[metric]
        fig = go.Figure(go.Scatter(x=sd["step"], y=sd[ycol], mode="lines+markers", line=dict(color=ORANGE, width=3, shape="spline"),
                                   marker=dict(size=9, color="#fff", line=dict(color=ORANGE, width=3)),
                                   fill="tozeroy", fillcolor="rgba(255,138,36,.13)"))
        lo, hi = sd[ycol].min(), sd[ycol].max()
        pad = max((hi - lo) * 0.5, 1)
        fig.update_yaxes(range=[max(lo - pad, 0), hi + pad])
        st.plotly_chart(plotly_style(fig, 300), key="flow_chart")
        last, first = sd[ycol].iloc[-1], sd[ycol].iloc[0]
        st.markdown(f"<span class='muted'>{metric}: {fmt(first)} → </span><b>{fmt(last)}</b> "
                    f"<span class='{'green' if last >= first else 'red'}'>({last - first:+,})</span>", unsafe_allow_html=True)

    with st.container(key="card_models"):
        h1, h2 = st.columns([4, 1])
        h1.markdown('<div class="card-title">Model Performance</div>', unsafe_allow_html=True)
        if h2.button("Run", help="Run the ablation study and model comparison (may take several minutes)"):
            run_experiments()
        comp = st.session_state["exp"][2] if "exp" in st.session_state else None
        cols = st.columns(3)
        items = [("Dummy (most frequent)", "BASELINE"), ("Logistic Regression + scaler", "LINEAR"),
                 ("Random Forest (no scaler)", "FINAL")]
        for col, (name, tag) in zip(cols, items):
            acc = f"{comp.loc[name, 'test_acc']:.3f}" if comp is not None else "—"
            f1 = f"{comp.loc[name, 'test_macro_f1']:.3f}" if comp is not None else "—"
            active = " active" if tag == "FINAL" else ""
            chk = '<div class="check">✓</div>' if tag == "FINAL" else ""
            col.markdown("".join(l.strip() for l in f"""<div class="mcard{active}">{chk}<div class="circles"><i class="c1"></i><i class="c2"></i></div>
            <div class="num">Accuracy {acc}</div><div class="foot"><div><small>Model</small>{name.split(' (')[0].split(' +')[0].upper()}</div>
            <div><small>Macro F1</small>{f1}</div><div>{tag}</div></div></div>""".splitlines()), unsafe_allow_html=True)
        if comp is None:
            st.caption("Click Run to compute model results (or open the Ablation Study or Models page).")


def page_pipeline():
    st.markdown('<h1 class="page-title">Pipeline</h1>', unsafe_allow_html=True)
    colors = ["#dbeafe", "#e0e7ff", "#fef9c3", "#fde68a", "#fed7aa", "#fecaca", "#e9d5ff", "#bbf7d0", "#dbeafe"]
    boxes = f'<div class="arrow">{icon("arrow", 20)}</div>'.join(
        f'<div class="box" style="background:{c}"><b>{t}</b><span>{d}</span></div>' for (t, d), c in zip(P.PIPELINE_STEPS, colors))
    with st.container(key="card_pipe"):
        st.markdown('<div class="card-title">Data Engineering Pipeline: Input → Process → Output</div>', unsafe_allow_html=True)
        html(f'<div class="flow">{boxes}</div>')
    with st.container(key="card_steplog"):
        st.markdown('<div class="card-title">Step Log (validation after each stage)</div>', unsafe_allow_html=True)
        st.dataframe(de["step_df"])
        st.caption("`validate()` verifies: no missing values · consistent X/y row counts · matching train/test schema · "
                   "categorical columns retained · labels restricted to the five defined classes.")
    with st.container(key="card_target"):
        st.markdown('<div class="card-title">Target Definition: Five-Class Attack Taxonomy</div>', unsafe_allow_html=True)
        rows = [("Normal", "normal"), ("DoS", ", ".join(P.DOS)), ("Probe", ", ".join(P.PROBE)),
                ("R2L", ", ".join(P.R2L)), ("U2R", ", ".join(P.U2R))]
        st.dataframe(pd.DataFrame(rows, columns=["Attack Type", "Label (subtype)"]))
    with st.container(key="card_raw"):
        st.markdown('<div class="card-title">Extraction: Raw Data Sample (KDDTrain+)</div>', unsafe_allow_html=True)
        st.dataframe(train.head(20))


def page_quality():
    st.markdown('<h1 class="page-title">Data Quality</h1>', unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("Missing values", de["n_missing"])
    m[1].metric("Duplicate records (train / test)", f"{de['n_dup_train']} / {de['n_dup_test']}")
    m[2].metric("Constant features", len(de["const_cols"]))
    m[3].metric("Test-only attack subtypes", len(de["unseen"]))
    with st.container(key="card_qr"):
        st.markdown('<div class="card-title">Data Quality Report (Issue → Resolution)</div>', unsafe_allow_html=True)
        st.dataframe(de["quality_report"])
    a, b = st.columns(2)
    with a.container(key="card_class"):
        st.markdown('<div class="card-title">Class Distribution (log scale)</div>', unsafe_allow_html=True)
        cdf = de["counts"].reset_index().melt(id_vars="Attack Type", var_name="set", value_name="count")
        fig = px.bar(cdf, x="Attack Type", y="count", color="set", barmode="group", log_y=True,
                     labels={"set": "Dataset", "count": "Count"},
                     color_discrete_map={"train": ORANGE, "test": "#5b8def"})
        st.plotly_chart(plotly_style(fig), key="class_chart")
    with b.container(key="card_skew"):
        st.markdown('<div class="card-title">Top 10 Most Skewed Features</div>', unsafe_allow_html=True)
        sk = de["skew"].head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=sk.values, y=sk.index, orientation="h", marker_color=ORANGE))
        st.plotly_chart(plotly_style(fig), key="skew_chart")
    a, b = st.columns([1, 1.2])
    with a.container(key="card_pairs"):
        st.markdown(f'<div class="card-title">Feature Pairs with Correlation > {corr_th}</div>', unsafe_allow_html=True)
        st.dataframe(de["corr_pairs"].round(3))
        st.markdown(f"**Removed:** `{', '.join(de['corr_drop']) or '-'}`")
    with b.container(key="card_heat"):
        st.markdown('<div class="card-title">Correlation Heatmap (training set, after deduplication)</div>', unsafe_allow_html=True)
        cm = de["corr_matrix"]
        fig = px.imshow(cm, color_continuous_scale=["#ffffff", "#ffc98f", ORANGE, "#c2410c"], zmin=0, zmax=1, aspect="auto")
        st.plotly_chart(plotly_style(fig, 420), key="heat_chart")
    with st.container(key="card_unseen"):
        st.markdown('<div class="card-title">Attack Subtypes Present Only in the Test Set</div>', unsafe_allow_html=True)
        st.write(f"{len(de['unseen'])} subtypes account for **{fmt(de['n_unseen_rows'])} / {fmt(len(test))}** test records")
        st.write(", ".join(de["unseen"]))


def page_features():
    st.markdown('<h1 class="page-title">Feature Engineering</h1>', unsafe_allow_html=True)
    spec = pd.DataFrame([
        ["bytes_ratio", "src_bytes / (dst_bytes + 1)", "Ratio of outbound to inbound data volume"],
        ["src_bytes_log", "log1p(src_bytes)", "Reduces skewness"],
        ["dst_bytes_log", "log1p(dst_bytes)", "Reduces skewness"],
        ["duration_log", "log1p(duration)", "Reduces skewness"],
        ["err_rate_mean", "mean(serror_rate, rerror_rate)", "Aggregated error rate"],
        ["srv_ratio", "srv_count / (count + 1)", "Proportion of connections to the same service"],
        ["host_srv_ratio", "dst_host_srv_count / (dst_host_count + 1)", "Host-level service proportion"],
        ["suspicious_cnt", "hot + failed_logins + compromised + ...", "Aggregated suspicious-behavior indicators"],
        ["service (grouped)", "Top 15 services in the training set; all others = 'other'", "Reduces one-hot encoding cardinality"],
    ], columns=["Feature", "Formula", "Purpose"])
    with st.container(key="card_fe"):
        st.markdown('<div class="card-title">Engineered Features (evaluated in the Ablation Study)</div>', unsafe_allow_html=True)
        st.dataframe(spec)
    _, _, Xte = de["setups"]["3 + engineered features"]
    Xtr = de["setups"]["3 + engineered features"][0]
    with st.container(key="card_fe2"):
        st.markdown('<div class="card-title">Sample Data after Feature Engineering</div>', unsafe_allow_html=True)
        new_cols = ["bytes_ratio", "src_bytes_log", "dst_bytes_log", "duration_log", "err_rate_mean",
                    "srv_ratio", "host_srv_ratio", "suspicious_cnt", "service"]
        st.dataframe(Xtr[new_cols].head(15))
        st.write(f"Top services (training set): `{', '.join(sorted(de['top_services']))}`")
        st.write(f"Features: **{Xtr.shape[1]}** (numeric {Xtr.shape[1] - 3} + categorical 3, one-hot encoded within the model pipeline)")


def page_ablation():
    st.markdown('<h1 class="page-title">Ablation Study</h1>', unsafe_allow_html=True)
    with st.container(key="card_ab"):
        h1, h2 = st.columns([4, 1])
        h1.markdown('<div class="card-title">Does Data Engineering Improve Performance? (Random Forest)</div>', unsafe_allow_html=True)
        if h2.button("Run"):
            run_experiments()
        st.caption("The best setup is selected by **cross-validated macro F1 on the training set**. "
                   "KDDTest+ results are reported for evaluation only, to prevent data leakage.")
        if "exp" not in st.session_state:
            st.info("Click Run to execute 3-fold cross-validation for all four setups.")
            return
        ab, best, _ = st.session_state["exp"]
        st.dataframe(ab)
        st.success(f"Best setup (CV macro F1 on training set): **{best}**")
        long = ab.reset_index().melt(id_vars="setup", value_vars=["cv_macro_f1", "test_macro_f1"], var_name="metric")
        fig = px.bar(long, x="setup", y="value", color="metric", barmode="group",
                     labels={"setup": "Setup", "value": "Macro F1", "metric": "Metric"},
                     color_discrete_map={"cv_macro_f1": ORANGE, "test_macro_f1": "#5b8def"})
        st.plotly_chart(plotly_style(fig, 360), key="ab_chart")
        st.caption("Cross-validation scores on the training set are typically higher than test scores because "
                   "training records are highly similar and the test set contains novel attack subtypes.")


def page_models():
    st.markdown('<h1 class="page-title">Models</h1>', unsafe_allow_html=True)
    if "exp" not in st.session_state:
        with st.container(key="card_m0"):
            st.info("Experiments have not been run yet. Run them to compute the ablation study and model comparison.")
            if st.button("Run Experiments"):
                run_experiments()
                st.rerun()
        return
    ab, best, comp = st.session_state["exp"]
    with st.container(key="card_m1"):
        st.markdown(f'<div class="card-title">Model Comparison (setup: {best})</div>', unsafe_allow_html=True)
        st.dataframe(comp)
        st.caption("StandardScaler is required only for Logistic Regression; Random Forest does not require feature scaling "
                   "(see the comparison table above).")
    model, pred, _ = get_final(best, n_est, de, data_key)
    y_test = de["y_test"]
    rep = pd.DataFrame(classification_report(y_test, pred, labels=P.LABELS, digits=3, zero_division=0, output_dict=True)).T
    a, b = st.columns(2)
    with a.container(key="card_rep"):
        st.markdown('<div class="card-title">Classification Report (KDDTest+)</div>', unsafe_allow_html=True)
        st.dataframe(rep.round(3))
    with b.container(key="card_cm"):
        st.markdown('<div class="card-title">Confusion Matrix (normalized by true class = recall)</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, pred, labels=P.LABELS, normalize="true")
        fig = px.imshow(cm, x=P.LABELS, y=P.LABELS, text_auto=".2f", color_continuous_scale=["#fff", ORANGE], zmin=0, zmax=1)
        fig.update_layout(xaxis_title="Predicted", yaxis_title="True")
        st.plotly_chart(plotly_style(fig, 360), key="cm_chart")
    with st.container(key="card_imp"):
        st.markdown('<div class="card-title">Top 15 Feature Importances</div>', unsafe_allow_html=True)
        names = model.named_steps["pre"].get_feature_names_out()
        imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False).head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ORANGE))
        st.plotly_chart(plotly_style(fig, 420), key="imp_chart")


def page_error():
    st.markdown('<h1 class="page-title">Error Analysis: R2L & U2R</h1>', unsafe_allow_html=True)
    if "exp" not in st.session_state:
        st.info("Showing results for the default setup. Open the Models page and click Run Experiments "
                "to use the best setup selected by cross-validation.")
    model, pred, _ = get_final(best_setup, n_est, de, data_key)
    sub, by_seen, r2l_pred = P.error_analysis(test, pred, de["unseen"])
    with st.container(key="card_e1"):
        st.markdown('<div class="card-title">Recall by Seen / Unseen Subtype</div>', unsafe_allow_html=True)
        st.dataframe(by_seen)
    with st.container(key="card_e2"):
        st.markdown('<div class="card-title">Recall by Attack Subtype (sorted by frequency)</div>', unsafe_allow_html=True)
        st.dataframe(sub.head(25))
    with st.container(key="card_e3"):
        st.markdown('<div class="card-title">Predicted Class of R2L Records</div>', unsafe_allow_html=True)
        fig = px.bar(r2l_pred.reset_index(), x="pred", y="count", color="pred", color_discrete_map=CLASS_COLORS,
                     labels={"pred": "Predicted class", "count": "Count"})
        st.plotly_chart(plotly_style(fig, 300), key="r2l_chart")


def page_predict():
    st.markdown('<h1 class="page-title">Prediction</h1>', unsafe_allow_html=True)
    model, pred, proba = get_final(best_setup, n_est, de, data_key)
    classes = list(model.named_steps["clf"].classes_)
    with st.container(key="card_p1"):
        st.markdown('<div class="card-title">Select a connection from KDDTest+ and view the model prediction</div>', unsafe_allow_html=True)
        idx = st.slider("Record index", 0, len(test) - 1, 0)
        true, p = test["Attack Type"].iloc[idx], pred[idx]
        ok = true == p
        c = st.columns(3)
        c[0].metric("Actual", f"{true}  ({test['label'].iloc[idx]})")
        c[1].metric("Predicted", p)
        c[2].metric("Result", "Correct" if ok else "Incorrect")
        fig = go.Figure(go.Bar(x=classes, y=proba[idx], marker_color=[CLASS_COLORS[k] for k in classes]))
        fig.update_yaxes(range=[0, 1], title="Probability")
        st.plotly_chart(plotly_style(fig, 300), key="pred_chart")
        with st.expander("Feature values for this record (after data engineering)"):
            Xte = de["setups"][best_setup][2]
            st.dataframe(Xte.iloc[[idx]].T)


PAGES = dict(zip(NAV, [page_dashboard, page_pipeline, page_quality, page_features,
                       page_ablation, page_models, page_error, page_predict]))
PAGES[page]()
