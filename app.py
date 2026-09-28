"""
NSL-KDD Data Engineering Dashboard
Course 204426 | Semester 1/2569 | Group Project

Streamlit application built on the pipeline in attack_type_nslkdd_DE.ipynb.
Run:  streamlit run app.py
"""
import io
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

import pipeline as P
from styles import ACCENT, CLASS_COLORS, CSS, GRID, LOGO_SVG, MUTED_BLUE, NAVY

# ============================================================ project information
# Edit these values before submission (group name in English; student ID + name).
PROJECT_TITLE = "Data Engineering Pipeline for Network Intrusion Data (NSL-KDD)"
GROUP_NAME = "Group Name"
MEMBERS = [
    ("Student ID 1", "Member Name 1"),
    ("Student ID 2", "Member Name 2"),
    ("Student ID 3", "Member Name 3"),
]
CAT = ["protocol_type", "service", "flag"]
BEST_DEFAULT = "3 + engineered features"

st.set_page_config(page_title="NSL-KDD Data Engineering Dashboard", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================ helpers
def html(s: str):
    """Render an HTML block (indentation and blank lines are removed so Markdown does not treat it as code)."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


def fmt(n) -> str:
    return f"{int(n):,}"


def header(title: str, eyebrow: str, sub: str = ""):
    html(f'<div class="eyebrow">{eyebrow}</div>')
    st.markdown(f'<h1 class="page-title">{title}</h1>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<p class="page-sub">{sub}</p>', unsafe_allow_html=True)


def card_title(text: str):
    st.markdown(f'<div class="card-title">{text}</div>', unsafe_allow_html=True)


def table(headers, rows) -> str:
    """Formal HTML table (all cell text is escaped)."""
    th = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="ftable"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def bullets(items) -> str:
    return '<ul class="plain">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def plotly_style(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Noto Sans Thai, sans-serif", color=NAVY, size=12),
                      legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


HEAT_SCALE = ["#FFFFFF", "#C9D6E6", ACCENT, NAVY]


# ============================================================ data (cached)
@st.cache_resource(show_spinner="Loading NSL-KDD and executing the data engineering pipeline ...")
def get_data(train_bytes=None, test_bytes=None, corr_threshold=0.95):
    tr_src = io.BytesIO(train_bytes) if train_bytes else None
    te_src = io.BytesIO(test_bytes) if test_bytes else None
    train, test = P.load_raw(tr_src, te_src)
    train, test = P.add_attack_type(train, test)
    de = P.run_de(train, test, corr_threshold)
    return train, test, de


@st.cache_data(show_spinner="Running ablation with cross-validation. This may take several minutes ...")
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
NAV = ["Introduction", "Problem Analysis", "Tools and Techniques", "Data Quality", "Feature Engineering",
       "Ablation Study", "Model Evaluation", "Error Analysis", "Demonstration", "Conclusion"]

with st.sidebar:
    html(f"""<div class="brand">{LOGO_SVG}<div class="name">NSL-KDD<span class="tag">Data Engineering</span></div></div>
    <div class="side-title">Navigation</div>""")
    page = st.radio("Navigation", NAV, label_visibility="collapsed", key="nav")
    st.markdown('<div class="side-title">Configuration</div>', unsafe_allow_html=True)
    with st.expander("Data and Model Settings"):
        up_tr = st.file_uploader("KDDTrain+.txt (optional; downloaded automatically if omitted)", type=["txt", "csv"])
        up_te = st.file_uploader("KDDTest+.txt (optional)", type=["txt", "csv"])
        corr_th = st.slider("Correlation threshold", 0.80, 0.99, 0.95, 0.01)
        n_est = st.select_slider("Random Forest: number of trees", [20, 50, 100, 200], value=100)
    html("""<div class="side-title">References</div>
    <a class="side-link" href="https://github.com/defcom17/NSL_KDD" target="_blank">NSL-KDD dataset repository</a>
    <a class="side-link" href="https://docs.streamlit.io" target="_blank">Streamlit documentation</a>""")

# ------- load
try:
    train, test, de = get_data(up_tr.getvalue() if up_tr else None,
                               up_te.getvalue() if up_te else None, corr_th)
except Exception as e:  # no network or missing files
    st.error("The dataset could not be loaded. Place `KDDTrain+.txt` and `KDDTest+.txt` in the `data/` folder, "
             "or upload them under Configuration > Data and Model Settings.")
    st.exception(e)
    st.stop()

data_key = f"{len(train)}-{len(test)}-{corr_th}"
if st.session_state.get("exp_key") != (data_key, n_est):
    st.session_state.pop("exp", None)

best_setup = st.session_state["exp"][1] if "exp" in st.session_state else BEST_DEFAULT

html(f"""<div class="topbar"><div><b>204426</b> &nbsp;|&nbsp; Data Engineering Group Project</div>
<div>{escape(GROUP_NAME)} &nbsp;|&nbsp; Semester 1/2569</div></div>""")


def run_experiments():
    st.session_state["exp"] = get_experiments(n_est, de, data_key)
    st.session_state["exp_key"] = (data_key, n_est)


# ============================================================ static content
PIPELINE_STEPS = [
    ("Input", "NSL-KDD", "KDDTrain+ and KDDTest+"),
    ("1  Extraction", "Load files and apply", "the 43-column schema"),
    ("2  Quality Check", "Missing, duplicate,", "constant, skewness"),
    ("3  Cleaning", "Remove duplicates and", "constant features"),
    ("4  Feature Selection", "Correlation filter on", "training data"),
    ("5  Feature Engineering", "Ratios, log transforms,", "category grouping"),
    ("6  Preprocessing", "One-hot encoding,", "scaling for LR only"),
    ("7  Validation", "Assertions and step log", "after every stage"),
    ("Output", "Clean data, model,", "and evaluation"),
]

IPO_ROWS = [
    ("Extraction", "KDDTrain+.txt and KDDTest+.txt (headerless CSV files)",
     "Read with an explicit 43-column schema; assert column count and train/test schema equality",
     "Raw training and test tables"),
    ("Target construction", "Attack subtype (label)",
     "Map each subtype to one of five categories: Normal, DoS, Probe, R2L, U2R",
     "Attack Type (classification target)"),
    ("Quality assessment", "Raw features",
     "Count missing values, duplicates and constant columns; compute skewness and class balance",
     "Data Quality Report"),
    ("Cleaning", "Raw training and test tables",
     "Remove duplicate records from the training set; drop constant columns from both sets",
     "Cleaned tables"),
    ("Feature selection", "Cleaned training features",
     "Compute the absolute correlation matrix on training data; drop one feature from each pair above the threshold",
     "Reduced feature set"),
    ("Feature engineering", "Reduced feature set",
     "Create ratio, log1p, mean and count features; group rare service values into 'other'",
     "Engineered feature set"),
    ("Preprocessing", "Engineered feature set",
     "One-hot encode categorical columns; standardise numeric columns for Logistic Regression only",
     "Model-ready matrices"),
    ("Validation", "Output of every stage",
     "Assert no missing values, aligned X/y lengths, identical schema, categorical columns present, valid labels",
     "Step log (rows and features per stage)"),
    ("Evaluation", "Model-ready matrices",
     "Ablation with stratified 3-fold cross-validation; compare Dummy, Logistic Regression, Random Forest",
     "Metrics, confusion matrix, error analysis"),
]

FEATURE_GROUPS = [
    ("Basic connection", "9", "duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent"),
    ("Content", "13", "hot, num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_root, "
                      "num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_host_login, is_guest_login"),
    ("Time-based traffic", "9", "count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, "
                                "same_srv_rate, diff_srv_rate, srv_diff_host_rate"),
    ("Host-based traffic", "10", "dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, "
                                 "dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, "
                                 "dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate"),
    ("Label and metadata", "2", "label (attack subtype), difficulty (difficulty score)"),
]

ATTACK_DESC = {
    "Normal": "Legitimate network connection.",
    "DoS": "Denial of service: a resource is overwhelmed so that legitimate users cannot access it.",
    "Probe": "Surveillance and scanning of hosts or ports to discover vulnerabilities.",
    "R2L": "Remote to local: unauthorised access to a machine from a remote host without an account.",
    "U2R": "User to root: a local user gains unauthorised superuser privileges.",
}

TECHNIQUES = [
    dict(no="T1", name="Data Extraction with Schema Enforcement", cat="Data Extraction", tool="pandas",
         objective="Load raw connection records reliably and guarantee that training and test data share one schema.",
         inp="KDDTrain+.txt and KDDTest+.txt, headerless CSV files.",
         process="Read with an explicit list of 43 column names; assert the column count and that both tables have identical columns.",
         out="Two raw DataFrames with named columns."),
    dict(no="T2", name="Data Profiling and Quality Assessment", cat="Data Quality", tool="pandas, NumPy",
         objective="Identify data problems before transforming, so that every later step is justified by evidence.",
         inp="Raw training and test features.",
         process="Count missing values, duplicates and constant columns; compute skewness of numeric features; measure class imbalance and subtypes unseen in training.",
         out="Data Quality Report (problem, result, action)."),
    dict(no="T3", name="Data Cleaning", cat="Data Cleaning", tool="pandas",
         objective="Remove records and columns that add no information and can bias a model.",
         inp="Raw tables and the profiling results.",
         process="Drop duplicate rows from the training set (test set kept as published); drop constant columns from both sets.",
         out="Cleaned training and test tables."),
    dict(no="T4", name="Feature Selection by Correlation Filtering", cat="Data Reduction", tool="pandas, NumPy",
         objective="Eliminate redundant features that carry the same information and encourage overfitting.",
         inp="Cleaned training features (numeric).",
         process="Compute the absolute Pearson correlation matrix on training data only; for each pair above the threshold drop one feature.",
         out="Reduced feature set and the list of correlated pairs."),
    dict(no="T5", name="Data Transformation and Feature Engineering", cat="Data Transformation", tool="pandas, NumPy",
         objective="Reduce skewness and expose domain relationships, tested experimentally rather than assumed to help.",
         inp="Reduced feature set.",
         process="Apply log1p to heavily skewed columns; build ratio, mean and count features; group rare service values into 'other'.",
         out="Engineered feature set."),
    dict(no="T6", name="Encoding and Scaling in a Model Pipeline", cat="Data Preprocessing", tool="scikit-learn",
         objective="Convert features to a numeric form and apply scaling only where the algorithm requires it.",
         inp="Engineered feature set.",
         process="ColumnTransformer with one-hot encoding (unknown categories ignored) for categorical columns; StandardScaler for Logistic Regression only.",
         out="Model-ready matrices."),
    dict(no="T7", name="Data Validation and Step Logging", cat="Data Validation", tool="Python assertions",
         objective="Detect silent errors early and document how each stage changes the data.",
         inp="Data produced by every stage.",
         process="Assert no missing values, aligned row counts, identical schemas, presence of categorical columns and valid labels; record rows and features.",
         out="Step log table."),
    dict(no="T8", name="Ablation Study with Cross-Validation", cat="Experiment Design", tool="scikit-learn",
         objective="Provide evidence of which data engineering steps improve the downstream model, without test-set leakage.",
         inp="Four cumulative setups (raw, cleaned, selected, engineered).",
         process="Stratified 3-fold cross-validated macro F1 on training data selects the best setup; KDDTest+ is used for reporting only.",
         out="Ablation table, selected setup, model comparison."),
]


# ============================================================ pages
def page_intro():
    header("Data Engineering Pipeline for Network Intrusion Data", "Chapter 1  |  Introduction",
           "A reproducible pipeline that prepares the NSL-KDD benchmark for the classification of network attacks.")

    with st.container(key="card_i1"):
        card_title("Project Information")
        html(table(["Item", "Detail"], [("Group name", GROUP_NAME), ("Project title", PROJECT_TITLE),
                                        ("Course", "204426, Semester 1/2569")]))
        st.markdown("**Project members**")
        html(table(["Student ID", "Name"], MEMBERS))

    a, b = st.columns(2)
    with a.container(key="card_i2"):
        card_title("Background and Motivation")
        st.markdown(
            "Intrusion detection systems increasingly rely on machine learning models trained on network connection records. "
            "The quality of the input data limits the quality of any such model. The NSL-KDD benchmark, although an improved "
            "version of KDD Cup 1999, still exhibits duplicated records, constant attributes, redundant correlated attributes, "
            "heavily skewed numeric distributions, severe class imbalance, and attack subtypes that appear only in the test set.")
    with b.container(key="card_i3"):
        card_title("Objectives")
        html(bullets([
            "Design and implement a data engineering pipeline covering extraction, quality assessment, cleaning, "
            "feature selection, transformation and validation.",
            "Justify every transformation by an observed data problem, supported by an ablation study that uses "
            "cross-validation on training data only.",
            "Demonstrate the pipeline output through a downstream classifier for five categories: Normal, DoS, Probe, R2L and U2R.",
            "Provide an interactive dashboard that documents each stage of the pipeline.",
        ]))

    counts = de["counts"]
    cols = st.columns(4)
    stats = [("Training records", fmt(counts["train"].sum()), "KDDTrain+ (before cleaning)"),
             ("Test records", fmt(counts["test"].sum()), "KDDTest+ (kept as published)"),
             ("Duplicate training records", fmt(de["n_dup_train"]), "Removed during cleaning"),
             ("Attack categories", "5", "Normal, DoS, Probe, R2L, U2R")]
    for i, (col, (label, value, foot)) in enumerate(zip(cols, stats)):
        with col.container(key=f"card_stat_{i}"):
            html(f'<div class="stat-label">{label}</div><div class="stat-value">{value}</div><div class="stat-foot">{foot}</div>')

    with st.container(key="card_i4"):
        card_title("Scope and Principles")
        html(bullets([
            "All data engineering decisions are derived from the training set only, to prevent data leakage.",
            "KDDTest+ is used for reporting results and is never used to select a setup.",
            "The classifier is a downstream application that measures the effect of data engineering; "
            "the focus of the project is the pipeline.",
        ]))


def pipeline_svg() -> str:
    bw, bh, gx, gy = 190, 92, 37, 50
    width = 5 * bw + 4 * gx
    height = 2 * bh + gy + 6
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" '
             f'font-family="Inter, Noto Sans Thai, sans-serif">',
             '<defs><marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0 0 L10 5 L0 10 z" fill="#6B7280"/></marker></defs>']
    pos = []
    for i in range(len(PIPELINE_STEPS)):
        r, c = divmod(i, 5)
        pos.append((c * (bw + gx), r * (bh + gy) + 2))
    for i, (title, l1, l2) in enumerate(PIPELINE_STEPS):
        x, y = pos[i]
        terminal = i in (0, len(PIPELINE_STEPS) - 1)
        fill, stroke = (NAVY, NAVY) if terminal else ("#FFFFFF", "#C4CEDC")
        t_col, d_col = ("#FFFFFF", "#C9D3E3") if terminal else (NAVY, "#4B5563")
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        if not terminal:
            parts.append(f'<rect x="{x}" y="{y}" width="4" height="{bh}" fill="{ACCENT}"/>')
        parts.append(f'<text x="{x + 18}" y="{y + 30}" font-size="14" font-weight="600" fill="{t_col}">{escape(title)}</text>')
        parts.append(f'<text x="{x + 18}" y="{y + 55}" font-size="12" fill="{d_col}">{escape(l1)}</text>')
        parts.append(f'<text x="{x + 18}" y="{y + 72}" font-size="12" fill="{d_col}">{escape(l2)}</text>')
    for i in range(len(PIPELINE_STEPS) - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if y1 == y2:
            parts.append(f'<line x1="{x1 + bw}" y1="{y1 + bh / 2}" x2="{x2 - 2}" y2="{y2 + bh / 2}" '
                         f'stroke="#6B7280" stroke-width="1.5" marker-end="url(#arw)"/>')
        else:
            cx1, cx2 = x1 + bw / 2, x2 + bw / 2
            mid = y1 + bh + gy / 2
            parts.append(f'<path d="M{cx1} {y1 + bh} V{mid} H{cx2} V{y2 - 2}" fill="none" stroke="#6B7280" '
                         f'stroke-width="1.5" marker-end="url(#arw)"/>')
    parts.append("</svg>")
    return "".join(parts)


def page_problem():
    header("Problem and Problem Analysis", "Chapter 2  |  Problem and Problem Analysis",
           "The problem addressed, the data used, and the input, process and output of each stage.")

    with st.container(key="card_p0"):
        card_title("Problem Statement")
        st.markdown(
            "Raw network intrusion data cannot be passed directly to a classifier without loss of reliability. "
            "Duplicate records inflate the apparent performance, constant and redundant features add noise, skewed features "
            "hinder distance-based and linear models, and rare attack classes are overwhelmed by normal traffic. "
            "This project builds a data engineering pipeline that diagnoses these problems, applies a targeted remedy to each, "
            "and verifies the effect experimentally.")

    with st.container(key="card_p1"):
        card_title("Pipeline Overview: Input, Process, Output")
        html(f'<div style="overflow-x:auto">{pipeline_svg()}</div>')

    with st.container(key="card_p2"):
        card_title("Input, Process and Output of Each Stage")
        html(table(["Stage", "Input", "Process", "Output"], IPO_ROWS))

    with st.container(key="card_p3"):
        card_title("Data Description")
        html(table(["Source file", "Role", "Records", "Columns"], [
            ("KDDTrain+.txt", "Training data (all engineering decisions)", fmt(len(train)), "43"),
            ("KDDTest+.txt", "Evaluation data (reporting only)", fmt(len(test)), "43"),
        ]))
        st.caption("Source: NSL-KDD, a revised version of the KDD Cup 1999 data set in which redundant records were reduced "
                   "(repository: github.com/defcom17/NSL_KDD). Each record is one network connection described by 41 features, "
                   "an attack label and a difficulty score.")
        st.markdown("**Feature groups**")
        html(table(["Group", "Count", "Attributes"], FEATURE_GROUPS))

    with st.container(key="card_p4"):
        card_title("Target Variable: Five Attack Categories")
        groups = {"Normal": "normal", "DoS": ", ".join(P.DOS), "Probe": ", ".join(P.PROBE),
                  "R2L": ", ".join(P.R2L), "U2R": ", ".join(P.U2R)}
        html(table(["Category", "Description", "Attack subtypes (label)"],
                   [(k, ATTACK_DESC[k], groups[k]) for k in P.LABELS]))

    with st.container(key="card_p5"):
        card_title("Sample of Extracted Raw Data (KDDTrain+)")
        st.dataframe(train.head(20))


def page_techniques():
    header("Tools and Techniques", "Chapter 3  |  Tools and Techniques",
           "Each technique is presented with its objective, input, process and output, followed by evidence computed from the data.")

    with st.container(key="card_t0"):
        card_title("Tools")
        html(table(["Tool", "Purpose in this project"], [
            ("Python 3", "Implementation language of the pipeline and the dashboard"),
            ("pandas and NumPy", "Data extraction, profiling, cleaning and transformation"),
            ("scikit-learn", "ColumnTransformer, encoders, scaler, models, cross-validation and metrics"),
            ("Plotly", "Interactive data visualisation"),
            ("Streamlit", "Interactive dashboard that documents and demonstrates the pipeline"),
        ]))

    sd = de["step_df"]
    n = len(sd)

    def delta(i, j, col):
        return int(sd[col].iloc[j] - sd[col].iloc[i]) if n > j else 0

    evidence = {
        "T1": f"Training data: {fmt(len(train))} rows. Test data: {fmt(len(test))} rows. Both validated against 43 columns.",
        "T2": (f"Missing values: {de['n_missing']}. Duplicates: train {de['n_dup_train']}, test {de['n_dup_test']}. "
               f"Constant features: {len(de['const_cols'])}. Maximum absolute skewness: {de['skew'].max():.0f}."),
        "T3": (f"Training rows changed by {delta(0, 1, 'train_rows'):+,}; "
               f"features changed by {delta(0, 1, 'n_features'):+d} (constant: {', '.join(de['const_cols']) or 'none'})."),
        "T4": (f"Threshold {corr_th}: {len(de['corr_drop'])} feature(s) removed "
               f"({', '.join(de['corr_drop']) or 'none'}); features changed by {delta(1, 2, 'n_features'):+d}."),
        "T5": f"Features changed by {delta(2, 3, 'n_features'):+d}. Top services kept: {len(de['top_services'])}; the rest grouped as 'other'.",
        "T6": f"Categorical columns one-hot encoded: {', '.join(CAT)}. Scaling applied to Logistic Regression only.",
        "T7": f"{n} pipeline stages validated; the step log below records rows and features at each stage.",
        "T8": "Executed on the Ablation Study page: four cumulative setups compared by cross-validated macro F1.",
    }
    for i, t in enumerate(TECHNIQUES):
        with st.container(key=f"card_tech_{i}"):
            html(f"""<div class="tech-head"><span class="tech-no">{t['no']}</span><span class="tech-name">{t['name']}</span>
            <span class="chip">{t['cat']}</span></div>
            <div class="tech-grid">
            <div><div class="lbl">Objective</div><div class="txt">{t['objective']}</div></div>
            <div><div class="lbl">Tool</div><div class="txt">{t['tool']}</div></div>
            <div><div class="lbl">Input</div><div class="txt">{t['inp']}</div></div>
            <div><div class="lbl">Output</div><div class="txt">{t['out']}</div></div>
            <div class="wide"><div class="lbl">Process</div><div class="txt">{t['process']}</div></div>
            </div>""")
            html(f'<div class="evidence"><b>Evidence.</b> {evidence[t["no"]]}</div>')

    with st.container(key="card_flow"):
        h1, h2 = st.columns([1, 1])
        with h1:
            card_title("Data Flow Across Pipeline Stages")
        with h2.container(key="pills"):
            metric = st.radio("Metric", ["Features", "Train rows", "Test rows"], horizontal=True, label_visibility="collapsed")
        sdf = sd.reset_index()
        ycol = {"Features": "n_features", "Train rows": "train_rows", "Test rows": "test_rows"}[metric]
        fig = go.Figure(go.Scatter(x=sdf["step"], y=sdf[ycol], mode="lines+markers", line=dict(color=ACCENT, width=3),
                                   marker=dict(size=9, color="#fff", line=dict(color=ACCENT, width=3)),
                                   fill="tozeroy", fillcolor="rgba(47,93,138,.10)"))
        lo, hi = sdf[ycol].min(), sdf[ycol].max()
        pad = max((hi - lo) * 0.5, 1)
        fig.update_yaxes(range=[max(lo - pad, 0), hi + pad])
        st.plotly_chart(plotly_style(fig, 300), key="flow_chart")
        last, first = sdf[ycol].iloc[-1], sdf[ycol].iloc[0]
        st.markdown(f"<span class='muted'>{metric}: {fmt(first)} to </span><b>{fmt(last)}</b> "
                    f"<span class='{'pos' if last >= first else 'neg'}'>({last - first:+,})</span>", unsafe_allow_html=True)

    with st.container(key="card_steplog"):
        card_title("Step Log (validated after every stage)")
        st.dataframe(sd)
        st.caption("The validation routine asserts: no missing values, equal X and y lengths, identical train and test schema, "
                   "categorical columns present, and every label belonging to the five categories.")


def page_quality():
    header("Data Quality", "Chapter 3  |  Tools and Techniques",
           "Technique T2 and T4: profiling, cleaning and correlation-based feature selection.")
    m = st.columns(4)
    m[0].metric("Missing values", de["n_missing"])
    m[1].metric("Duplicates (train / test)", f"{de['n_dup_train']} / {de['n_dup_test']}")
    m[2].metric("Constant features", len(de["const_cols"]))
    m[3].metric("Unseen subtypes in test", len(de["unseen"]))

    counts = de["counts"]
    min_class = counts["train"].idxmin()
    top_skew = de["skew"].head(3)
    report = pd.DataFrame([
        ["Missing values", f"{de['n_missing']}", "No action required"],
        ["Duplicate records", f"Train {de['n_dup_train']} | Test {de['n_dup_test']}", "Removed from the training set; test set retained as published"],
        ["Constant features", f"{len(de['const_cols'])}: {', '.join(de['const_cols']) or 'none'}", "Removed (no information content)"],
        [f"Highly correlated features (> {corr_th})", f"{len(de['corr_drop'])}: {', '.join(de['corr_drop']) or 'none'}", "Removed (redundant)"],
        ["Categorical features", f"{len(CAT)}: {', '.join(CAT)}", "One-hot encoding (unknown categories ignored)"],
        ["Skewed numeric features", f"{', '.join(top_skew.index)} (maximum skewness {top_skew.iloc[0]:.0f})",
         "Log transformation evaluated in feature engineering"],
        ["Class imbalance", f"{min_class} = {fmt(counts.loc[min_class, 'train'])} vs Normal = {fmt(counts.loc['Normal', 'train'])}",
         "Balanced class weights; Macro F1 reported alongside accuracy"],
        ["Unseen attack subtypes in test", f"{len(de['unseen'])} subtypes ({fmt(de['n_unseen_rows'])} rows)",
         "Reported as a limitation and analysed in Error Analysis"],
    ], columns=["Data Quality Check", "Result", "Action"])

    with st.container(key="card_qr"):
        card_title("Data Quality Report: Problem and Remedy")
        html(table(report.columns, report.values.tolist()))

    a, b = st.columns(2)
    with a.container(key="card_class"):
        card_title("Class Distribution (logarithmic scale)")
        cdf = counts.reset_index().melt(id_vars="Attack Type", var_name="set", value_name="count")
        fig = px.bar(cdf, x="Attack Type", y="count", color="set", barmode="group", log_y=True,
                     color_discrete_map={"train": NAVY, "test": MUTED_BLUE})
        st.plotly_chart(plotly_style(fig), key="class_chart")
    with b.container(key="card_skew"):
        card_title("Ten Most Skewed Features (absolute skewness)")
        sk = de["skew"].head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=sk.values, y=sk.index, orientation="h", marker_color=ACCENT))
        st.plotly_chart(plotly_style(fig), key="skew_chart")

    a, b = st.columns([1, 1.2])
    with a.container(key="card_pairs"):
        card_title(f"Feature Pairs with Correlation Above {corr_th}")
        st.dataframe(de["corr_pairs"].round(3))
        st.markdown(f"**Removed features:** `{', '.join(de['corr_drop']) or 'none'}`")
    with b.container(key="card_heat"):
        card_title("Correlation Heatmap (training data after deduplication)")
        fig = px.imshow(de["corr_matrix"], color_continuous_scale=HEAT_SCALE, zmin=0, zmax=1, aspect="auto")
        st.plotly_chart(plotly_style(fig, 420), key="heat_chart")

    with st.container(key="card_unseen"):
        card_title("Attack Subtypes Present Only in the Test Set")
        st.write(f"{len(de['unseen'])} subtypes account for **{fmt(de['n_unseen_rows'])} of {fmt(len(test))}** test records.")
        st.write(", ".join(de["unseen"]))


def page_features():
    header("Feature Engineering", "Chapter 3  |  Tools and Techniques",
           "Technique T5: derived features created to address skewness and to expose domain relationships.")
    spec = pd.DataFrame([
        ["bytes_ratio", "Ratio", "src_bytes / (dst_bytes + 1)", "Relative volume of outbound to inbound traffic"],
        ["src_bytes_log", "Log transformation", "log(1 + src_bytes)", "Reduce skewness"],
        ["dst_bytes_log", "Log transformation", "log(1 + dst_bytes)", "Reduce skewness"],
        ["duration_log", "Log transformation", "log(1 + duration)", "Reduce skewness"],
        ["err_rate_mean", "Aggregation", "mean(serror_rate, rerror_rate)", "Combine error indicators"],
        ["srv_ratio", "Ratio", "srv_count / (count + 1)", "Proportion of connections to the same service"],
        ["host_srv_ratio", "Ratio", "dst_host_srv_count / (dst_host_count + 1)", "Service proportion at host level"],
        ["suspicious_cnt", "Aggregation", "hot + num_failed_logins + num_compromised + root_shell + su_attempted + "
                                          "num_file_creations + num_shells + num_access_files", "Aggregate suspicious behaviour indicators"],
        ["service (grouped)", "Category grouping", "Top 15 services in training; all others labelled 'other'", "Reduce one-hot cardinality"],
    ], columns=["Feature", "Technique", "Formula", "Purpose"])
    with st.container(key="card_fe"):
        card_title("Engineered Features (experimental; evaluated in the Ablation Study)")
        html(table(spec.columns, spec.values.tolist()))
    Xtr = de["setups"]["3 + engineered features"][0]
    with st.container(key="card_fe2"):
        card_title("Sample Data After Feature Engineering")
        new_cols = ["bytes_ratio", "src_bytes_log", "dst_bytes_log", "duration_log", "err_rate_mean",
                    "srv_ratio", "host_srv_ratio", "suspicious_cnt", "service"]
        st.dataframe(Xtr[new_cols].head(15))
        st.write(f"Top services (training): `{', '.join(sorted(de['top_services']))}`")
        st.write(f"Total features: **{Xtr.shape[1]}** ({Xtr.shape[1] - len(CAT)} numeric and {len(CAT)} categorical; "
                 "categorical features are one-hot encoded inside the model pipeline).")


def page_ablation():
    header("Ablation Study", "Chapter 3  |  Tools and Techniques",
           "Technique T8: does each data engineering step improve the downstream model?")
    with st.container(key="card_ab"):
        h1, h2 = st.columns([4, 1])
        with h1:
            card_title("Effect of Data Engineering on Random Forest Performance")
        if h2.button("Run ablation"):
            run_experiments()
        st.caption("The best setup is selected by 3-fold cross-validated macro F1 on the training set. "
                   "Results on KDDTest+ are reported only, to prevent data leakage.")
        if "exp" not in st.session_state:
            st.info("Select Run ablation to execute 3-fold cross-validation for all four setups.")
            return
        ab, best, _ = st.session_state["exp"]
        st.dataframe(ab)
        st.success(f"Selected setup (highest cross-validated macro F1 on training data): {best}")
        long = ab.reset_index().melt(id_vars="setup", value_vars=["cv_macro_f1", "test_macro_f1"], var_name="metric")
        fig = px.bar(long, x="setup", y="value", color="metric", barmode="group",
                     color_discrete_map={"cv_macro_f1": NAVY, "test_macro_f1": MUTED_BLUE})
        st.plotly_chart(plotly_style(fig, 360), key="ab_chart")
        st.caption("Cross-validation scores on the training set are typically higher than test scores because training records "
                   "are highly similar to one another and the test set contains previously unseen attack subtypes.")


def page_models():
    header("Model Evaluation", "Chapter 3  |  Tools and Techniques",
           "Downstream application: comparison of models trained on the output of the pipeline.")
    if "exp" not in st.session_state:
        with st.container(key="card_m0"):
            st.info("Experiments have not been executed. Run them to compute the ablation and the model comparison.")
            if st.button("Run experiments"):
                run_experiments()
                st.rerun()
        return
    ab, best, comp = st.session_state["exp"]
    with st.container(key="card_m1"):
        card_title(f"Model Comparison (setup: {best})")
        st.dataframe(comp)
        st.caption("Feature scaling is required for Logistic Regression only. Random Forest is insensitive to feature scale, "
                   "as confirmed by the comparison above.")
    model, pred, _ = get_final(best, n_est, de, data_key)
    y_test = de["y_test"]
    rep = pd.DataFrame(classification_report(y_test, pred, labels=P.LABELS, digits=3, zero_division=0, output_dict=True)).T
    a, b = st.columns(2)
    with a.container(key="card_rep"):
        card_title("Classification Report (KDDTest+)")
        st.dataframe(rep.round(3))
    with b.container(key="card_cm"):
        card_title("Confusion Matrix (normalised by true class, equal to recall)")
        cm = confusion_matrix(y_test, pred, labels=P.LABELS, normalize="true")
        fig = px.imshow(cm, x=P.LABELS, y=P.LABELS, text_auto=".2f", color_continuous_scale=["#FFFFFF", NAVY], zmin=0, zmax=1)
        fig.update_layout(xaxis_title="Predicted", yaxis_title="True")
        st.plotly_chart(plotly_style(fig, 360), key="cm_chart")
    with st.container(key="card_imp"):
        card_title("Top 15 Feature Importances")
        names = model.named_steps["pre"].get_feature_names_out()
        imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False).head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ACCENT))
        st.plotly_chart(plotly_style(fig, 420), key="imp_chart")


def page_error():
    header("Error Analysis: R2L and U2R", "Chapter 3  |  Tools and Techniques",
           "Why the two rarest attack categories are difficult to detect.")
    if "exp" not in st.session_state:
        st.info(f"Experiments have not been executed. The results below use the default setup ({BEST_DEFAULT}).")
    model, pred, _ = get_final(best_setup, n_est, de, data_key)
    sub, by_seen, r2l_pred = P.error_analysis(test, pred, de["unseen"])
    with st.container(key="card_e1"):
        card_title("Recall for Seen and Unseen Subtypes")
        st.dataframe(by_seen)
    with st.container(key="card_e2"):
        card_title("Recall by Attack Subtype (ordered by frequency)")
        st.dataframe(sub.head(25))
    with st.container(key="card_e3"):
        card_title("Predicted Classes for R2L Records")
        fig = px.bar(r2l_pred.reset_index(), x="pred", y="count", color="pred", color_discrete_map=CLASS_COLORS)
        st.plotly_chart(plotly_style(fig, 300), key="r2l_chart")


def page_demo():
    header("Demonstration", "Program Demonstration",
           "Select a connection record from KDDTest+ and inspect the prediction produced by the trained model.")
    model, pred, proba = get_final(best_setup, n_est, de, data_key)
    classes = list(model.named_steps["clf"].classes_)
    with st.container(key="card_p1"):
        card_title("Classify a Test Connection")
        idx = st.slider("Record index", 0, len(test) - 1, 0)
        true, p = test["Attack Type"].iloc[idx], pred[idx]
        c = st.columns(3)
        c[0].metric("Actual", f"{true} ({test['label'].iloc[idx]})")
        c[1].metric("Predicted", p)
        c[2].metric("Result", "Correct" if true == p else "Incorrect")
        fig = go.Figure(go.Bar(x=classes, y=proba[idx], marker_color=[CLASS_COLORS[k] for k in classes]))
        fig.update_yaxes(range=[0, 1], title="Predicted probability")
        st.plotly_chart(plotly_style(fig, 300), key="pred_chart")
        with st.expander("Feature values of this record (after data engineering)"):
            Xte = de["setups"][best_setup][2]
            st.dataframe(Xte.iloc[[idx]].T)


def page_conclusion():
    header("Conclusion and Recommendations", "Chapter 4  |  Summary and Recommendations",
           "Summary of the results of the pipeline, its limitations, and suggested future work.")
    counts = de["counts"]
    findings = [
        f"Quality assessment found {de['n_missing']} missing values, {fmt(de['n_dup_train'])} duplicate training records, "
        f"{len(de['const_cols'])} constant feature(s) and {len(de['corr_drop'])} feature(s) correlated above {corr_th}; each was addressed by a specific cleaning or selection step.",
        f"The test set contains {len(de['unseen'])} attack subtypes that never occur in training, covering "
        f"{fmt(de['n_unseen_rows'])} of {fmt(len(test))} test records.",
        f"Class imbalance is severe: the smallest class ({counts['train'].idxmin()}) has {fmt(counts['train'].min())} training records, "
        f"compared with {fmt(counts.loc['Normal', 'train'])} for Normal.",
    ]
    with st.container(key="card_c1"):
        card_title("Summary of Results")
        if "exp" in st.session_state:
            ab, best, comp = st.session_state["exp"]
            raw, sel = ab.iloc[0], ab.loc[best]
            findings.append(
                f"The setup selected by cross-validated macro F1 on training data is \"{best}\" (CV macro F1 {sel['cv_macro_f1']:.3f}). "
                f"Test macro F1 changed from {raw['test_macro_f1']:.3f} (raw data) to {sel['test_macro_f1']:.3f} "
                f"({sel['test_macro_f1'] - raw['test_macro_f1']:+.3f}).")
            html(bullets(findings))
            final = comp.loc["Random Forest (no scaler)"]
            k = st.columns(4)
            k[0].metric("Test accuracy", f"{final['test_acc']:.3f}")
            k[1].metric("Test macro F1", f"{final['test_macro_f1']:.3f}")
            if "recall_R2L" in comp.columns:
                k[2].metric("R2L recall", f"{final['recall_R2L']:.3f}")
                k[3].metric("U2R recall", f"{final['recall_U2R']:.3f}")
        else:
            html(bullets(findings))
            st.info("Model results are not yet available. Run the experiments to include accuracy, macro F1 and recall.")
            if st.button("Run experiments"):
                run_experiments()
                st.rerun()

    a, b = st.columns(2)
    with a.container(key="card_c2"):
        card_title("Limitations")
        html(bullets([
            "Cross-validation on the training set is optimistic because training records are highly similar to one another.",
            "R2L and U2R contain few training records, and many test-set attacks of these types are subtypes never seen in training.",
            "Random Forest hyperparameters are fixed to limit complexity and were not tuned extensively.",
            "The pipeline was evaluated on a single benchmark data set collected in a laboratory setting.",
        ]))
    with b.container(key="card_c3"):
        card_title("Recommendations")
        html(bullets([
            "Apply resampling or cost-sensitive learning to improve recall for R2L and U2R.",
            "Evaluate anomaly-detection approaches for attack subtypes that are absent from training data.",
            "Tune model hyperparameters with nested cross-validation and use grouped splits to reduce optimistic estimates.",
            "Schedule and version the pipeline with a workflow orchestrator and add data-drift monitoring for production use.",
            "Validate the pipeline on more recent traffic data sets.",
        ]))


PAGES = {"Introduction": page_intro, "Problem Analysis": page_problem, "Tools and Techniques": page_techniques,
         "Data Quality": page_quality, "Feature Engineering": page_features, "Ablation Study": page_ablation,
         "Model Evaluation": page_models, "Error Analysis": page_error, "Demonstration": page_demo,
         "Conclusion": page_conclusion}
PAGES[page]()
