import datetime as dt
import os

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Network Intrusion Detection", layout="wide")

if not (os.path.exists("model.joblib") and os.path.exists("sample_test.csv")):
    with st.spinner("First run: downloading NSL-KDD and training the model (about 1 minute)..."):
        import train_model
        train_model.main()


@st.cache_resource
def load():
    return joblib.load("model.joblib"), pd.read_csv("sample_test.csv")


B, SAMPLE = load()
RAW, CAT = B["raw"], ["protocol_type", "service", "flag"]
NUMCOLS = [c for c in RAW if c not in CAT]
COL = {"Normal": "#93c5fd", "DoS": "#1e3a8a", "Probe": "#3b82f6", "R2L": "#1d4ed8", "U2R": "#0b1e4f", "Uncertain": "#94a3b8"}
CLASSES = list(COL)
SEV = {"Normal": "None", "Probe": "Medium", "DoS": "High", "R2L": "High", "U2R": "Critical", "Uncertain": "Review"}

st.markdown("""<style>
html, body, [class*="css"] {font-family: "Segoe UI", Arial, sans-serif;}
#MainMenu, footer {visibility: hidden;}
[data-testid="stSidebar"] {background: #0b1e4f;}
[data-testid="stSidebar"] * {color: #e2e8f0;}
h1, h2, h3 {color: #0b1e4f;}
.kpi {border: 1px solid #93a5cf; border-left: 4px solid #1d4ed8; background: #f5f8fe; padding: 8px 12px;}
.kl {font-size: 12px; color: #475569; text-transform: uppercase; letter-spacing: .04em;}
.kv {font-size: 24px; font-weight: 600; color: #0b1e4f;}
.xlwrap {overflow-x: auto; margin-bottom: 12px;}
table.xl {border-collapse: collapse; width: 100%; font-size: 13px;}
table.xl th {background: #1e3a8a; color: #fff; border: 1px solid #93a5cf; padding: 6px 10px; text-align: left;}
table.xl td {border: 1px solid #cbd5e1; padding: 5px 10px; color: #0f172a;}
table.xl tr:nth-child(even) td {background: #eef3fb;}
table.xl td:first-child {background: #e2e8f0 !important; color: #475569; text-align: center; width: 44px;}
</style>""", unsafe_allow_html=True)


def kpi(label, value):
    return f'<div class="kpi"><div class="kl">{label}</div><div class="kv">{value}</div></div>'


def xl(df, n=200):
    d = df.head(n).copy()
    d.insert(0, "#", range(1, len(d) + 1))
    st.markdown('<div class="xlwrap">' + d.to_html(index=False, classes="xl", border=0) + "</div>", unsafe_allow_html=True)


# ---------- Model helpers ----------
def prep(df):
    X = df.copy()
    for c in RAW:
        if c not in X:
            X[c] = B["defaults"][c]
    return X[RAW].drop(columns=B["drop"])


def predict(df, thr):
    proba = B["model"].predict_proba(prep(df))
    cls = B["model"].classes_
    out = pd.DataFrame({"Prediction": cls[proba.argmax(1)], "Confidence": proba.max(1).round(3)}, index=df.index)
    out.loc[out["Confidence"] < thr, "Prediction"] = "Uncertain"
    out["Severity"] = out["Prediction"].map(SEV)
    return out, pd.DataFrame(proba, columns=cls, index=df.index)


@st.cache_data
def stream_pred(thr):
    out, _ = predict(SAMPLE[RAW], thr)
    return pd.concat([SAMPLE[RAW], out], axis=1)


# ---------- Sidebar ----------
st.sidebar.markdown("### Network Intrusion Detection")
page = st.sidebar.radio("Navigation", ["Live Monitor", "Analyze", "Info"], label_visibility="collapsed")
thr = st.sidebar.slider("Confidence threshold", 0.30, 0.90, 0.50, 0.05,
                        help="Predictions with confidence below this value are labelled Uncertain.")
st.sidebar.caption("Model: Random Forest trained on NSL-KDD (KDDTrain+).")


# ---------- Page: Live Monitor ----------
def page_monitor():
    S = st.session_state
    S.setdefault("idx", 0)
    S.setdefault("running", False)
    S.setdefault("t0", dt.datetime.now().replace(microsecond=0))
    P = stream_pred(thr)
    st.title("Live Monitor")
    st.caption("Simulated stream: connection records from KDDTest+ are replayed one by one and classified by the model.")
    c = st.columns([1, 1, 1, 2, 3])
    if c[0].button("Start", type="primary", use_container_width=True):
        S.running = True
    if c[1].button("Pause", use_container_width=True):
        S.running = False
    if c[2].button("Reset", use_container_width=True):
        S.idx, S.running, S.t0 = 0, False, dt.datetime.now().replace(microsecond=0)
    c[3].selectbox("Speed (rows per second)", [1, 5, 10, 25], index=1, key="speed")

    def body():
        if S.running:
            S.idx = min(S.idx + S.speed, len(P))
            if S.idx >= len(P):
                S.running = False
                st.rerun()
        d = P.iloc[:S.idx]
        if d.empty:
            st.info("Press Start to begin the replay.")
            return
        alerts = d[d["Prediction"] != "Normal"]
        k = st.columns(4)
        k[0].markdown(kpi("Connections processed", f"{len(d):,}"), unsafe_allow_html=True)
        k[1].markdown(kpi("Alerts", f"{len(alerts):,}"), unsafe_allow_html=True)
        k[2].markdown(kpi("Alert rate", f"{len(alerts) / len(d):.1%}"), unsafe_allow_html=True)
        k[3].markdown(kpi("Critical", f"{int((d['Severity'] == 'Critical').sum()):,}"), unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([3, 2])
        cum = pd.get_dummies(d["Prediction"]).reindex(columns=CLASSES, fill_value=0).astype(int).cumsum().reset_index(drop=True)
        left.markdown("**Cumulative connections by predicted class**")
        left.line_chart(cum.iloc[::max(1, len(cum) // 300)], color=list(COL.values()), height=260)
        counts = d["Prediction"].value_counts().reindex(CLASSES, fill_value=0).rename_axis("Class").reset_index(name="Count")
        right.markdown("**Current distribution**")
        right.altair_chart(alt.Chart(counts).mark_bar().encode(
            x=alt.X("Class", sort=CLASSES, title=None), y=alt.Y("Count", title=None),
            color=alt.Color("Class", scale=alt.Scale(domain=CLASSES, range=list(COL.values())), legend=None),
        ).properties(height=260), use_container_width=True)
        r = d.tail(15).iloc[::-1]
        st.markdown("**Recent activity**")
        xl(pd.DataFrame({
            "Time": [(S.t0 + dt.timedelta(seconds=int(i))).strftime("%H:%M:%S") for i in r.index],
            "Connection": r.index + 1, "Protocol": r["protocol_type"], "Service": r["service"], "Flag": r["flag"],
            "Src bytes": r["src_bytes"], "Dst bytes": r["dst_bytes"], "Prediction": r["Prediction"],
            "Confidence": r["Confidence"], "Severity": r["Severity"],
        }))

    st.fragment(run_every=1 if S.running else None)(body)()


# ---------- Page: Analyze ----------
FORM = ["protocol_type", "service", "flag", "duration", "src_bytes", "dst_bytes", "count", "srv_count",
        "serror_rate", "rerror_rate", "same_srv_rate", "logged_in", "num_failed_logins", "hot"]
FLT = ["serror_rate", "rerror_rate", "same_srv_rate"]
LABEL = {"protocol_type": "Protocol", "service": "Service", "flag": "Connection flag", "duration": "Duration (s)",
         "src_bytes": "Source bytes", "dst_bytes": "Destination bytes", "count": "Connections to same host (2s)",
         "srv_count": "Connections to same service (2s)", "serror_rate": "SYN error rate",
         "rerror_rate": "REJ error rate", "same_srv_rate": "Same service rate", "logged_in": "Logged in",
         "num_failed_logins": "Failed logins", "hot": "Hot indicators"}


def conv(f, v):
    if f in CAT:
        return str(v)
    return float(v) if f in FLT else int(round(float(v)))


def load_example(cls):
    row = SAMPLE[SAMPLE["Attack Type"] == cls].iloc[0]
    for f in FORM:
        st.session_state["f_" + f] = conv(f, row[f])


def single(thr):
    for f in FORM:
        st.session_state.setdefault("f_" + f, conv(f, B["defaults"][f]))
    st.markdown("**Load an example record**")
    b = st.columns(6)
    for i, cls in enumerate(["Normal", "DoS", "Probe", "R2L"]):
        b[i].button(cls, on_click=load_example, args=(cls,), use_container_width=True, key="ex_" + cls)
    cols = st.columns(3)
    for i, f in enumerate(FORM):
        col, key = cols[i % 3], "f_" + f
        if f in CAT:
            col.selectbox(LABEL[f], B["opts"][f], key=key)
        elif f == "logged_in":
            col.selectbox(LABEL[f], [0, 1], key=key)
        elif f in FLT:
            col.number_input(LABEL[f], 0.0, 1.0, step=0.05, key=key)
        else:
            col.number_input(LABEL[f], min_value=0, step=1, key=key)
    st.caption("Fields not shown use the training-set median (numeric) or mode (categorical).")
    if st.button("Run prediction", type="primary"):
        row = pd.DataFrame([{**B["defaults"], **{f: st.session_state["f_" + f] for f in FORM}}])
        out, pr = predict(row, thr)
        k = st.columns(3)
        k[0].markdown(kpi("Prediction", out["Prediction"][0]), unsafe_allow_html=True)
        k[1].markdown(kpi("Confidence", f"{out['Confidence'][0]:.1%}"), unsafe_allow_html=True)
        k[2].markdown(kpi("Severity", out["Severity"][0]), unsafe_allow_html=True)
        st.write("")
        a, c = st.columns(2)
        with a:
            st.markdown("**Class probabilities**")
            xl(pd.DataFrame({"Class": pr.columns, "Probability": pr.iloc[0].round(3).values}))
        with c:
            st.markdown("**Most influential features (model-wide) and this record's values**")
            top = sorted(B["importance"].items(), key=lambda x: -x[1])[:8]
            xl(pd.DataFrame([{"Feature": f, "Importance": round(v, 3), "Value": row[f][0]} for f, v in top]))


def validate(df):
    X, checks = df.copy(), []
    missing = [c for c in RAW if c not in X]
    extra = [c for c in X.columns if c not in RAW]
    checks.append(["Row count", f"{len(X):,}", "None"])
    checks.append(["Missing required columns", ", ".join(missing) or "None", "Filled with training median/mode" if missing else "None"])
    checks.append(["Extra columns", ", ".join(extra) or "None", "Ignored" if extra else "None"])
    if len(missing) > 0.25 * len(RAW):
        return None, checks
    for c in missing:
        X[c] = B["defaults"][c]
    X[NUMCOLS] = X[NUMCOLS].apply(pd.to_numeric, errors="coerce")
    n_na = int(X[RAW].isna().sum().sum())
    unk = sum(int((~X[c].isin(B["opts"][c])).sum()) for c in CAT)
    checks.append(["Missing or non-numeric cells", f"{n_na:,}", "Filled with training median/mode" if n_na else "None"])
    checks.append(["Unknown category values", f"{unk:,}", "Ignored by the encoder" if unk else "None"])
    return X.fillna(B["defaults"])[RAW], checks


def batch(thr):
    c1, c2 = st.columns([3, 2])
    up = c1.file_uploader("Upload CSV (header row required, NSL-KDD column names)", type="csv")
    use = c2.checkbox("Use built-in sample (500 rows from KDDTest+)")
    c2.download_button("Download template CSV", SAMPLE[RAW].head(20).to_csv(index=False), "template.csv", "text/csv")
    if use:
        data = SAMPLE[RAW].head(500)
    elif up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            return
    else:
        st.info("Upload a CSV file or tick the built-in sample to begin.")
        return
    clean, checks = validate(data)
    st.markdown("**Data validation**")
    xl(pd.DataFrame(checks, columns=["Check", "Result", "Action"]))
    if clean is None:
        st.error("Too many required columns are missing. Use the template CSV for the expected format.")
        return
    out, _ = predict(clean, thr)
    res = pd.concat([clean, out], axis=1)
    k = st.columns(4)
    k[0].markdown(kpi("Rows analysed", f"{len(res):,}"), unsafe_allow_html=True)
    k[1].markdown(kpi("Alerts", f"{int((res['Prediction'] != 'Normal').sum()):,}"), unsafe_allow_html=True)
    k[2].markdown(kpi("Critical", f"{int((res['Severity'] == 'Critical').sum()):,}"), unsafe_allow_html=True)
    k[3].markdown(kpi("Uncertain", f"{int((res['Prediction'] == 'Uncertain').sum()):,}"), unsafe_allow_html=True)
    st.write("")
    summ = res["Prediction"].value_counts().reindex(CLASSES, fill_value=0).rename_axis("Class").reset_index(name="Count")
    summ["Share"] = (summ["Count"] / len(res)).map("{:.1%}".format)
    summ["Severity"] = summ["Class"].map(SEV)
    st.markdown("**Summary by class**")
    xl(summ)
    st.markdown("**Results**")
    f1, f2 = st.columns([3, 2])
    sel = f1.multiselect("Filter by prediction", CLASSES, default=CLASSES)
    minc = f2.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    view = res[res["Prediction"].isin(sel) & (res["Confidence"] >= minc)]
    view = view[["protocol_type", "service", "flag", "duration", "src_bytes", "dst_bytes", "Prediction", "Confidence", "Severity"]]
    view.index = view.index + 1
    st.dataframe(view, use_container_width=True, height=380)
    st.download_button("Download full results (CSV)", res.to_csv(index=False), "predictions.csv", "text/csv")


def page_analyze():
    st.title("Analyze")
    t1, t2 = st.tabs(["Single connection", "Upload file"])
    with t1:
        single(thr)
    with t2:
        batch(thr)


# ---------- Page: Info ----------
def page_info():
    m = B["metrics"]
    st.title("Info")
    st.subheader("Model performance (KDDTest+)")
    k = st.columns(3)
    k[0].markdown(kpi("Accuracy", f"{m['acc']:.1%}"), unsafe_allow_html=True)
    k[1].markdown(kpi("Macro F1", f"{m['macro_f1']:.3f}"), unsafe_allow_html=True)
    k[2].markdown(kpi("Features used", m["n_features"]), unsafe_allow_html=True)
    st.write("")
    xl(pd.DataFrame({"Class": list(m["recall"]), "Test rows": list(m["support"].values()),
                     "Recall": [f"{v:.1%}" for v in m["recall"].values()]}))
    st.caption("R2L and U2R have low recall: they are rare in training data and KDDTest+ contains attack subtypes "
               "never seen in training. Use the confidence threshold to flag uncertain cases for manual review.")
    st.subheader("Attack categories")
    xl(pd.DataFrame([
        ["DoS", "Floods a service with traffic to make it unavailable", "neptune, smurf, back, teardrop", "Rate-limit or block the source; enable SYN protection"],
        ["Probe", "Scans hosts and ports to discover services", "ipsweep, portsweep, nmap, satan", "Block scanning sources; review exposed ports"],
        ["R2L", "Remote attacker gains local access without an account", "guess_passwd, warezclient, ftp_write", "Enforce strong credentials; review login logs"],
        ["U2R", "Local user escalates to administrator privileges", "buffer_overflow, rootkit, loadmodule", "Isolate the host; patch and audit privileges"],
    ], columns=["Class", "Description", "Example subtypes", "Suggested response"]))


{"Live Monitor": page_monitor, "Analyze": page_analyze, "Info": page_info}[page]()
