import datetime as dt
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NetGuard", layout="wide")

if not (os.path.exists("model.joblib") and os.path.exists("sample_test.csv")):
    with st.spinner("First run: preparing the model (about 1 minute)..."):
        import train_model
        train_model.main()


@st.cache_resource
def load():
    return joblib.load("model.joblib"), pd.read_csv("sample_test.csv")


B, SAMPLE = load()
S = st.session_state
RAW, CAT = B["raw"], ["protocol_type", "service", "flag"]
NUMCOLS = [c for c in RAW if c not in CAT]
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R", "Uncertain"]
SEV = {"Normal": "None", "Probe": "Medium", "DoS": "High", "R2L": "High", "U2R": "Critical", "Uncertain": "Review"}
RANK = ["None", "Review", "Medium", "High", "Critical"]
GUIDE = {
    "Normal": ("No threat detected", "This connection looks like ordinary activity.", "No action needed."),
    "DoS": ("Denial of service", "The source is flooding the destination to make a service unavailable.", "Rate-limit or block the source and enable flood protection."),
    "Probe": ("Scanning or probing", "The source is scanning hosts or ports to find out which services are open.", "Block the scanning source and review which ports are exposed."),
    "R2L": ("Remote access attempt", "A remote attacker is trying to get into a machine without an account, for example by guessing passwords.", "Enforce strong credentials and review the login logs."),
    "U2R": ("Privilege escalation", "A local user is trying to gain administrator rights.", "Isolate the host, patch it and audit its privileges."),
    "Uncertain": ("Needs review", "The model is not confident enough to name a category.", "Review this connection manually."),
}
FLAGS = {"SF": "Completed normally", "S0": "Attempt made, no reply", "REJ": "Rejected by destination",
         "RSTO": "Reset by source", "RSTR": "Reset by destination", "S1": "Established, never closed",
         "S2": "Closed by source only", "S3": "Closed by destination only", "SH": "Source half-open",
         "RSTOS0": "Source reset after first packet", "OTH": "Other"}
PCT = ["serror_rate", "rerror_rate", "same_srv_rate"]
FIELDS = {
    "protocol_type": ("Protocol", "Transport protocol used by the connection."),
    "service": ("Destination service", "Service the connection was aimed at, for example http, ftp or smtp."),
    "flag": ("Connection outcome", "How the connection ended."),
    "duration": ("Duration (seconds)", "How long the connection lasted."),
    "src_bytes": ("Data sent by source (bytes)", "Bytes sent from the source to the destination."),
    "dst_bytes": ("Data returned by destination (bytes)", "Bytes sent back from the destination to the source."),
    "count": ("Connections to the same host", "Connections to the same destination host in the last 2 seconds."),
    "srv_count": ("Connections to the same service", "Connections to the same service in the last 2 seconds."),
    "serror_rate": ("Half-open connections", "Share of those connections that never finished the handshake."),
    "rerror_rate": ("Rejected connections", "Share of those connections that the destination rejected."),
    "same_srv_rate": ("Aimed at the same service", "Share of those connections that went to the same service."),
    "logged_in": ("Login succeeded", "Whether the connection logged in successfully."),
    "num_failed_logins": ("Failed login attempts", "Failed logins during the connection."),
    "hot": ("Sensitive system actions", "Actions such as entering system directories or running programs."),
}
EXAMPLES = {"Ordinary traffic": "Normal", "Flood-like traffic": "DoS", "Scanning-like traffic": "Probe", "Remote login attempt": "R2L"}

st.markdown("""<style>
html, body, [class*="css"] {font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stAppViewContainer"] {background: radial-gradient(circle at 0% 0%, #cfe1ff 0, transparent 38%),
  radial-gradient(circle at 100% 8%, #d4f0fb 0, transparent 34%), #f5f9ff;}
.block-container {padding-top: 2.2rem; max-width: 1200px;}
[data-testid="stSidebar"] {background: #fff; border-right: 1px solid #e3ecfa;}
.brand {display: flex; align-items: center; gap: 12px; font-size: 27px; font-weight: 700; color: #0b1e4f; margin: 10px 0 26px;}
.cap {font-size: 13px; color: #7b8db0; margin: 18px 0 8px 4px; letter-spacing: .03em;}
.pt {font-size: 38px; font-weight: 700; color: #0b1e4f; margin: 0; padding: 0;}
.ps {color: #52637f; margin: 6px 0 22px; max-width: 760px; line-height: 1.5;}
[class*="st-key-card_"] {background: #fff; border: 1px solid #e0e9f8; border-radius: 22px; padding: 20px 24px;
  box-shadow: 0 10px 30px rgba(37, 99, 235, .07); margin-bottom: 10px;}
.ch {font-weight: 600; color: #0b1e4f; margin-bottom: 8px; font-size: 16px;}
.kpi {background: #fff; border: 1px solid #e0e9f8; border-radius: 22px; padding: 18px 22px; box-shadow: 0 10px 30px rgba(37, 99, 235, .07);}
.kl {font-size: 14px; color: #52637f;}
.kv {font-size: 30px; font-weight: 700; color: #0b1e4f; margin-top: 4px;}
.pill {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; background: #e0ecff; color: #1d4ed8;}
.pill.hot {background: linear-gradient(135deg, #1d4ed8, #38bdf8); color: #fff;}
.brow {display: flex; align-items: center; gap: 12px; margin: 9px 0; font-size: 14px; color: #0b1e4f;}
.bl {width: 90px;} .bv {width: 90px; text-align: right; color: #52637f;}
.bt {flex: 1; height: 10px; background: #e8f0fd; border-radius: 6px; overflow: hidden;}
.bf {height: 100%; background: linear-gradient(90deg, #2563eb, #38bdf8); border-radius: 6px;}
.xlwrap {overflow-x: auto; margin: 4px 0 8px;}
table.xl {border-collapse: collapse; width: 100%; font-size: 13.5px;}
table.xl th {background: #1e40af; color: #fff; border: 1px solid #b6c8ee; padding: 7px 10px; text-align: left; font-weight: 600;}
table.xl td {border: 1px solid #d6e2f6; padding: 6px 10px; color: #0f172a;}
table.xl tr:nth-child(even) td {background: #f1f6ff;}
table.xl td:first-child {background: #e6eefb !important; color: #64748b; text-align: center; width: 44px;}
.stButton > button {border-radius: 12px; font-weight: 600;}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #2563eb, #38bdf8); border: 0; color: #fff;}
[data-testid="stSidebar"] .stButton > button {justify-content: flex-start; border: 0; background: transparent; color: #3b5583;
  border-radius: 16px; padding: .65rem 1rem; font-weight: 500; box-shadow: none;}
[data-testid="stSidebar"] .stButton > button[kind="primary"], [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #2563eb, #38bdf8); color: #fff; font-weight: 600; box-shadow: 0 10px 22px rgba(37, 99, 235, .35);}
[data-testid="stSidebar"] .stButton > button p {color: inherit;}
</style>""", unsafe_allow_html=True)


# ---------- Helpers ----------
def card(name):
    return st.container(key="card_" + name)


def head(title, sub):
    st.markdown(f'<h1 class="pt">{title}</h1><p class="ps">{sub}</p>', unsafe_allow_html=True)


def kpi(col, label, value):
    col.markdown(f'<div class="kpi"><div class="kl">{label}</div><div class="kv">{value}</div></div>', unsafe_allow_html=True)


def xl(df, n=200):
    d = df.head(n).copy()
    d.insert(0, "#", range(1, len(d) + 1))
    st.markdown('<div class="xlwrap">' + d.to_html(index=False, classes="xl", border=0) + "</div>", unsafe_allow_html=True)


def bars(items):
    st.markdown("".join(
        f'<div class="brow"><span class="bl">{l}</span><div class="bt"><div class="bf" style="width:{max(w * 100, 0.6):.1f}%"></div></div>'
        f'<span class="bv">{t}</span></div>' for l, w, t in items), unsafe_allow_html=True)


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


def log(res, source):
    d = pd.DataFrame({"Time": dt.datetime.now().strftime("%H:%M:%S"), "Source": source, "Protocol": res["protocol_type"].values,
                      "Service": res["service"].values, "Result": res["Prediction"].values,
                      "Confidence": res["Confidence"].values, "Severity": res["Severity"].values})
    S.hist = pd.concat([S.hist, d], ignore_index=True)


# ---------- Sidebar ----------
S.setdefault("page", "Dashboard")
S.setdefault("hist", pd.DataFrame(columns=["Time", "Source", "Protocol", "Service", "Result", "Confidence", "Severity"]))
PAGES = {"Dashboard": ":material/space_dashboard:", "Detection": ":material/shield:",
         "Batch analysis": ":material/upload_file:", "Guide": ":material/menu_book:"}
with st.sidebar:
    st.markdown('<div class="brand"><svg width="38" height="38" viewBox="0 0 40 40"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
                '<stop offset="0" stop-color="#1d4ed8"/><stop offset="1" stop-color="#38bdf8"/></linearGradient></defs>'
                '<path d="M20 3 35 11.5v17L20 37 5 28.5v-17z" fill="url(#g)"/><path d="M20 12 28 16.5v7L20 28 12 23.5v-7z" fill="#fff" opacity=".92"/></svg>NetGuard</div>'
                '<div class="cap">Pages</div>', unsafe_allow_html=True)
    for p, ic in PAGES.items():
        if st.button(p, icon=ic, key="nav_" + p, type="primary" if S.page == p else "secondary", use_container_width=True):
            S.page = p
            st.rerun()
    st.markdown('<div class="cap">Settings</div>', unsafe_allow_html=True)
    with st.expander("Detection settings"):
        thr = st.slider("Confidence threshold", 0.30, 0.90, 0.50, 0.05,
                        help="If the model is less confident than this, the connection is marked Needs review instead of being given an attack type.")


# ---------- Dashboard ----------
def page_dashboard():
    head("Dashboard", "A summary of every connection you have checked in this session, entered by hand or uploaded as a file.")
    h = S.hist
    if h.empty:
        with card("empty"):
            st.markdown('<div class="ch">Nothing checked yet</div>Enter the details of a connection on the Detection page, '
                        'or upload a log file on the Batch analysis page. Results will appear here.', unsafe_allow_html=True)
            if st.button("Check a connection", type="primary"):
                S.page = "Detection"
                st.rerun()
        return
    threats = h[h["Result"] != "Normal"]
    top = max(h["Severity"], key=RANK.index)
    k = st.columns(4)
    kpi(k[0], "Connections checked", f"{len(h):,}")
    kpi(k[1], "Threats detected", f"{len(threats):,}")
    kpi(k[2], "Threat rate", f"{len(threats) / len(h):.1%}")
    kpi(k[3], "Highest severity", top)
    st.write("")
    a, b = st.columns([3, 2])
    with a, card("dist"):
        st.markdown('<div class="ch">Results by category</div>', unsafe_allow_html=True)
        cnt = h["Result"].value_counts().reindex(CLASSES, fill_value=0)
        bars([(c, n / len(h), f"{n:,}") for c, n in cnt.items()])
    with b, card("sev"):
        st.markdown('<div class="ch">Results by severity</div>', unsafe_allow_html=True)
        sv = h["Severity"].value_counts().reindex(RANK[::-1], fill_value=0)
        bars([(s, n / len(h), f"{n:,}") for s, n in sv.items()])
    with card("recent"):
        st.markdown('<div class="ch">Recent checks</div>', unsafe_allow_html=True)
        xl(h.tail(12).iloc[::-1].reset_index(drop=True))
        if st.button("Clear history"):
            S.hist = S.hist.iloc[0:0]
            st.rerun()


# ---------- Detection ----------
def conv(f, v):
    if f in CAT:
        return str(v)
    return int(round(float(v) * 100)) if f in PCT else int(round(float(v)))


def load_example(name):
    row = SAMPLE[SAMPLE["Attack Type"] == EXAMPLES[name]].iloc[0]
    for f in FIELDS:
        S["f_" + f] = conv(f, row[f])


def field(f):
    lab, hp, k = *FIELDS[f], "f_" + f
    if f == "protocol_type":
        st.selectbox(lab, B["opts"][f], key=k, help=hp, format_func=str.upper)
    elif f == "service":
        st.selectbox(lab, B["opts"][f], key=k, help=hp)
    elif f == "flag":
        st.selectbox(lab, B["opts"][f], key=k, help=hp, format_func=lambda v: FLAGS.get(v, v))
    elif f == "logged_in":
        st.selectbox(lab, [0, 1], key=k, help=hp, format_func=lambda v: "Yes" if v else "No")
    elif f in PCT:
        st.slider(lab, 0, 100, step=5, key=k, help=hp, format="%d%%")
    else:
        st.number_input(lab, min_value=0, step=1, key=k, help=hp)


def page_detection():
    head("Detection", "Enter the details of one network connection, for example from a firewall or server log, and the model will tell you whether it looks like an attack.")
    for f in FIELDS:
        S.setdefault("f_" + f, conv(f, B["defaults"][f]))
    with st.expander("Start from a typical record"):
        c = st.columns([2, 1, 3])
        c[0].selectbox("Record type", list(EXAMPLES), key="ex_name", label_visibility="collapsed")
        c[1].button("Fill the form", on_click=lambda: load_example(S.ex_name), use_container_width=True)
    left, right = st.columns(2)
    with left, card("conn"):
        st.markdown('<div class="ch">Connection</div>', unsafe_allow_html=True)
        for f in ["protocol_type", "service", "flag", "duration"]:
            field(f)
    with left, card("vol"):
        st.markdown('<div class="ch">Traffic volume</div>', unsafe_allow_html=True)
        for f in ["src_bytes", "dst_bytes"]:
            field(f)
    with right, card("recent"):
        st.markdown('<div class="ch">Recent activity from the same source (last 2 seconds)</div>', unsafe_allow_html=True)
        for f in ["count", "srv_count", "serror_rate", "rerror_rate", "same_srv_rate"]:
            field(f)
    with right, card("login"):
        st.markdown('<div class="ch">Login activity</div>', unsafe_allow_html=True)
        for f in ["logged_in", "num_failed_logins", "hot"]:
            field(f)
    st.caption("Details not listed here are filled with typical values, so the result is an estimate based on what you enter.")
    if st.button("Analyse connection", type="primary"):
        row = pd.DataFrame([{**B["defaults"], **{f: (S["f_" + f] / 100 if f in PCT else S["f_" + f]) for f in FIELDS}}])
        out, pr = predict(row, thr)
        log(pd.concat([row, out], axis=1), "Manual entry")
        S.last = (row, out, pr)
    if "last" in S:
        result(*S.last)


def result(row, out, pr):
    res, conf, sev = out["Prediction"][0], out["Confidence"][0], out["Severity"][0]
    title, what, todo = GUIDE[res]
    with card("result"):
        st.markdown(f'<div class="ch">Result</div><span class="pill{"" if res == "Normal" else " hot"}">Severity: {sev}</span>'
                    f'<div class="kv">{title}</div><div class="kl">Confidence {conf:.0%}</div><p style="margin-top:10px">{what}<br>'
                    f'<b>Suggested action:</b> {todo}</p>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a, card("probs"):
        st.markdown('<div class="ch">How likely each category is</div>', unsafe_allow_html=True)
        bars([(c, p, f"{p:.1%}") for c, p in pr.iloc[0].items()])
    with b, card("why"):
        st.markdown('<div class="ch">Factors the model relies on most</div>', unsafe_allow_html=True)
        top = sorted(B["importance"].items(), key=lambda x: -x[1])[:6]
        xl(pd.DataFrame([{"Factor": FIELDS[f][0] if f in FIELDS else f.replace("_", " "), "Weight": f"{v:.1%}",
                          "Your value": f"{row[f][0] * 100:.0f}%" if f in PCT else row[f][0]} for f, v in top]))


# ---------- Batch analysis ----------
def validate(df):
    X, checks = df.copy(), []
    missing = [c for c in RAW if c not in X]
    extra = [c for c in X.columns if c not in RAW]
    checks.append(["Rows", f"{len(X):,}", "None"])
    checks.append(["Missing columns", ", ".join(missing) or "None", "Filled with typical values" if missing else "None"])
    checks.append(["Extra columns", ", ".join(extra) or "None", "Ignored" if extra else "None"])
    if len(missing) > 0.25 * len(RAW):
        return None, checks
    for c in missing:
        X[c] = B["defaults"][c]
    X[NUMCOLS] = X[NUMCOLS].apply(pd.to_numeric, errors="coerce")
    n_na = int(X[RAW].isna().sum().sum())
    unk = sum(int((~X[c].isin(B["opts"][c])).sum()) for c in CAT)
    checks.append(["Empty or non-numeric cells", f"{n_na:,}", "Filled with typical values" if n_na else "None"])
    checks.append(["Unrecognised category values", f"{unk:,}", "Ignored by the model" if unk else "None"])
    return X.fillna(B["defaults"])[RAW], checks


def page_batch():
    head("Batch analysis", "Upload a CSV file of connection records and check all of them at once.")
    with card("upload"):
        c1, c2 = st.columns([3, 2])
        up = c1.file_uploader("CSV file with a header row", type="csv")
        use = c2.checkbox("Use a sample file of 500 connections")
        c2.download_button("Download the expected format", SAMPLE[RAW].head(20).to_csv(index=False), "template.csv", "text/csv")
    if use:
        data = SAMPLE[RAW].head(500)
    elif up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as e:
            st.error(f"The file could not be read: {e}")
            return
    else:
        st.info("Upload a CSV file, or tick the sample option, to begin.")
        return
    clean, checks = validate(data)
    with card("checks"):
        st.markdown('<div class="ch">File check</div>', unsafe_allow_html=True)
        xl(pd.DataFrame(checks, columns=["Check", "Result", "Action taken"]))
    if clean is None:
        st.error("Too many required columns are missing. Download the expected format and match its column names.")
        return
    out, _ = predict(clean, thr)
    res = pd.concat([clean, out], axis=1)
    k = st.columns(4)
    kpi(k[0], "Connections checked", f"{len(res):,}")
    kpi(k[1], "Threats detected", f"{int((res['Prediction'] != 'Normal').sum()):,}")
    kpi(k[2], "Critical", f"{int((res['Severity'] == 'Critical').sum()):,}")
    kpi(k[3], "Needs review", f"{int((res['Prediction'] == 'Uncertain').sum()):,}")
    st.write("")
    with card("summary"):
        st.markdown('<div class="ch">Results by category</div>', unsafe_allow_html=True)
        cnt = res["Prediction"].value_counts().reindex(CLASSES, fill_value=0)
        bars([(c, n / len(res), f"{n:,}") for c, n in cnt.items()])
    with card("table"):
        st.markdown('<div class="ch">All connections</div>', unsafe_allow_html=True)
        f1, f2 = st.columns([3, 2])
        sel = f1.multiselect("Show categories", CLASSES, default=CLASSES)
        minc = f2.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        view = res[res["Prediction"].isin(sel) & (res["Confidence"] >= minc)]
        view = view[["protocol_type", "service", "flag", "duration", "src_bytes", "dst_bytes", "Prediction", "Confidence", "Severity"]]
        view.index = view.index + 1
        st.dataframe(view, use_container_width=True, height=380)
        d1, d2 = st.columns(2)
        d1.download_button("Download results (CSV)", res.to_csv(index=False), "results.csv", "text/csv", use_container_width=True)
        if d2.button("Add to dashboard", use_container_width=True):
            log(res, "File upload")
            st.success("Added to the dashboard.")


# ---------- Guide ----------
def page_guide():
    m = B["metrics"]
    head("Guide", "What each result means and how well the model performs.")
    with card("types"):
        st.markdown('<div class="ch">Attack categories</div>', unsafe_allow_html=True)
        xl(pd.DataFrame([[c, SEV[c], GUIDE[c][1], GUIDE[c][2]] for c in CLASSES[1:5]],
                        columns=["Category", "Severity", "What it is", "Suggested action"]))
    k = st.columns(3)
    kpi(k[0], "Overall accuracy", f"{m['acc']:.1%}")
    kpi(k[1], "Balanced score (macro F1)", f"{m['macro_f1']:.3f}")
    kpi(k[2], "Inputs used by the model", m["n_features"])
    st.write("")
    with card("perf"):
        st.markdown('<div class="ch">Detection rate by category on held-out test data</div>', unsafe_allow_html=True)
        xl(pd.DataFrame({"Category": list(m["recall"]), "Test connections": list(m["support"].values()),
                         "Correctly detected": [f"{v:.1%}" for v in m["recall"].values()]}))
        st.caption("Remote access attempts and privilege escalation are detected less reliably. They are rare in the training data, "
                   "and the test data contains attack variants the model has never seen. Raise the confidence threshold "
                   "to send doubtful cases to manual review.")


{"Dashboard": page_dashboard, "Detection": page_detection, "Batch analysis": page_batch, "Guide": page_guide}[S.page]()
