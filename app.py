import datetime as dt
import io
import os

import joblib
import pandas as pd
import streamlit as st

from train_model import COLS

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
LBL = {"Normal": "Normal", "DoS": "Denial of service", "Probe": "Scanning", "R2L": "Remote access",
       "U2R": "Privilege escalation", "Uncertain": "Needs review"}
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
PCT = ["serror_rate"]
FIELDS = {
    "protocol_type": ("Protocol", "Transport protocol used by the connection."),
    "service": ("Destination service", "Service the connection was aimed at, for example http, ftp or smtp."),
    "flag": ("Connection outcome", "How the connection ended."),
    "logged_in": ("Login succeeded", "Whether the connection logged in successfully."),
    "src_bytes": ("Data sent (bytes)", "Bytes sent from the source to the destination."),
    "dst_bytes": ("Data returned (bytes)", "Bytes sent back from the destination to the source."),
    "count": ("Connections to the same host", "Connections from this source to the same destination host in the last 2 seconds."),
    "serror_rate": ("Half-open connections", "Share of those recent connections that never finished the handshake."),
}
EXAMPLES = {"Ordinary traffic": "Normal", "Flood-like traffic": "DoS", "Scanning-like traffic": "Probe",
            "Remote login attempt": "R2L", "Typical values (reset)": None}

st.markdown("""<style>
html, body, [class*="css"] {font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stAppViewContainer"] {background: radial-gradient(circle at 0% 0%, #cfe1ff 0, transparent 38%),
  radial-gradient(circle at 100% 8%, #d4f0fb 0, transparent 34%), #f5f9ff;}
.block-container {padding-top: 2.4rem; max-width: 1180px;}
[data-testid="stSidebar"] {background: #fff; border-right: 1px solid #e3ecfa;}
.brand {display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 700; color: #0b1e4f; margin: 10px 0 26px;}
.cap {font-size: 13px; color: #7b8db0; margin: 18px 0 8px 4px;}
.pt {font-size: 28px; font-weight: 700; color: #0b1e4f; letter-spacing: -.01em; margin: 0 0 4px;}
.ps {font-size: 15px; color: #52637f; margin: 0 0 22px;}
[class*="st-key-card_"] {background: #fff; border: 1px solid #e0e9f8; border-radius: 22px; padding: 20px 24px;
  box-shadow: 0 10px 30px rgba(37, 99, 235, .07); margin-bottom: 10px;}
.ch {font-weight: 600; color: #0b1e4f; margin-bottom: 8px; font-size: 16px;}
.kpi {background: #fff; border: 1px solid #e0e9f8; border-radius: 22px; padding: 18px 22px; box-shadow: 0 10px 30px rgba(37, 99, 235, .07);}
.kl {font-size: 14px; color: #52637f;}
.kv {font-size: 28px; font-weight: 700; color: #0b1e4f; margin-top: 4px;}
.pill {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; background: #e0ecff; color: #1d4ed8;}
.pill.hot {background: linear-gradient(135deg, #1d4ed8, #38bdf8); color: #fff;}
.brow {display: flex; align-items: center; gap: 12px; margin: 9px 0; font-size: 14px; color: #0b1e4f;}
.bl {width: 150px;} .bv {width: 70px; text-align: right; color: #52637f;}
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
[data-testid="stSidebar"] .stButton > button {justify-content: flex-start !important; text-align: left; border: 0; background: transparent;
  color: #3b5583; border-radius: 16px; padding: .65rem 1rem; font-weight: 500; box-shadow: none;}
[data-testid="stSidebar"] .stButton > button > div {justify-content: flex-start !important; width: 100%;}
[data-testid="stSidebar"] .stButton > button p {text-align: left; color: inherit;}
[data-testid="stSidebar"] .stButton > button[kind="primary"], [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #2563eb, #38bdf8); color: #fff; font-weight: 600; box-shadow: 0 10px 22px rgba(37, 99, 235, .35);}
</style>""", unsafe_allow_html=True)


st.markdown("""<style>
@import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap");
.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp li, .stApp td, .stApp th,
.kpi, .pt, .ps, .ch, .kv, .brand {font-family: "Plus Jakarta Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;}
header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none;}
.block-container {padding-top: 1.6rem; max-width: 1180px;}
.st-key-topbar {background: #fff; border: 1px solid #e0e9f8; border-radius: 22px; padding: 10px 18px;
  box-shadow: 0 10px 30px rgba(37, 99, 235, .07); margin-bottom: 30px;}
.st-key-topbar .brand {margin: 0; font-size: 22px;}
.st-key-topbar .stButton > button, .st-key-topbar [data-testid="stPopover"] button {border-radius: 14px; border: 0; background: transparent;
  color: #3b5583; font-weight: 600; padding: .55rem 1rem; box-shadow: none;}
.st-key-topbar .stButton > button p {color: inherit;}
.st-key-topbar .stButton > button[kind="primary"], .st-key-topbar .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #2563eb, #38bdf8); color: #fff; box-shadow: 0 8px 18px rgba(37, 99, 235, .32);}
.st-key-topbar [data-testid="stPopover"] button {border: 1px solid #d6e2f6; background: #f5f9ff; color: #1e40af;}
.hd {display: flex; align-items: center; gap: 16px; margin: 0 0 26px;}
.hi {flex: none; width: 54px; height: 54px; border-radius: 17px; background: linear-gradient(135deg, #2563eb, #38bdf8);
  display: flex; align-items: center; justify-content: center; box-shadow: 0 10px 22px rgba(37, 99, 235, .28);}
.pt {font-size: 30px; font-weight: 700; letter-spacing: -.02em; line-height: 1.15; margin: 0;}
.ps {font-size: 15px; line-height: 1.5; color: #52637f; margin: 4px 0 0; max-width: 720px;}
.ch {display: flex; align-items: center; gap: 10px; font-size: 16px; font-weight: 700; color: #0b1e4f; margin-bottom: 10px;}
.ch::before {content: ''; width: 4px; height: 16px; border-radius: 3px; background: linear-gradient(180deg, #2563eb, #38bdf8);}
.kv {font-variant-numeric: tabular-nums;}
</style>""", unsafe_allow_html=True)


# ---------- Helpers ----------
def card(name):
    return st.container(key="card_" + name)


ICONS = {
    "Detection": '<path d="M12 3l8 3v6c0 5-3.4 8-8 9-4.6-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/>',
    "Batch analysis": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M12 17v-6M12 11l-2.5 2.5M12 11l2.5 2.5"/>',
    "Dashboard": '<rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.5"/>',
}


def head(title, sub):
    st.markdown(f'<div class="hd"><div class="hi"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" '
                f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONS[title]}</svg></div>'
                f'<div><div class="pt">{title}</div><div class="ps">{sub}</div></div></div>', unsafe_allow_html=True)


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


@st.cache_data(show_spinner="Analysing connections...")
def predict_cached(df, thr):
    return predict(df, thr)[0]


def log(res, source):
    d = pd.DataFrame({"Time": dt.datetime.now().strftime("%H:%M:%S"), "Source": source, "Protocol": res["protocol_type"].values,
                      "Service": res["service"].values, "Result": res["Prediction"].values,
                      "Confidence": res["Confidence"].values, "Severity": res["Severity"].values})
    S.hist = pd.concat([S.hist, d], ignore_index=True)


# ---------- Top bar ----------
PAGES = {"Detection": ":material/shield:", "Batch analysis": ":material/upload_file:", "Dashboard": ":material/space_dashboard:"}
if S.get("page") not in PAGES:
    S.page = "Detection"
S.setdefault("hist", pd.DataFrame(columns=["Time", "Source", "Protocol", "Service", "Result", "Confidence", "Severity"]))
S.setdefault("added", set())
BRAND = ('<div class="brand"><svg width="34" height="34" viewBox="0 0 40 40"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
         '<stop offset="0" stop-color="#1d4ed8"/><stop offset="1" stop-color="#38bdf8"/></linearGradient></defs>'
         '<path d="M20 3 35 11.5v17L20 37 5 28.5v-17z" fill="url(#g)"/><path d="M20 12 28 16.5v7L20 28 12 23.5v-7z" fill="#fff" opacity=".92"/></svg>NetGuard</div>')
with st.container(key="topbar"):
    cols = st.columns([2.2, 1.4, 1.8, 1.5, 2.0, 1.4], vertical_alignment="center")
    cols[0].markdown(BRAND, unsafe_allow_html=True)
    for c, (p, ic) in zip(cols[1:4], PAGES.items()):
        if c.button(p, icon=ic, key="nav_" + p, type="primary" if S.page == p else "secondary", use_container_width=True):
            S.page = p
            st.rerun()
    with cols[5].popover("Settings", icon=":material/tune:", use_container_width=True):
        thr = st.slider("Confidence threshold", 0.30, 0.90, 0.50, 0.05,
                        help="If the model is less confident than this, the connection is marked Needs review instead of being given an attack type.")


# ---------- Check connection ----------
def conv(f, v):
    if f in CAT:
        return str(v)
    return int(round(float(v) * 100)) if f in PCT else int(round(float(v)))


def load_example(name):
    cls = EXAMPLES[name]
    if cls is None:
        S.pop("hidden", None)
        src = B["defaults"]
    else:
        src = SAMPLE[SAMPLE["Attack Type"] == cls].iloc[0]
        S.hidden = {c: src[c] for c in RAW}
    for f in FIELDS:
        S["f_" + f] = conv(f, src[f])


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


def show(f, v):
    if f in PCT:
        return f"{v * 100:.0f}%"
    if f == "flag":
        return FLAGS.get(v, v)
    if f == "logged_in":
        return "Yes" if v else "No"
    if f == "protocol_type":
        return str(v).upper()
    return f"{v:,.0f}" if pd.api.types.is_number(v) else v


def result(row, out, pr):
    res, conf, sev = out["Prediction"][0], out["Confidence"][0], out["Severity"][0]
    title, what, todo = GUIDE[res]
    with card("result"):
        st.markdown(f'<div class="ch">Result</div><span class="pill{"" if res == "Normal" else " hot"}">Severity: {sev}</span>'
                    f'<div class="kv">{title}</div><div class="kl">Confidence {conf:.0%}</div>'
                    f'<p style="margin:10px 0 4px">{what}</p><p><b>Suggested action:</b> {todo}</p>'
                    '<div class="ch" style="margin-top:16px">Likelihood by category</div>', unsafe_allow_html=True)
        bars([(LBL[c], p, f"{p:.1%}") for c, p in pr.iloc[0].items()])
        st.markdown('<div class="ch" style="margin-top:16px">What influenced this most</div>', unsafe_allow_html=True)
        top = [f for f, _ in sorted(B["importance"].items(), key=lambda x: -x[1]) if f in FIELDS][:4]
        xl(pd.DataFrame([{"Detail": FIELDS[f][0], "Your value": show(f, row[f][0])} for f in top]))


def page_check():
    head("Detection", "Enter the details of one network connection, for example from a firewall or server log, and the model will tell you whether it looks like an attack.")
    for f in FIELDS:
        S.setdefault("f_" + f, conv(f, B["defaults"][f]))
    with st.expander("Start from an example"):
        c = st.columns([2, 1, 3])
        c[0].selectbox("Example", list(EXAMPLES), key="ex_name", label_visibility="collapsed")
        c[1].button("Fill the form", on_click=lambda: load_example(S.ex_name), use_container_width=True)
    left, right = st.columns([3, 2], gap="large")
    with left, card("form"):
        st.markdown('<div class="ch">Connection details</div>', unsafe_allow_html=True)
        for pair in [("protocol_type", "service"), ("flag", "logged_in"), ("src_bytes", "dst_bytes"), ("count", "serror_rate")]:
            a, b = st.columns(2)
            with a:
                field(pair[0])
            with b:
                field(pair[1])
        st.caption("Anything not listed here is filled in with typical values, or taken from the example you loaded.")
        if st.button("Analyse connection", type="primary", use_container_width=True):
            base = S.get("hidden", B["defaults"])
            row = pd.DataFrame([{**base, **{f: (S["f_" + f] / 100 if f in PCT else S["f_" + f]) for f in FIELDS}}])
            out, pr = predict(row, thr)
            log(pd.concat([row, out], axis=1), "Manual entry")
            S.last = (row, out, pr)
    with right:
        if "last" in S:
            result(*S.last)
        else:
            with card("result"):
                st.markdown('<div class="ch">Result</div>Fill in the details and select Analyse connection. The result appears here.',
                            unsafe_allow_html=True)


# ---------- Check file ----------
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


def read_upload(f):
    """Read a CSV that may have a header row, no header row, or every line wrapped in quotes."""
    lines = []
    for ln in f.getvalue().decode("utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if ln.startswith('"') and ln.endswith('"') and '"' not in ln[1:-1]:
            ln = ln[1:-1]
        if ln:
            lines.append(ln)
    text = "\n".join(lines)
    df = pd.read_csv(io.StringIO(text))
    if any(c in df.columns for c in RAW):
        return df, None
    df = pd.read_csv(io.StringIO(text), header=None)
    if df.shape[1] not in (41, 42, 43):
        raise ValueError(f"expected 41 to 43 columns but found {df.shape[1]}")
    df.columns = COLS[:df.shape[1]]
    return df, "No header row was found, so the standard column order was assumed."


def page_file():
    head("Batch analysis", "Upload a CSV file of connection records and check all of them at once.")
    with card("upload"):
        c1, c2 = st.columns([3, 2])
        up = c1.file_uploader("CSV file", type="csv")
        use = c2.checkbox("Use a sample file of 500 connections")
        c2.download_button("Download the expected format", SAMPLE[RAW].head(20).to_csv(index=False), "template.csv", "text/csv")
    if use:
        data, key = SAMPLE[RAW].head(500), "sample"
    elif up is not None:
        try:
            data, note = read_upload(up)
        except Exception as e:
            st.error(f"The file could not be read: {e}")
            return
        key = f"{up.name}:{up.size}"
        if note:
            st.info(note)
    else:
        st.info("Upload a CSV file, or tick the sample option, to begin.")
        return
    clean, checks = validate(data)
    with st.expander("File check", expanded=clean is None or any(c[2] != "None" for c in checks)):
        xl(pd.DataFrame(checks, columns=["Check", "Result", "Action taken"]))
    if clean is None:
        st.error("Too many required columns are missing. Download the expected format and match its column names.")
        return
    res = pd.concat([clean, predict_cached(clean, thr)], axis=1)
    k = st.columns(4)
    kpi(k[0], "Connections checked", f"{len(res):,}")
    kpi(k[1], "Threats detected", f"{int((res['Prediction'] != 'Normal').sum()):,}")
    kpi(k[2], "Critical", f"{int((res['Severity'] == 'Critical').sum()):,}")
    kpi(k[3], "Needs review", f"{int((res['Prediction'] == 'Uncertain').sum()):,}")
    st.write("")
    with card("summary"):
        st.markdown('<div class="ch">Results by category</div>', unsafe_allow_html=True)
        cnt = res["Prediction"].value_counts().reindex(CLASSES, fill_value=0)
        bars([(LBL[c], n / len(res), f"{n:,}") for c, n in cnt.items()])
    with card("table"):
        st.markdown('<div class="ch">All connections</div>', unsafe_allow_html=True)
        f1, f2 = st.columns([3, 2])
        sel = f1.multiselect("Show categories", CLASSES, default=CLASSES, format_func=LBL.get)
        minc = f2.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        view = res[res["Prediction"].isin(sel) & (res["Confidence"] >= minc)]
        view = view[["protocol_type", "service", "flag", "src_bytes", "dst_bytes", "Prediction", "Confidence", "Severity"]]
        view = view.assign(Prediction=view["Prediction"].map(LBL))
        view.index = view.index + 1
        st.dataframe(view, use_container_width=True, height=380)
        d1, d2 = st.columns(2)
        d1.download_button("Download results (CSV)", res.assign(Prediction=res["Prediction"].map(LBL)).to_csv(index=False),
                           "results.csv", "text/csv", use_container_width=True)
        done = key in S.added
        if d2.button("Added to summary" if done else "Add to summary", disabled=done, use_container_width=True):
            log(res, "File upload")
            S.added.add(key)
            st.rerun()


# ---------- Summary ----------
def page_summary():
    head("Dashboard", "A summary of every connection you have checked in this session, entered by hand or uploaded as a file.")
    h = S.hist
    if h.empty:
        with card("empty"):
            st.markdown('<div class="ch">Nothing checked yet</div>Check a connection or upload a file, and the results will be summarised here.',
                        unsafe_allow_html=True)
            if st.button("Check a connection", type="primary"):
                S.page = "Detection"
                st.rerun()
        return
    threats = h[h["Result"] != "Normal"]
    k = st.columns(4)
    kpi(k[0], "Connections checked", f"{len(h):,}")
    kpi(k[1], "Threats detected", f"{len(threats):,}")
    kpi(k[2], "Threat rate", f"{len(threats) / len(h):.1%}")
    kpi(k[3], "Highest severity", max(h["Severity"].unique(), key=RANK.index))
    st.write("")
    a, b = st.columns([3, 2])
    with a, card("dist"):
        st.markdown('<div class="ch">Results by category</div>', unsafe_allow_html=True)
        cnt = h["Result"].value_counts().reindex(CLASSES, fill_value=0)
        bars([(LBL[c], n / len(h), f"{n:,}") for c, n in cnt.items()])
    with b, card("sev"):
        st.markdown('<div class="ch">Results by severity</div>', unsafe_allow_html=True)
        sv = h["Severity"].value_counts().reindex(RANK[::-1], fill_value=0)
        bars([(s, n / len(h), f"{n:,}") for s, n in sv.items()])
    with card("recent"):
        st.markdown('<div class="ch">Recent checks</div>', unsafe_allow_html=True)
        t = h.tail(12).iloc[::-1].reset_index(drop=True)
        xl(t.assign(Result=t["Result"].map(LBL)))
        if st.button("Clear history"):
            S.hist, S.added = S.hist.iloc[0:0], set()
            st.rerun()


{"Detection": page_check, "Batch analysis": page_file, "Dashboard": page_summary}[S.page]()
