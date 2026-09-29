import datetime as dt
import io
import os

import joblib
import pandas as pd
import streamlit as st

from train_model import COLS

APP_NAME = "NetGuard"  # change the site name here; it is used in the tab title and the top bar
# One set of names and descriptions, shared by the top-bar buttons, the home cards and each page header.
PAGE_INFO = {
    "Detection": ("Threat Detection", "Classify a connection and see its severity, confidence, and suggested action."),
    "Batch analysis": ("Batch Analysis", "Upload network logs and inspect all predicted results together."),
    "Dashboard": ("Threat Dashboard", "Track detected threats, severity, and recent checks."),
}
_pg = st.session_state.get("page")
st.set_page_config(page_title=f"{PAGE_INFO[_pg][0]} · {APP_NAME}" if _pg in PAGE_INFO else f"{APP_NAME} · Cyber Threat Monitoring",
                   layout="wide")

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
.stApp, .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp li, .stApp td, .stApp th,
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


st.markdown("""<style>
.stApp, [data-testid="stAppViewContainer"] {background: radial-gradient(circle at 85% 3%, #17254a 0%, transparent 34%), radial-gradient(circle at 3% 68%, #211638 0%, transparent 36%), #0d0b1c; color: #f8f8ff;}
.block-container {max-width: 1240px; padding-top: 1.4rem; padding-bottom: 5rem;}
.st-key-topbar {background: #111123; border: 1px solid #263152; border-radius: 0; padding: 10px 22px; box-shadow: 0 0 26px #22d3ee15;}
.st-key-topbar .brand {color: white;} .st-key-topbar .stButton > button {color: #c4c9df;}
.st-key-topbar .stButton > button[kind="primary"], .st-key-topbar .stButton > button[data-testid="stBaseButton-primary"] {background: linear-gradient(105deg,#8b2bfa,#43a9f4);}
[class*="st-key-card_"] {background: linear-gradient(145deg,#202b48,#171a35); border: 1px solid #334467; border-radius: 13px; box-shadow: 0 10px 30px #05050d55;}
.kpi {background: linear-gradient(145deg,#212c4b,#171b35); border: 1px solid #334467; border-radius: 13px; box-shadow: none;}
.pt,.ch,.kv,.brand {color: #fff;} .ps,.kl,.bv {color: #b7bfd6;}
.ch::before {background: linear-gradient(#9e3cf5,#23d6e5);}
.hi,.stButton > button[kind="primary"],.stButton > button[data-testid="stBaseButton-primary"] {background: linear-gradient(105deg,#9031f5,#43a9f4);}
.pill {background:#293759;color:#91e9fa;} .pill.hot {background:linear-gradient(105deg,#8e30f4,#38bdf8);}
.bt {background:#11172b;} .bf {background:linear-gradient(90deg,#902ffa,#26d8e7);}
table.xl th {background:#29395f;border-color:#435475;} table.xl td {color:#f0f3ff;border-color:#35445f;} table.xl td:first-child,table.xl tr:nth-child(even) td {background:#1a2441 !important;color:#d0daef;} table.xl tr:nth-child(odd) td {background:#141a31;}
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p, .stCaption p {color:inherit;}
[data-testid="stSelectbox"] > div > div, [data-testid="stNumberInput"] input, [data-testid="stFileUploader"], [data-testid="stMultiSelect"] > div > div {background:#161d35;color:#f8f8ff;border-color:#3b4b70;}
[data-testid="stDataFrame"], [data-testid="stExpander"] {border-color:#394868;}
.hero {min-height:390px;display:flex;align-items:center;padding:30px 2% 40px;gap:3%;}
.hero-copy {flex:1.05}.hero-art {flex:.95; min-height:300px;position:relative;display:grid;place-items:center;}
.eyebrow {font-weight:700;font-size:12px;letter-spacing:.24em;color:#24d7d2;margin-bottom:13px;}
.hero h1 {font-size:clamp(34px,4.3vw,63px);line-height:1.12;letter-spacing:-.04em;color:#fff;margin:0 0 22px;font-weight:800;}
.hero p {max-width:540px;color:#b9c0d7;font-size:16px;line-height:1.7;}
.cyber-orb {width:min(340px,75%);aspect-ratio:1;border-radius:30%;display:grid;place-items:center;transform:rotate(-12deg);background:radial-gradient(circle at 50% 40%,#19dded55,transparent 60%),linear-gradient(140deg,#6437f355,#0b2c4b);border:2px solid #2ad9f0;box-shadow:0 0 45px #1accf44d, inset 0 0 65px #932ef655;}
.cyber-orb svg {width:66%;filter:drop-shadow(0 0 22px #20d7ed);transform:rotate(12deg)}
.hero-art:before,.hero-art:after {content:"";position:absolute;width:70%;height:70%;border:1px solid #40d7fa55;transform:rotate(45deg);z-index:0}.hero-art:after {width:53%;height:53%;border-color:#a43bf999;transform:rotate(20deg)} .cyber-orb {z-index:1}
.section-intro{text-align:center;padding:20px 0 14px}.section-intro h2{color:#fff;font-size:34px;margin:0 0 10px}.section-intro p{color:#abb5cc;max-width:650px;margin:auto;line-height:1.65}
.service-card {background:linear-gradient(145deg,#243253,#19213d);border:1px solid #354b72;border-radius:15px;padding:28px 20px;min-height:180px;text-align:center;box-shadow:0 15px 30px #05051055;margin-bottom:16px}
.service-icon{font-size:39px;margin-bottom:12px;filter:drop-shadow(0 0 15px #4ec6ff88)}.service-card h3{font-size:17px;color:#fff;margin:5px 0 9px}.service-card p{color:#b6c2d9;font-size:13px;line-height:1.5;margin:0}
@media(max-width:720px){.hero{flex-direction:column;padding:35px 1%}.hero-art{width:100%;min-height:250px}.hero h1{font-size:35px}.st-key-topbar{padding:10px}.st-key-topbar .brand{font-size:15px}}
</style>""", unsafe_allow_html=True)


# ---------- Helpers ----------
def card(name):
    return st.container(key="card_" + name)


ARTWORK = {
    "Detection": '<path d="M22 4 38 10v13c0 12-7 20-16 25C13 43 6 35 6 23V10z"/><path d="m15 25 5 5 10-12"/>',
    "Batch analysis": '<path d="M4 14V9a3 3 0 0 1 3-3h12l5 6h18a3 3 0 0 1 3 3v4"/><path d="M7 18h36l-4 25H5L2 22a4 4 0 0 1 5-4z"/>',
    "Dashboard": '<rect x="5" y="5" width="38" height="38" rx="4"/><path d="M13 34V23m9 11V13m9 21V20m9 14V11"/>',
}


def head(page):
    title, sub = PAGE_INFO[page]
    st.markdown(f'<div class="hd"><div class="hi"><svg width="26" height="26" viewBox="0 0 48 48" fill="none" stroke="#fff" '
                f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round">{ARTWORK[page]}</svg></div>'
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
PAGES = {"Home": ":material/home:", "Detection": ":material/verified_user:", "Batch analysis": ":material/folder_open:", "Dashboard": ":material/bar_chart:"}
if S.get("page") not in PAGES:
    S.page = "Home"
S.setdefault("hist", pd.DataFrame(columns=["Time", "Source", "Protocol", "Service", "Result", "Confidence", "Severity"]))
S.setdefault("added", set())
BRAND = ('<div class="brand"><svg width="34" height="34" viewBox="0 0 40 40"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
         '<stop offset="0" stop-color="#9134f7"/><stop offset="1" stop-color="#24d5e7"/></linearGradient></defs>'
         '<path d="M20 3 35 11.5v17L20 37 5 28.5v-17z" fill="url(#g)"/><path d="M20 12 28 16.5v7L20 28 12 23.5v-7z" fill="#fff" opacity=".92"/></svg>' + APP_NAME + '</div>')
with st.container(key="topbar"):
    cols = st.columns([1.5, 0.95, 1.6, 1.5, 1.6, 1.25], gap="small", vertical_alignment="center")
    cols[0].markdown(BRAND, unsafe_allow_html=True)
    for c, (p, ic) in zip(cols[1:5], PAGES.items()):
        if c.button(PAGE_INFO[p][0] if p in PAGE_INFO else p, icon=ic, key="nav_" + p, type="primary" if S.page == p else "secondary", use_container_width=True):
            S.page = p
            st.rerun()
    with cols[5].popover("Settings", icon=":material/tune:", use_container_width=True):
        st.selectbox("Appearance", ["Dark", "Light"], key="appearance", help="Choose the app theme independently of your device theme.")
        thr = st.slider("Confidence threshold", 0.30, 0.90, 0.50, 0.05,
                        help="If the model is less confident than this, the connection is marked Needs review instead of being given an attack type.")


# Explicit widget colours keep contrast correct even when the browser's theme differs.
LIGHT = S.get("appearance", "Dark") == "Light"
palette = {
    "bg": "#f4f7ff" if LIGHT else "#0d0b1c",
    "surface": "#ffffff" if LIGHT else "#151a31",
    "card": "#ffffff" if LIGHT else "#202a48",
    "border": "#d7e0ef" if LIGHT else "#344664",
    "text": "#172342" if LIGHT else "#f7f8ff",
    "muted": "#4c5d79" if LIGHT else "#b9c4db",
    "input": "#f8faff" if LIGHT else "#1a2440",
    "track": "#d3dff2" if LIGHT else "#11172b",
    "accent": "#0e7490" if LIGHT else "#24d7d2",
    "icon": "#0e7490" if LIGHT else "#55d7ed",
    "glow": "#0891b224" if LIGHT else "#43d5e733",
    "tagbg": "#dbeafe" if LIGHT else "#293759",
    "tagfg": "#1e3a8a" if LIGHT else "#91e9fa",
    "thbg": "#dbe7fb" if LIGHT else "#29395f",
    "thfg": "#0f1f45" if LIGHT else "#ffffff",
    "row1": "#ffffff" if LIGHT else "#141a31",
    "row2": "#f1f6ff" if LIGHT else "#1a2441",
    "idx": "#e6eefb" if LIGHT else "#1a2441",
    "idxfg": "#4c5d79" if LIGHT else "#d0daef",
    "tdborder": "#d0dcef" if LIGHT else "#35445f",
}
ROOT = ";".join(f"--app-{k}:{v}" for k, v in palette.items()) + ";color-scheme:" + ("light" if LIGHT else "dark")
st.markdown("""<style>
.stApp,[data-testid="stAppViewContainer"] {background:var(--app-bg)!important;color:var(--app-text)!important;}
.st-key-topbar,[class*="st-key-card_"] {background:var(--app-surface)!important;border-color:var(--app-border)!important;}
.kpi,.service-card {background:var(--app-card)!important;border-color:var(--app-border)!important;}
.pt,.ch,.kv,.brand,.section-intro h2,.service-card h3,.hero h1 {color:var(--app-text)!important;}
.ps,.kl,.bv,.section-intro p,.service-card p,.hero p {color:var(--app-muted)!important;}
.st-key-topbar .stButton > button {white-space:nowrap!important;min-width:0!important;padding:.55rem .35rem!important;font-size:clamp(11px,1.15vw,15px)!important;}
.st-key-topbar .stButton > button p {white-space:nowrap!important;overflow:visible!important;text-overflow:clip!important;}
.st-key-topbar .stButton > button[kind="secondary"] {border:1px solid transparent!important;background:transparent!important;color:var(--app-muted)!important;}
.st-key-topbar [data-testid="stPopover"] button {background:var(--app-input)!important;color:var(--app-text)!important;border:1px solid var(--app-border)!important;white-space:nowrap!important;}
.stButton > button[kind="secondary"],.stButton > button[data-testid="stBaseButton-secondary"],.stDownloadButton > button {background:var(--app-input)!important;border:1px solid var(--app-border)!important;color:var(--app-text)!important;box-shadow:none!important;}
.stButton > button[kind="secondary"] p,.stDownloadButton button p,.st-key-topbar [data-testid="stPopover"] button p {color:inherit!important;}
.stButton > button[kind="primary"],.stButton > button[data-testid="stBaseButton-primary"] {background:linear-gradient(105deg,#8e30f4,#389ff1)!important;color:#fff!important;border:0!important;}
.stButton > button[kind="primary"] p {color:#fff!important;}
.stButton > button:hover,.stDownloadButton > button:hover {border-color:#49c7e8!important;filter:brightness(1.08);}
[data-testid="stWidgetLabel"] p,[data-testid="stMarkdownContainer"] p,[data-testid="stCaptionContainer"] p {color:var(--app-text);}
[data-testid="stSelectbox"] > div > div,[data-testid="stNumberInput"] input,[data-testid="stMultiSelect"] > div > div,[data-testid="stFileUploader"], [data-testid="stPopoverBody"] {background:var(--app-input)!important;color:var(--app-text)!important;border-color:var(--app-border)!important;}
[data-testid="stExpander"], [data-testid="stDataFrame"] {background:var(--app-surface)!important;border-color:var(--app-border)!important;color:var(--app-text)!important;}
.st-key-topbar {border-radius:24px!important;padding:14px 22px!important;overflow:visible;}
.st-key-topbar [data-testid="stHorizontalBlock"] {align-items:center!important;}
.st-key-topbar [data-testid="column"] {min-width:0!important;display:flex;align-items:center;justify-content:center;}
.st-key-topbar [data-testid="column"]:first-child {justify-content:flex-start;}
.st-key-topbar [data-testid="column"] > div {width:100%;}
.st-key-topbar .brand {display:flex;align-items:center;gap:9px;margin:0!important;white-space:nowrap;line-height:1;}
.st-key-topbar .stButton,.st-key-topbar [data-testid="stPopover"] {width:100%;}
.st-key-topbar .stButton > button,.st-key-topbar [data-testid="stPopover"] > button {width:100%!important;min-height:48px!important;height:48px!important;display:flex!important;align-items:center!important;justify-content:center!important;gap:7px!important;margin:0!important;border-radius:15px!important;line-height:1!important;}
.st-key-topbar .stButton > button > div,.st-key-topbar [data-testid="stPopover"] button > div {display:flex!important;align-items:center!important;justify-content:center!important;gap:7px!important;min-width:0!important;width:auto!important;}
.st-key-topbar .stButton > button p,.st-key-topbar [data-testid="stPopover"] button p {margin:0!important;white-space:nowrap!important;overflow:visible!important;text-overflow:clip!important;line-height:1.15!important;}
.st-key-topbar .stButton > button svg,.st-key-topbar [data-testid="stPopover"] button svg {flex:none;width:18px;height:18px;}
.service-icon svg {width:44px;height:44px;fill:none;stroke:#55d7ed;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 11px #43d5e766)}
.service-card {height:190px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:22px 16px;}
.service-card h3 {margin:9px 0 8px}.service-card p {min-height:42px;}
@media(max-width:900px){.block-container{padding-left:1rem;padding-right:1rem}.st-key-topbar .brand{font-size:17px}.st-key-topbar .brand svg{width:26px;height:26px}.st-key-topbar .stButton > button{font-size:11px!important;padding:.4rem .15rem!important}.st-key-topbar [data-testid="stPopover"] button{font-size:11px!important;padding:.4rem .15rem!important}}
</style>""".replace("</style>", "\n:root {" + ROOT + "}\n</style>"), unsafe_allow_html=True)

# Contrast fixes: every text/background pair below comes from the palette, so it stays readable in Dark and Light.
st.markdown("""<style>
.brow, .bl {color:var(--app-text)!important;}
.bt {background:var(--app-track)!important;}
.eyebrow {color:var(--app-accent)!important;}
.pill:not(.hot) {background:var(--app-tagbg)!important;color:var(--app-tagfg)!important;}
.pill.hot {background:linear-gradient(105deg,#7a2bea,#1f7ae0)!important;color:#fff!important;}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {background:linear-gradient(105deg,#7f2df0,#2f86e8)!important;}

table.xl th {background:var(--app-thbg)!important;color:var(--app-thfg)!important;border-color:var(--app-tdborder)!important;}
table.xl td {color:var(--app-text)!important;border-color:var(--app-tdborder)!important;}
table.xl tbody tr:nth-child(odd) td {background:var(--app-row1)!important;}
table.xl tbody tr:nth-child(even) td {background:var(--app-row2)!important;}
table.xl tbody tr td:first-child {background:var(--app-idx)!important;color:var(--app-idxfg)!important;}

/* Home cards: no filter (it drew a square box behind the first icon); glow comes from a soft gradient instead */
.service-icon {filter:none!important;width:68px;height:68px;margin-bottom:6px;display:grid;place-items:center;border-radius:50%;
  background:radial-gradient(circle,var(--app-glow) 0,transparent 70%);}
.service-icon svg {width:44px;height:44px;stroke:var(--app-icon)!important;filter:none!important;}
.service-card {height:210px;}
[data-testid="stHeaderActionElements"], .section-intro h2 a, .service-card h3 a, .hero h1 a {display:none!important;}

/* Top bar: keep the site name on the same line as the buttons */
.st-key-topbar [data-testid="stHorizontalBlock"] {align-items:center!important;flex-wrap:nowrap!important;}
.st-key-topbar [data-testid="stColumn"], .st-key-topbar [data-testid="column"] {min-width:0!important;display:flex!important;align-items:center!important;justify-content:center!important;}
.st-key-topbar [data-testid="stColumn"]:first-child, .st-key-topbar [data-testid="column"]:first-child {justify-content:flex-start!important;}
.st-key-topbar [data-testid="stColumn"] > div, .st-key-topbar [data-testid="column"] > div {width:100%;}
.st-key-topbar [data-testid="stElementContainer"], .st-key-topbar [data-testid="stMarkdown"], .st-key-topbar [data-testid="stMarkdownContainer"] {margin:0!important;padding:0!important;}
.st-key-topbar .brand {min-height:48px;margin:0!important;padding:0!important;}

/* Streamlit's own widgets */
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] label * {color:var(--app-text)!important;}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {background:var(--app-surface)!important;border-color:var(--app-border)!important;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {color:var(--app-text)!important;}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] {color:var(--app-muted)!important;}
[data-testid="stSliderThumbValue"] {color:var(--app-text)!important;}
[data-testid="stFileUploaderDropzone"] {background:var(--app-input)!important;border:1px dashed var(--app-border)!important;}
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploaderFile"] * {color:var(--app-muted)!important;}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button * {color:var(--app-text)!important;}
[data-testid="stAlert"], [data-testid="stAlertContainer"] {background:var(--app-input)!important;border:1px solid var(--app-border)!important;border-radius:12px;}
[data-testid="stAlert"] *, [data-testid="stAlertContainer"] * {color:var(--app-text)!important;}
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {background:var(--app-input)!important;border-color:var(--app-border)!important;}
[data-baseweb="select"] *, [data-baseweb="input"] input, [data-baseweb="base-input"] input, [data-testid="stNumberInput"] input
  {color:var(--app-text)!important;-webkit-text-fill-color:var(--app-text)!important;}
[data-testid="stNumberInput"] button {background:var(--app-input)!important;color:var(--app-text)!important;border-color:var(--app-border)!important;}
[data-baseweb="tag"] {background:var(--app-tagbg)!important;}
[data-baseweb="tag"], [data-baseweb="tag"] * {color:var(--app-tagfg)!important;-webkit-text-fill-color:var(--app-tagfg)!important;}
[data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"] {background:var(--app-surface)!important;border:1px solid var(--app-border);}
[data-baseweb="menu"] li, ul[role="listbox"] li, [role="option"] {background:transparent!important;}
[data-baseweb="menu"] li, [data-baseweb="menu"] li *, [role="option"], [role="option"] * {color:var(--app-text)!important;}
[role="option"]:hover, [role="option"][aria-selected="true"], [data-baseweb="menu"] li:hover {background:var(--app-tagbg)!important;}
[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] * {background:var(--app-surface)!important;color:var(--app-text)!important;}
</style>""", unsafe_allow_html=True)


# ---------- Home ----------
def page_home():
    st.markdown('''<section class="hero"><div class="hero-copy"><div class="eyebrow">CYBER THREAT MONITORING</div>
    <h1>PROTECT YOUR DATA<br>FROM CYBER ATTACKS</h1>
    <p>Analyse network connections, identify suspicious activity, and monitor threats in one place with NetGuard.</p>
    </div><div class="hero-art"><div class="cyber-orb"><svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M60 8 103 25v32c0 31-18 49-43 56C35 106 17 88 17 57V25L60 8Z" fill="#152d58" stroke="#47dcf4" stroke-width="5"/><path d="m41 59 13 13 27-30" stroke="#51eff0" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/><circle cx="60" cy="60" r="48" stroke="#a84ef9" stroke-opacity=".65" stroke-dasharray="4 8"/></svg></div></div></section>''', unsafe_allow_html=True)
    a, b, spare = st.columns([1.35, 1.35, 4.3])
    if a.button("Analyse a connection →", type="primary", use_container_width=True):
        S.page = "Detection"
        st.rerun()
    if b.button("View dashboard", use_container_width=True):
        S.page = "Dashboard"
        st.rerun()
    st.markdown('''<div class="section-intro"><div class="eyebrow">OUR TOOLS</div><h2>Manage Security Services</h2><p>Check a connection, analyse a CSV file, and review results from this session.</p></div>''', unsafe_allow_html=True)
    for col, page in zip(st.columns(3, gap="medium"), PAGE_INFO):
        title, desc = PAGE_INFO[page]
        with col:
            st.markdown(f'<div class="service-card"><div class="service-icon"><svg viewBox="0 0 48 48" aria-hidden="true">{ARTWORK[page]}</svg></div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
            if st.button("Open " + title + " →", key="home_" + page, use_container_width=True):
                S.page = page
                st.rerun()
    st.markdown('''<div class="section-intro" style="padding-top:65px"><div class="eyebrow">HOW IT WORKS</div><h2>From Log to Insight</h2><p>Enter connection details or upload a CSV, review the model's prediction, then explore the results in your dashboard.</p></div>''', unsafe_allow_html=True)


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
    head("Detection")
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
    head("Batch analysis")
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
    head("Dashboard")
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


{"Home": page_home, "Detection": page_check, "Batch analysis": page_file, "Dashboard": page_summary}[S.page]()
