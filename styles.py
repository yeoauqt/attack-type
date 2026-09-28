"""CSS ให้หน้าตาเหมือน dashboard 'Extej Wallet' ในภาพตัวอย่าง (โทนส้ม-ขาว, การ์ดมุมโค้ง, sidebar เมนู pill)"""

ORANGE = "#ff8a24"
NAVY = "#1e2340"
CLASS_COLORS = {"Normal": "#ff9f43", "DoS": "#5b8def", "Probe": "#14b8a6", "R2L": "#ec4899", "U2R": "#8b5cf6"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Poppins','Noto Sans Thai',sans-serif; color:{NAVY}; }}
.stApp {{
  background:
    radial-gradient(circle at 8% 0%, rgba(255,190,150,.35), transparent 32%),
    radial-gradient(circle at 92% 0%, rgba(150,225,215,.35), transparent 30%),
    #f6f7fb;
}}
#MainMenu, footer {{ visibility:hidden; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1rem; padding-bottom:3rem; max-width:1400px; }}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{ background:#ffffff; border-right:1px solid #eef0f5; width:270px !important; }}
section[data-testid="stSidebar"] > div {{ padding-top:.6rem; }}
.brand {{ display:flex; align-items:center; gap:10px; padding:6px 6px 18px 6px; }}
.brand .name {{ font-size:26px; font-weight:700; color:{NAVY}; letter-spacing:.3px; }}
.side-title {{ font-size:11px; letter-spacing:1px; color:#aab0c0; margin:14px 8px 6px; text-transform:uppercase; }}
section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:4px; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
  width:100%; padding:11px 14px; border-radius:12px; margin:0; cursor:pointer; transition:.15s;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display:none; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size:14.5px; color:#8a90a2; margin:0; font-weight:500; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background:#fff4e8; }}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
  background:linear-gradient(90deg,#ffa54f,{ORANGE}); box-shadow:0 10px 20px rgba(255,138,36,.35);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ color:#fff; font-weight:600; }}
.side-link {{ display:block; padding:9px 14px; color:#8a90a2; font-size:14px; text-decoration:none; border-radius:10px; }}
.side-link:hover {{ background:#fff4e8; color:{ORANGE}; }}

/* ---------- Topbar ---------- */
.topbar {{ display:flex; align-items:center; gap:16px; background:#fff; border-radius:16px; padding:12px 20px;
  box-shadow:0 6px 24px rgba(30,35,64,.05); margin-bottom:18px; }}
.topbar .search {{ flex:1; background:#f6f7fb; border-radius:10px; padding:10px 16px; color:#aab0c0; font-size:13px; }}
.topbar .ico {{ font-size:17px; color:#9aa1b4; }}
.topbar .avatar {{ width:40px; height:40px; border-radius:50%; background:linear-gradient(135deg,#ffb35c,#ff7a1a);
  color:#fff; display:flex; align-items:center; justify-content:center; font-weight:600; }}
.topbar .who b {{ font-size:14px; display:block; line-height:1.1; }}
.topbar .who span {{ font-size:12px; color:#9aa1b4; }}
.topbar .toggle {{ width:44px; height:24px; border-radius:20px; background:#eef0f5; position:relative; }}
.topbar .toggle::after {{ content:''; position:absolute; top:3px; left:23px; width:18px; height:18px; border-radius:50%; background:#ffd166; }}

h1.page-title {{ font-size:28px; font-weight:700; margin:0 0 14px 2px; color:{NAVY}; }}

/* ---------- Cards ( st.container(key="card_*") ) ---------- */
[class*="st-key-card"] {{
  background:#fff; border-radius:18px; padding:20px 22px; box-shadow:0 6px 24px rgba(30,35,64,.05);
  border:1px solid #f0f1f6; margin-bottom:6px;
}}
.card-head {{ display:flex; justify-content:space-between; align-items:center; font-size:15px; color:{NAVY}; font-weight:500; }}
.card-head .sub {{ font-size:12px; color:#8a90a2; }}
.card-title {{ font-size:19px; font-weight:600; margin-bottom:6px; }}
.big {{ font-size:28px; font-weight:700; color:{NAVY}; margin:8px 0 2px; }}
.muted {{ color:#9aa1b4; font-size:12.5px; }}
.green {{ color:#22b573; font-weight:600; }}
.red {{ color:#ef4b5b; font-weight:600; }}
.segbar {{ display:flex; height:3px; border-radius:3px; overflow:hidden; margin:16px 0 8px; }}
.legend {{ display:flex; gap:16px; font-size:12px; color:#4b5168; flex-wrap:wrap; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }}
.row-between {{ display:flex; justify-content:space-between; align-items:baseline; }}

/* ---------- Pipeline diagram ---------- */
.flow {{ display:flex; align-items:stretch; gap:6px; overflow-x:auto; padding:6px 2px 10px; }}
.flow .box {{ min-width:128px; flex:1; border-radius:14px; padding:12px 10px; text-align:center; border:1px solid rgba(30,35,64,.12); }}
.flow .box b {{ display:block; font-size:12.5px; margin-bottom:5px; }}
.flow .box span {{ font-size:11.5px; color:#4b5168; white-space:pre-line; }}
.flow .arrow {{ align-self:center; color:#9aa1b4; font-size:18px; }}

/* ---------- Bank-card style (model cards) ---------- */
.mcard {{ border:1px solid #eceef4; border-radius:16px; padding:18px 18px 14px; background:#fff; height:100%; position:relative; overflow:hidden; }}
.mcard.active {{ border:2px solid {ORANGE}; }}
.mcard .circles {{ display:flex; }}
.mcard .c1, .mcard .c2 {{ width:24px; height:24px; border-radius:50%; }}
.mcard .c1 {{ background:#ef4b4b; }} .mcard .c2 {{ background:#ffab2e; margin-left:-9px; opacity:.95; }}
.mcard .num {{ font-size:19px; font-weight:600; margin:22px 0 22px; letter-spacing:.5px; }}
.mcard .foot {{ display:flex; justify-content:space-between; font-size:11.5px; font-weight:600; }}
.mcard .foot small {{ display:block; color:#b6bccb; font-weight:400; font-size:10px; }}
.mcard .check {{ position:absolute; right:16px; top:16px; width:20px; height:20px; border-radius:50%; background:{ORANGE}; color:#fff; font-size:12px; text-align:center; line-height:20px; }}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {{
  background:linear-gradient(180deg,#ffa24c,{ORANGE}); color:#fff; border:none; border-radius:10px; font-weight:600;
  padding:.45rem 1.2rem; box-shadow:0 8px 16px rgba(255,138,36,.28);
}}
.stButton > button:hover {{ color:#fff; filter:brightness(1.05); border:none; }}
.stButton > button:focus:not(:active) {{ color:#fff; border:none; }}

/* ---------- Pills (1D 7D 1M ...) ---------- */
.st-key-pills div[role="radiogroup"] {{ flex-direction:row; gap:4px; background:#f6f7fb; padding:4px; border-radius:12px; width:fit-content; }}
.st-key-pills label {{ padding:4px 14px; border-radius:9px; margin:0; }}
.st-key-pills label > div:first-child {{ display:none; }}
.st-key-pills label p {{ font-size:13px; color:#8a90a2; }}
.st-key-pills label:has(input:checked) {{ background:{ORANGE}; }}
.st-key-pills label:has(input:checked) p {{ color:#fff; font-weight:600; }}

/* tables / metrics */
[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; }}
[data-testid="stMetric"] {{ background:#fff8f1; border-radius:14px; padding:12px 16px; }}
</style>
"""

LOGO_SVG = """<svg width="38" height="38" viewBox="0 0 40 40"><polygon points="20,2 36,11 36,29 20,38 4,29 4,11" fill="#ef3e3e"/>
<polygon points="20,9 30,15 30,25 20,31 10,25 10,15" fill="none" stroke="#fff" stroke-width="3"/></svg>"""
