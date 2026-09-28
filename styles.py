"""Visual theme for the NSL-KDD Data Engineering Dashboard.

Design direction: formal, academic and restrained.
Deep navy as the primary colour, steel blue as the single accent, hairline
borders instead of shadows, serif headings paired with a neutral sans body.
"""

NAVY = "#14284B"
ACCENT = "#2F5D8A"
MUTED_BLUE = "#8FA6C4"
INK = "#1F2937"
GRID = "#E6E9EF"

# Muted, print-friendly category colours
CLASS_COLORS = {
    "Normal": "#2E6F5E",
    "DoS": "#9E3B3B",
    "Probe": "#B0822D",
    "R2L": "#4A5F8F",
    "U2R": "#6B4C7A",
}

LOGO_SVG = """
<svg width="30" height="30" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M16 3 L27 7 V15 C27 21.5 22.5 26.5 16 29 C9.5 26.5 5 21.5 5 15 V7 Z"
        stroke="#14284B" stroke-width="2" fill="none" stroke-linejoin="round"/>
  <path d="M10.5 16 H14 L16 11 L18.5 20 L20 16 H21.5" stroke="#2F5D8A" stroke-width="2"
        fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:wght@500;600;700&display=swap');

:root {
  --navy: #14284B;
  --accent: #2F5D8A;
  --ink: #1F2937;
  --muted: #6B7280;
  --line: #DFE3EA;
  --bg: #F4F5F8;
  --card: #FFFFFF;
  --tint: #EEF2F7;
}

html, body, .stApp, [class*="css"] {
  font-family: 'Inter', 'Noto Sans Thai', -apple-system, 'Segoe UI', sans-serif;
  color: var(--ink);
}
.stApp { background: var(--bg); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }

/* ---------- top bar ---------- */
.topbar {
  display: flex; justify-content: space-between; align-items: center;
  background: var(--navy); color: #E7ECF4; border-radius: 4px;
  padding: .65rem 1.2rem; font-size: .8rem; letter-spacing: .04em; margin-bottom: 1.6rem;
}
.topbar b { color: #fff; font-weight: 600; }

/* ---------- headings ---------- */
.eyebrow {
  font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); font-weight: 600; margin-bottom: .2rem;
}
h1.page-title {
  font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  font-weight: 600; color: var(--navy); font-size: 2.1rem; line-height: 1.2;
  margin: 0 0 .35rem 0; padding: 0;
}
p.page-sub { color: var(--muted); font-size: .95rem; margin: 0 0 1.4rem 0; max-width: 62rem; line-height: 1.55; }
.card-title {
  font-family: 'Source Serif 4', Georgia, serif; font-size: 1.18rem; font-weight: 600;
  color: var(--navy); padding-bottom: .55rem; margin-bottom: .9rem; border-bottom: 1px solid var(--line);
}
.card-note { color: var(--muted); font-size: .85rem; line-height: 1.5; }

/* ---------- cards (any st.container whose key starts with card_) ---------- */
[class*="st-key-card_"] {
  background: var(--card); border: 1px solid var(--line); border-radius: 4px;
  padding: 1.3rem 1.6rem 1.2rem; margin-bottom: 1rem;
}
[class*="st-key-card_"] p, [class*="st-key-card_"] li { line-height: 1.6; }

/* ---------- statistic blocks ---------- */
.stat-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 600; }
.stat-value { font-family: 'Source Serif 4', Georgia, serif; font-size: 2rem; font-weight: 600; color: var(--navy); margin: .2rem 0; }
.stat-foot { font-size: .82rem; color: var(--muted); }

/* ---------- formal tables ---------- */
table.ftable { width: 100%; border-collapse: collapse; font-size: .88rem; margin: .2rem 0 .4rem; }
table.ftable th {
  text-align: left; font-size: .7rem; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; padding: .55rem .7rem; border-bottom: 2px solid var(--navy);
}
table.ftable td { padding: .65rem .7rem; border-bottom: 1px solid var(--line); vertical-align: top; line-height: 1.5; }
table.ftable tr:last-child td { border-bottom: none; }
table.ftable td:first-child { font-weight: 600; color: var(--navy); white-space: nowrap; }

/* ---------- technique blocks ---------- */
.tech-head { display: flex; align-items: baseline; gap: .8rem; margin-bottom: .3rem; }
.tech-no { font-family: 'Source Serif 4', serif; font-size: 1.05rem; font-weight: 700; color: var(--accent); }
.tech-name { font-family: 'Source Serif 4', serif; font-size: 1.15rem; font-weight: 600; color: var(--navy); }
.chip {
  display: inline-block; font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 600;
  color: var(--accent); background: var(--tint); border: 1px solid #D5DEEA; border-radius: 3px; padding: .12rem .5rem; margin-left: auto;
}
.tech-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .9rem 2rem; margin-top: .8rem; }
.tech-grid .lbl { font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin-bottom: .15rem; }
.tech-grid .wide { grid-column: 1 / -1; }
.tech-grid .txt { font-size: .9rem; line-height: 1.55; }
.evidence {
  margin-top: 1rem; padding: .6rem .9rem; background: var(--tint); border-left: 3px solid var(--accent);
  font-size: .86rem; color: var(--ink);
}
.evidence b { color: var(--navy); }

/* ---------- lists ---------- */
ul.plain { margin: .2rem 0 .2rem 1.1rem; padding: 0; }
ul.plain li { margin-bottom: .45rem; font-size: .93rem; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--line); }
.brand { display: flex; align-items: center; gap: .7rem; padding: .3rem .2rem 1rem; }
.brand .name { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.05rem; font-weight: 700; color: var(--navy); line-height: 1.2; }
.brand .tag { font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); display: block; font-family: 'Inter', sans-serif; font-weight: 500; }
.side-title {
  font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted);
  font-weight: 600; margin: 1.3rem 0 .4rem .2rem;
}
.side-link { display: block; padding: .35rem .2rem; font-size: .86rem; color: var(--accent) !important; text-decoration: none; }
.side-link:hover { text-decoration: underline; }

section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
  width: 100%; padding: .5rem .75rem; border-radius: 3px; border-left: 3px solid transparent; margin: 0;
}
section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label p { font-size: .9rem; color: #374151; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #F5F7FA; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background: var(--tint); border-left-color: var(--navy);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: var(--navy); font-weight: 600; }

/* ---------- segmented control (Data Flow metric) ---------- */
.st-key-pills div[role="radiogroup"] { justify-content: flex-end; gap: 0; }
.st-key-pills label[data-baseweb="radio"] { border: 1px solid var(--line); padding: .3rem .9rem; margin: 0; background: #fff; }
.st-key-pills label[data-baseweb="radio"] > div:first-child { display: none; }
.st-key-pills label[data-baseweb="radio"] p { font-size: .82rem; }
.st-key-pills label:has(input:checked) { background: var(--navy); border-color: var(--navy); }
.st-key-pills label:has(input:checked) p { color: #fff; }

/* ---------- widgets ---------- */
.stButton > button {
  background: var(--navy); color: #fff; border: 1px solid var(--navy); border-radius: 3px;
  padding: .45rem 1.2rem; font-weight: 500; letter-spacing: .02em;
}
.stButton > button:hover { background: var(--accent); border-color: var(--accent); color: #fff; }
.stButton > button:focus:not(:active) { color: #fff; border-color: var(--accent); }

div[data-testid="stMetric"] {
  background: #fff; border: 1px solid var(--line); border-radius: 4px; padding: .9rem 1.1rem;
}
div[data-testid="stMetricLabel"] p { font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 600; }
div[data-testid="stMetricValue"] { font-family: 'Source Serif 4', Georgia, serif; color: var(--navy); }
div[data-testid="stAlert"] { border-radius: 3px; }
div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 3px; background: #fff; }

.muted { color: var(--muted); font-size: .86rem; }
.pos { color: #2E6F5E; font-weight: 600; }
.neg { color: #9E3B3B; font-weight: 600; }
</style>
"""
