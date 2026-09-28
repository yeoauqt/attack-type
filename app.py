"""
NSL-KDD Data Engineering Dashboard  (204426 — เทอม 1/69)
Streamlit app: ใช้ pipeline จาก attack_type_nslkdd_DE.ipynb  |  UI ตามแบบ 'Extej Wallet'
รัน:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

import pipeline as P
from styles import CLASS_COLORS, CSS, LOGO_SVG, NAVY, ORANGE

st.set_page_config(page_title="NetGuard · NSL-KDD DE", page_icon="🛡️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


def html(s: str):
    """markdown ถือว่าบรรทัดที่เยื้อง 4 ช่องเป็น code จึงตัดช่องว่าง/บรรทัดว่างออกก่อน"""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


def fmt(n) -> str:
    return f"{int(n):,}"


# ============================================================ data (cached)
@st.cache_resource(show_spinner="กำลังโหลดข้อมูล NSL-KDD และรัน Data Engineering pipeline ...")
def get_data(train_bytes=None, test_bytes=None, corr_threshold=0.95):
    import io
    tr_src = io.BytesIO(train_bytes) if train_bytes else None
    te_src = io.BytesIO(test_bytes) if test_bytes else None
    train, test = P.load_raw(tr_src, te_src)
    train, test = P.add_attack_type(train, test)
    de = P.run_de(train, test, corr_threshold)
    return train, test, de


@st.cache_data(show_spinner="กำลังรัน ablation (CV) ... ใช้เวลาสักครู่")
def get_experiments(n_est, _de, key):
    ab = P.run_ablation(_de, n_est=n_est)
    best = ab["cv_macro_f1"].idxmax()
    comp = P.run_comparison(_de, best, n_est=n_est)
    return ab, best, comp


@st.cache_resource(show_spinner="กำลังเทรนโมเดลสุดท้าย ...")
def get_final(setup, n_est, _de, key):
    Xtr, ytr, Xte = _de["setups"][setup]
    model = P.make_model(Xtr, "rf", scale=False, n_est=n_est).fit(Xtr, ytr)
    return model, model.predict(Xte), model.predict_proba(Xte)


# ============================================================ sidebar
NAV = ["🏠  Dashboard", "🔀  Pipeline", "🧪  Data Quality", "🛠️  Feature Eng.",
       "📊  Ablation", "🤖  Models", "🔍  Error Analysis", "🎯  Predict"]

with st.sidebar:
    html(f'<div class="brand">{LOGO_SVG}<span class="name">NetGuard</span></div><div class="side-title">Pages</div>')
    page = st.radio("nav", NAV, label_visibility="collapsed", key="nav")
    st.markdown('<div class="side-title">Settings</div>', unsafe_allow_html=True)
    with st.expander("⚙️  Data & Model"):
        up_tr = st.file_uploader("KDDTrain+.txt (ไม่ใส่ = โหลดอัตโนมัติ)", type=["txt", "csv"])
        up_te = st.file_uploader("KDDTest+.txt", type=["txt", "csv"])
        corr_th = st.slider("Correlation threshold", 0.80, 0.99, 0.95, 0.01)
        n_est = st.select_slider("Random Forest: n_estimators", [20, 50, 100, 200], value=100)
    html("""<div class="side-title">Documentation & Support</div>
    <a class="side-link" href="https://github.com/defcom17/NSL_KDD" target="_blank">📄 NSL-KDD dataset</a>
    <a class="side-link" href="https://docs.streamlit.io" target="_blank">💬 Streamlit docs</a>""")

# ------- load
try:
    train, test, de = get_data(up_tr.getvalue() if up_tr else None,
                               up_te.getvalue() if up_te else None, corr_th)
except Exception as e:  # ไม่มีเน็ต/ไม่มีไฟล์
    st.error("โหลดข้อมูลไม่สำเร็จ — วางไฟล์ `KDDTrain+.txt` และ `KDDTest+.txt` ไว้ในโฟลเดอร์ `data/` "
             "หรืออัปโหลดที่ Sidebar › Data & Model")
    st.exception(e)
    st.stop()

data_key = f"{len(train)}-{len(test)}-{corr_th}"
if "exp" not in st.session_state or st.session_state.get("exp_key") != (data_key, n_est):
    st.session_state.pop("exp", None)

BEST_DEFAULT = "3 + engineered features"
best_setup = st.session_state["exp"][1] if "exp" in st.session_state else BEST_DEFAULT

# ------- topbar
html("""<div class="topbar"><div class="ico">🔍</div><div class="search">Search</div>
<div class="ico">🔔</div><div class="ico">✉️</div><div class="avatar">DE</div>
<div class="who"><b>Data Engineer</b><span>204426 · Group Project</span></div><div class="toggle"></div><div class="ico">🇬🇧</div></div>""")


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
    counts = de["counts"]
    total = counts["train"].sum()
    c0, c1, c2, c3 = st.columns([1.5, 1, 1, 1])

    with c0.container(key="card_dataset"):
        bar = "".join(f'<div style="width:{counts.loc[k, "train"] / total * 100:.2f}%;background:{CLASS_COLORS[k]}"></div>'
                      for k in P.LABELS)
        leg = "".join(f'<span><i class="dot" style="background:{CLASS_COLORS[k]}"></i>{k}</span>' for k in P.LABELS)
        n_dup = de["n_dup_train"]
        html(f"""<div class="card-head"><span>📦 Dataset</span><span class="sub">NSL-KDD ▾</span></div>
        <div class="big">{fmt(total)} rows</div>
        <div class="segbar">{bar}</div><div class="legend">{leg}</div>
        <div class="muted" style="margin-top:14px">Test rows (KDDTest+)</div>
        <div class="row-between"><b>{fmt(counts['test'].sum())}</b><span class="red">−{n_dup} duplicates in train</span></div>""")

    for col, k in zip((c1, c2, c3), ("Normal", "DoS", "Probe")):
        with col.container(key=f"card_cls_{k}"):
            share = counts.loc[k, "train"] / total * 100
            html(f"""<div class="card-head"><span><i class="dot" style="background:{CLASS_COLORS[k]}"></i>{k}</span><span class="sub">train ▾</span></div>
            <div class="big">{fmt(counts.loc[k, 'train'])}</div>
            <div class="muted">test = {fmt(counts.loc[k, 'test'])}</div>
            <div class="muted" style="margin-top:14px">Share of train</div>
            <div class="row-between"><b>{share:.1f}%</b><span class="green">R2L {fmt(counts.loc['R2L','train'])} · U2R {fmt(counts.loc['U2R','train'])}</span></div>""")

    with st.container(key="card_flow"):
        h1, h2 = st.columns([1, 1])
        h1.markdown('<div class="card-title">Data Flow</div>', unsafe_allow_html=True)
        with h2.container(key="pills"):
            metric = st.radio("metric", ["Features", "Train rows", "Test rows"], horizontal=True,
                              label_visibility="collapsed")
        sd = de["step_df"].reset_index()
        ycol = {"Features": "n_features", "Train rows": "train_rows", "Test rows": "test_rows"}[metric]
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
        h1.markdown('<div class="card-title">My Models</div>', unsafe_allow_html=True)
        if h2.button("＋ RUN", help="รัน ablation + เปรียบเทียบโมเดล (ใช้เวลาหลายนาที)"):
            run_experiments()
        comp = st.session_state["exp"][2] if "exp" in st.session_state else None
        cols = st.columns(3)
        items = [("Dummy (most frequent)", "BASELINE"), ("Logistic Regression + scaler", "LINEAR"),
                 ("Random Forest (no scaler)", "FINAL")]
        for col, (name, tag) in zip(cols, items):
            acc = f"{comp.loc[name, 'test_acc']:.3f}" if comp is not None else "— — — —"
            f1 = f"{comp.loc[name, 'test_macro_f1']:.3f}" if comp is not None else "—"
            active = " active" if tag == "FINAL" else ""
            chk = '<div class="check">✓</div>' if tag == "FINAL" else ""
            col.markdown("".join(l.strip() for l in f"""<div class="mcard{active}">{chk}<div class="circles"><i class="c1"></i><i class="c2"></i></div>
            <div class="num">Acc {acc}</div><div class="foot"><div><small>Model</small>{name.split(' (')[0].split(' +')[0].upper()}</div>
            <div><small>Macro F1</small>{f1}</div><div>{tag}</div></div></div>""".splitlines()), unsafe_allow_html=True)
        if comp is None:
            st.caption("กดปุ่ม ＋ RUN เพื่อคำนวณผลโมเดล (หรือไปที่หน้า Ablation / Models)")


def page_pipeline():
    st.markdown('<h1 class="page-title">Pipeline</h1>', unsafe_allow_html=True)
    colors = ["#dbeafe", "#e0e7ff", "#fef9c3", "#fde68a", "#fed7aa", "#fecaca", "#e9d5ff", "#bbf7d0", "#dbeafe"]
    boxes = '<div class="arrow">➜</div>'.join(
        f'<div class="box" style="background:{c}"><b>{t}</b><span>{d}</span></div>' for (t, d), c in zip(P.PIPELINE_STEPS, colors))
    with st.container(key="card_pipe"):
        st.markdown('<div class="card-title">Data Engineering Pipeline: Input → Process → Output</div>', unsafe_allow_html=True)
        html(f'<div class="flow">{boxes}</div>')
    with st.container(key="card_steplog"):
        st.markdown('<div class="card-title">Step Log (validate หลังทุกขั้น)</div>', unsafe_allow_html=True)
        st.dataframe(de["step_df"])
        st.caption("`validate()` ตรวจ: ไม่มี missing · จำนวนแถว X/y ตรงกัน · schema train/test ตรงกัน · "
                   "คอลัมน์ categorical ยังอยู่ · label อยู่ใน 5 หมวด")
    with st.container(key="card_target"):
        st.markdown('<div class="card-title">จัดกลุ่ม attack เป็น 5 หมวด (target)</div>', unsafe_allow_html=True)
        rows = [("Normal", "normal"), ("DoS", ", ".join(P.DOS)), ("Probe", ", ".join(P.PROBE)),
                ("R2L", ", ".join(P.R2L)), ("U2R", ", ".join(P.U2R))]
        st.dataframe(pd.DataFrame(rows, columns=["Attack Type", "label (subtype)"]))
    with st.container(key="card_raw"):
        st.markdown('<div class="card-title">Extraction: ตัวอย่างข้อมูลดิบ (KDDTrain+)</div>', unsafe_allow_html=True)
        st.dataframe(train.head(20))


def page_quality():
    st.markdown('<h1 class="page-title">Data Quality</h1>', unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("Missing", de["n_missing"])
    m[1].metric("Duplicate (train / test)", f"{de['n_dup_train']} / {de['n_dup_test']}")
    m[2].metric("Constant features", len(de["const_cols"]))
    m[3].metric("Unseen subtypes in test", len(de["unseen"]))
    with st.container(key="card_qr"):
        st.markdown('<div class="card-title">Data Quality Report (ปัญหา → วิธีแก้)</div>', unsafe_allow_html=True)
        st.dataframe(de["quality_report"])
    a, b = st.columns(2)
    with a.container(key="card_class"):
        st.markdown('<div class="card-title">Class distribution (log scale)</div>', unsafe_allow_html=True)
        cdf = de["counts"].reset_index().melt(id_vars="Attack Type", var_name="set", value_name="count")
        fig = px.bar(cdf, x="Attack Type", y="count", color="set", barmode="group", log_y=True,
                     color_discrete_map={"train": ORANGE, "test": "#5b8def"})
        st.plotly_chart(plotly_style(fig), key="class_chart")
    with b.container(key="card_skew"):
        st.markdown('<div class="card-title">Skewness สูงสุด 10 อันดับ</div>', unsafe_allow_html=True)
        sk = de["skew"].head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=sk.values, y=sk.index, orientation="h", marker_color=ORANGE))
        st.plotly_chart(plotly_style(fig), key="skew_chart")
    a, b = st.columns([1, 1.2])
    with a.container(key="card_pairs"):
        st.markdown(f'<div class="card-title">คู่ feature ที่ correlate > {corr_th}</div>', unsafe_allow_html=True)
        st.dataframe(de["corr_pairs"].round(3))
        st.markdown(f"**ลบ:** `{', '.join(de['corr_drop']) or '-'}`")
    with b.container(key="card_heat"):
        st.markdown('<div class="card-title">Correlation heatmap (train, หลัง dedup)</div>', unsafe_allow_html=True)
        cm = de["corr_matrix"]
        fig = px.imshow(cm, color_continuous_scale=["#ffffff", "#ffc98f", ORANGE, "#c2410c"], zmin=0, zmax=1, aspect="auto")
        st.plotly_chart(plotly_style(fig, 420), key="heat_chart")
    with st.container(key="card_unseen"):
        st.markdown('<div class="card-title">Attack subtype ที่มีเฉพาะใน test</div>', unsafe_allow_html=True)
        st.write(f"{len(de['unseen'])} subtypes = **{fmt(de['n_unseen_rows'])} / {fmt(len(test))}** แถวของ test")
        st.write(", ".join(de["unseen"]))


def page_features():
    st.markdown('<h1 class="page-title">Feature Engineering</h1>', unsafe_allow_html=True)
    spec = pd.DataFrame([
        ["bytes_ratio", "src_bytes / (dst_bytes + 1)", "อัตราส่วนข้อมูลส่งออก/รับเข้า"],
        ["src_bytes_log", "log1p(src_bytes)", "ลด skewness"],
        ["dst_bytes_log", "log1p(dst_bytes)", "ลด skewness"],
        ["duration_log", "log1p(duration)", "ลด skewness"],
        ["err_rate_mean", "mean(serror_rate, rerror_rate)", "รวม error rate"],
        ["srv_ratio", "srv_count / (count + 1)", "สัดส่วน connection ไป service เดียวกัน"],
        ["host_srv_ratio", "dst_host_srv_count / (dst_host_count + 1)", "สัดส่วนระดับ host"],
        ["suspicious_cnt", "hot + failed_logins + compromised + ...", "รวมพฤติกรรมน่าสงสัย"],
        ["service (grouped)", "top-15 จาก train, ที่เหลือ = 'other'", "ลด cardinality ของ one-hot"],
    ], columns=["Feature", "Formula", "Purpose"])
    with st.container(key="card_fe"):
        st.markdown('<div class="card-title">Engineered features (ทดลอง — วัดผลใน Ablation)</div>', unsafe_allow_html=True)
        st.dataframe(spec)
    _, _, Xte = de["setups"]["3 + engineered features"]
    Xtr = de["setups"]["3 + engineered features"][0]
    with st.container(key="card_fe2"):
        st.markdown('<div class="card-title">ตัวอย่างข้อมูลหลัง Feature Engineering</div>', unsafe_allow_html=True)
        new_cols = ["bytes_ratio", "src_bytes_log", "dst_bytes_log", "duration_log", "err_rate_mean",
                    "srv_ratio", "host_srv_ratio", "suspicious_cnt", "service"]
        st.dataframe(Xtr[new_cols].head(15))
        st.write(f"Top services (train): `{', '.join(sorted(de['top_services']))}`")
        st.write(f"Features: **{Xtr.shape[1]}** (numeric {Xtr.shape[1] - 3} + categorical 3 → one-hot ใน model pipeline)")


def page_ablation():
    st.markdown('<h1 class="page-title">Ablation Study</h1>', unsafe_allow_html=True)
    with st.container(key="card_ab"):
        h1, h2 = st.columns([4, 1])
        h1.markdown('<div class="card-title">Data Engineering ช่วยจริงไหม (Random Forest)</div>', unsafe_allow_html=True)
        if h2.button("▶ RUN"):
            run_experiments()
        st.caption("เลือก setup ที่ดีที่สุดจาก **CV macro F1 บน train** — ผลบน KDDTest+ ใช้รายงานเท่านั้น (ป้องกัน leakage)")
        if "exp" not in st.session_state:
            st.info("กดปุ่ม ▶ RUN เพื่อรัน CV 3-fold ทั้ง 4 setup")
            return
        ab, best, _ = st.session_state["exp"]
        st.dataframe(ab)
        st.success(f"Best setup (CV macro F1 บน train): **{best}**")
        long = ab.reset_index().melt(id_vars="setup", value_vars=["cv_macro_f1", "test_macro_f1"], var_name="metric")
        fig = px.bar(long, x="setup", y="value", color="metric", barmode="group",
                     color_discrete_map={"cv_macro_f1": ORANGE, "test_macro_f1": "#5b8def"})
        st.plotly_chart(plotly_style(fig, 360), key="ab_chart")
        st.caption("CV บน train มักสูงกว่า test เพราะแถวใน train คล้ายกันมาก และ test มี attack subtype ใหม่")


def page_models():
    st.markdown('<h1 class="page-title">Models</h1>', unsafe_allow_html=True)
    if "exp" not in st.session_state:
        with st.container(key="card_m0"):
            st.info("ยังไม่ได้รัน experiments — กดเพื่อคำนวณ (Ablation + เปรียบเทียบโมเดล)")
            if st.button("▶ RUN experiments"):
                run_experiments()
                st.rerun()
        return
    ab, best, comp = st.session_state["exp"]
    with st.container(key="card_m1"):
        st.markdown(f'<div class="card-title">Model Comparison (setup: {best})</div>', unsafe_allow_html=True)
        st.dataframe(comp)
        st.caption("StandardScaler จำเป็นกับ Logistic Regression เท่านั้น — Random Forest ไม่ต้อง scale (เทียบในตารางด้านบน)")
    model, pred, _ = get_final(best, n_est, de, data_key)
    y_test = de["y_test"]
    rep = pd.DataFrame(classification_report(y_test, pred, labels=P.LABELS, digits=3, zero_division=0, output_dict=True)).T
    a, b = st.columns(2)
    with a.container(key="card_rep"):
        st.markdown('<div class="card-title">Classification report (KDDTest+)</div>', unsafe_allow_html=True)
        st.dataframe(rep.round(3))
    with b.container(key="card_cm"):
        st.markdown('<div class="card-title">Confusion matrix (normalized by true class = recall)</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, pred, labels=P.LABELS, normalize="true")
        fig = px.imshow(cm, x=P.LABELS, y=P.LABELS, text_auto=".2f", color_continuous_scale=["#fff", ORANGE], zmin=0, zmax=1)
        fig.update_layout(xaxis_title="Predicted", yaxis_title="True")
        st.plotly_chart(plotly_style(fig, 360), key="cm_chart")
    with st.container(key="card_imp"):
        st.markdown('<div class="card-title">Top 15 feature importances</div>', unsafe_allow_html=True)
        names = model.named_steps["pre"].get_feature_names_out()
        imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False).head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ORANGE))
        st.plotly_chart(plotly_style(fig, 420), key="imp_chart")


def page_error():
    st.markdown('<h1 class="page-title">Error Analysis: R2L & U2R</h1>', unsafe_allow_html=True)
    if "exp" not in st.session_state:
        st.info("ไปหน้า Models แล้วกด ▶ RUN experiments ก่อน (หรือแสดงผลด้วย setup เริ่มต้น)")
    model, pred, _ = get_final(best_setup, n_est, de, data_key)
    sub, by_seen, r2l_pred = P.error_analysis(test, pred, de["unseen"])
    with st.container(key="card_e1"):
        st.markdown('<div class="card-title">Recall แยกตาม seen / unseen subtype</div>', unsafe_allow_html=True)
        st.dataframe(by_seen)
    with st.container(key="card_e2"):
        st.markdown('<div class="card-title">Recall แยกตาม attack subtype (เรียงตามจำนวน)</div>', unsafe_allow_html=True)
        st.dataframe(sub.head(25))
    with st.container(key="card_e3"):
        st.markdown('<div class="card-title">R2L ถูกทายเป็นอะไร</div>', unsafe_allow_html=True)
        fig = px.bar(r2l_pred.reset_index(), x="pred", y="count", color="pred", color_discrete_map=CLASS_COLORS)
        st.plotly_chart(plotly_style(fig, 300), key="r2l_chart")


def page_predict():
    st.markdown('<h1 class="page-title">Predict</h1>', unsafe_allow_html=True)
    model, pred, proba = get_final(best_setup, n_est, de, data_key)
    classes = list(model.named_steps["clf"].classes_)
    with st.container(key="card_p1"):
        st.markdown('<div class="card-title">เลือก connection จาก KDDTest+ แล้วให้โมเดลทำนาย</div>', unsafe_allow_html=True)
        idx = st.slider("แถวที่", 0, len(test) - 1, 0)
        true, p = test["Attack Type"].iloc[idx], pred[idx]
        ok = true == p
        c = st.columns(3)
        c[0].metric("Actual", f"{true}  ({test['label'].iloc[idx]})")
        c[1].metric("Predicted", p)
        c[2].metric("Result", "✅ ถูกต้อง" if ok else "❌ ผิด")
        fig = go.Figure(go.Bar(x=classes, y=proba[idx], marker_color=[CLASS_COLORS[k] for k in classes]))
        fig.update_yaxes(range=[0, 1], title="probability")
        st.plotly_chart(plotly_style(fig, 300), key="pred_chart")
        with st.expander("ค่า feature ของแถวนี้ (หลัง DE)"):
            Xte = de["setups"][best_setup][2]
            st.dataframe(Xte.iloc[[idx]].T)


{"🏠  Dashboard": page_dashboard, "🔀  Pipeline": page_pipeline, "🧪  Data Quality": page_quality,
 "🛠️  Feature Eng.": page_features, "📊  Ablation": page_ablation, "🤖  Models": page_models,
 "🔍  Error Analysis": page_error, "🎯  Predict": page_predict}[page]()
