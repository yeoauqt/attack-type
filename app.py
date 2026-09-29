import os, io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import pipeline as P

st.set_page_config(page_title="NSL-KDD Data Engineering", page_icon="🛡️", layout="wide")

ART_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "artifacts.joblib")
CLASS_COLORS = {"Normal": "#0f766e", "DoS": "#b45309", "Probe": "#1d4ed8", "R2L": "#9333ea", "U2R": "#be123c"}


@st.cache_resource(show_spinner=False)
def get_artifacts():
    if os.path.exists(ART_PATH):
        try:
            return joblib.load(ART_PATH)
        except Exception:
            pass  # เวอร์ชัน scikit-learn ไม่ตรงกับตอน build -> รัน pipeline ใหม่
    bar = st.progress(0.0, text="ยังไม่มี artifacts — กำลังรัน pipeline ครั้งแรก (ประมาณ 4–5 นาที)")
    art = P.run_all(lambda m, p: bar.progress(p, text=m))
    os.makedirs(os.path.dirname(ART_PATH), exist_ok=True)
    joblib.dump(art, ART_PATH, compress=3)
    bar.empty()
    return art


@st.cache_data(show_spinner=False)
def get_data():
    return P.load_raw()


art = get_artifacts()
cfg = art["cfg"]

st.sidebar.title("🛡️ NSL-KDD")
st.sidebar.caption("ตรวจจับการโจมตีเครือข่ายด้วย Machine Learning")

HOME = "🏠 หน้าแรก · ทำนายการโจมตี"
DETAIL_PAGES = [
    "1 · ภาพรวม Pipeline",
    "2 · Data Profiling & Quality",
    "3 · Pipeline Explorer (ก่อน–หลัง)",
    "4 · Ablation & Model",
]

if st.sidebar.button(HOME, width="stretch", type="primary"):
    st.session_state["page"] = HOME
with st.sidebar.expander("⚙️ รายละเอียดระบบเบื้องหลัง (สำหรับผู้สนใจ / อาจารย์)"):
    for p in DETAIL_PAGES:
        if st.button(p, key=p, width="stretch"):
            st.session_state["page"] = p

page = st.session_state.get("page", HOME)

st.sidebar.divider()
st.sidebar.markdown(f"**Setup ที่ชนะ (CV บน train)**  \n{cfg['best_setup']}")
st.sidebar.markdown(f"Test acc `{art['test_acc']:.3f}` · Macro F1 `{art['test_macro_f1']:.3f}`")


# =============================================================== 1
if page.startswith("1"):
    st.title("Data Engineering Pipeline: NSL-KDD")
    st.markdown(
        "**โจทย์:** ข้อมูล network traffic ดิบมีปัญหาคุณภาพหลายอย่าง (ซ้ำ, คอลัมน์ไม่มีข้อมูล, feature ซ้ำซ้อน, "
        "class ไม่สมดุลมาก) ก่อนส่งเข้าโมเดลจำแนกการโจมตี 5 หมวด (Normal / DoS / Probe / R2L / U2R) "
        "จึงต้องมี pipeline ที่ **ทุกขั้นตอนมาจากปัญหาที่พบจริง และมีหลักฐานการทดลองรองรับ**")

    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#334155"];
      in  [label="INPUT\\nKDDTrain+ / KDDTest+\\nหรือไฟล์ที่อัปโหลด", fillcolor="#dbeafe"];
      s1  [label="1 Extraction\\nโหลด + ใส่ schema\\n(43 คอลัมน์)", fillcolor="#e0e7ff"];
      s2  [label="2 Quality Check\\nMissing / Duplicate\\nConstant / Skew", fillcolor="#fef9c3"];
      s3  [label="3 Cleaning\\nลบ duplicate\\nลบ constant", fillcolor="#fde68a"];
      s4  [label="4 Feature Selection\\ncorrelation > 0.95", fillcolor="#fed7aa"];
      s5  [label="5 Feature Eng.\\n(ทดลอง + ablation)", fillcolor="#fecaca"];
      s6  [label="6 Preprocessing\\nOne-hot (+ Scaler เฉพาะ LR)", fillcolor="#e9d5ff"];
      s7  [label="7 Validation\\nassert หลังทุกขั้น", fillcolor="#bbf7d0"];
      out [label="OUTPUT\\nClean data → Model\\n→ Evaluation", fillcolor="#dbeafe"];
      in -> s1 -> s2 -> s3 -> s4 -> s5 -> s6 -> s7 -> out;
    }""", width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    sl = art["step_log"]
    c1.metric("แถว train (ดิบ → สะอาด)", f"{sl.train_rows.iloc[0]:,}", f"{sl.train_rows.iloc[-1]-sl.train_rows.iloc[0]:,}")
    c2.metric("Feature (ดิบ → สุดท้าย)", int(sl.n_features.iloc[0]), int(sl.n_features.iloc[-1] - sl.n_features.iloc[0]))
    c3.metric("Test accuracy", f"{art['test_acc']:.3f}")
    c4.metric("Test Macro F1", f"{art['test_macro_f1']:.3f}")

    st.subheader("เทคนิคที่ใช้ (พร้อมวัตถุประสงค์)")
    st.dataframe(pd.DataFrame([
        ["Data Extraction + Schema validation", "โหลดไฟล์ไม่มี header แล้วกำหนด 43 คอลัมน์ ตรวจจำนวนคอลัมน์ตรง schema", "กัน schema ผิดตั้งแต่ต้นน้ำ"],
        ["Data Profiling / Quality Report", "นับ missing, duplicate, constant, skewness, class imbalance", "ให้ทุก transformation มีเหตุผลจากข้อมูลจริง"],
        ["Deduplication + Drop constant", "ลบแถวซ้ำใน train และคอลัมน์ที่มีค่าเดียว", "ลดข้อมูลซ้ำที่ทำให้โมเดลเอนเอียง / คอลัมน์ไร้ข้อมูล"],
        ["Correlation-based Feature Selection", "ลบ feature ที่ |corr| > 0.95 (คำนวณจาก train)", "ลดความซ้ำซ้อน ลดโอกาส overfit"],
        ["Feature Engineering + Ablation", "ratio, log, รวม error rate, จัดกลุ่ม service หายาก แล้ววัดผลเทียบ", "ไม่ใช้เพียงเพราะคาดว่าจะช่วย"],
        ["Preprocessing Pipeline (ColumnTransformer)", "One-hot สำหรับ categorical, Scaler เฉพาะโมเดลที่ไวต่อ scale", "ทำ transform เดียวกันตอน train และ predict ไม่รั่ว"],
        ["Step-wise Validation + Step Log", "assert missing/schema/label หลังทุกขั้น และบันทึกจำนวนแถว/feature", "จับความผิดพลาดของ pipeline ทันที"],
    ], columns=["เทคนิค", "ทำอะไร", "วัตถุประสงค์"]), hide_index=True, width="stretch")
    st.info("หลักกัน data leakage: การตัดสินใจทั้งหมด (คอลัมน์ที่ลบ, top services, การเลือก setup) ใช้ **train เท่านั้น**")


# =============================================================== 2
elif page.startswith("2"):
    st.title("Data Profiling & Quality Report")
    st.caption("ผลตรวจข้อมูลดิบของ KDDTrain+ / KDDTest+ (คำนวณจาก train)")

    st.subheader("Data Quality Report: ปัญหา → วิธีแก้")
    st.dataframe(art["quality_report"], hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Class distribution (log scale)")
        cnt = art["counts"].reset_index(names="class").melt("class", var_name="set", value_name="count")
        fig = px.bar(cnt, x="class", y="count", color="set", barmode="group", log_y=True,
                     color_discrete_sequence=["#0f766e", "#94a3b8"], text_auto=",")
        fig.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("U2R มีเพียง 52 แถวใน train — สาเหตุที่ต้องใช้ class_weight และรายงาน Macro F1")
    with right:
        st.subheader("Skewness สูงสุด 10 อันดับ")
        sk = art["skew"].head(10).iloc[::-1]
        fig = px.bar(x=sk.values, y=sk.index, orientation="h", color_discrete_sequence=["#b45309"])
        fig.update_layout(height=380, margin=dict(t=10), xaxis_title="|skew|", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("ค่าเบ้สูงมากเป็นที่มาของการทดลอง log-transform")

    st.subheader("Attack subtype ที่มีเฉพาะใน test")
    tr, te = get_data()
    u = te[te["label"].isin(art["unseen"])]["label"].value_counts().rename_axis("subtype").reset_index(name="rows in test")
    u["category"] = u["subtype"].map(P.to_category)
    st.dataframe(u, hide_index=True, width="stretch")

    with st.expander("ดูข้อมูลดิบ (10 แถวแรก)"):
        st.dataframe(tr.head(10), width="stretch")


# =============================================================== 3
elif page.startswith("3"):
    st.title("Pipeline Explorer")
    st.caption("ดูผลของแต่ละขั้นตอน ก่อน → หลัง")

    st.subheader("Step Log")
    sl = art["step_log"].set_index("step").copy()
    sl["features_change"] = sl["n_features"].diff().fillna(0).astype(int)
    sl["train_rows_change"] = sl["train_rows"].diff().fillna(0).astype(int)
    st.dataframe(sl, width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("ขั้น 1: คอลัมน์/แถวที่ลบ")
        st.markdown(f"- Duplicate ใน train: **{art['n_dup_train']:,}** แถว → ลบ\n"
                    f"- Constant column: **{cfg['const_cols']}** → ลบ")
    with b:
        st.subheader("ขั้น 2: feature ที่ลบเพราะ correlate สูง")
        st.markdown("- " + "\n- ".join(f"`{c}`" for c in cfg["corr_drop"]))

    st.subheader("คู่ feature ที่ |corr| > 0.95 (หลักฐานว่าทำไมถึงลบ)")
    st.dataframe(art["pairs"].round(3), hide_index=True, width="stretch")

    st.subheader("Correlation heatmap (หลังลบ duplicate/constant)")
    corr = art["corr"]
    fig = px.imshow(corr, color_continuous_scale="Teal", zmin=0, zmax=1, aspect="auto")
    fig.update_layout(height=650, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("ขั้น 3: Feature ที่สร้างเพิ่ม")
    st.dataframe(pd.DataFrame([
        ["bytes_ratio", "src_bytes / (dst_bytes + 1)"], ["src_bytes_log, dst_bytes_log, duration_log", "log1p ของ feature เบ้"],
        ["err_rate_mean", "ค่าเฉลี่ย serror_rate, rerror_rate"], ["srv_ratio", "srv_count / (count + 1)"],
        ["host_srv_ratio", "dst_host_srv_count / (dst_host_count + 1)"], ["suspicious_cnt", "ผลรวม hot, failed_logins, compromised, ..."],
        ["service (จัดกลุ่ม)", "เก็บ top 15 จาก train ที่เหลือเป็น 'other'"],
    ], columns=["feature", "วิธีสร้าง"]), hide_index=True, width="stretch")

    st.subheader("ลองทดสอบ transform กับ sample")
    tr, te = get_data()
    n = st.slider("จำนวนแถวจาก KDDTest+", 5, 50, 10)
    sample = te.sample(n, random_state=1)[P.FEATURES]
    t1, t2 = st.tabs(["ก่อน (raw)", f"หลัง (setup {cfg['best_idx']})"])
    t1.dataframe(sample, width="stretch")
    t2.dataframe(P.apply_transform(sample, cfg, cfg["best_idx"]), width="stretch")


# =============================================================== 4
elif page.startswith("4"):
    st.title("Ablation Study & Model")
    st.markdown("เทียบทีละขั้นด้วย Random Forest — **เลือก setup จาก CV Macro F1 บน train** ส่วนผลบน KDDTest+ ใช้รายงานเท่านั้น")

    ab = art["ablation"].round(3)
    st.dataframe(ab.style.highlight_max(subset=["cv_macro_f1", "test_macro_f1"], color="#ccfbf1"), width="stretch")

    long = ab.reset_index().melt("setup", value_vars=["cv_macro_f1", "test_macro_f1"], var_name="metric", value_name="Macro F1")
    fig = px.bar(long, x="setup", y="Macro F1", color="metric", barmode="group", text_auto=".3f",
                 color_discrete_sequence=["#0f766e", "#b45309"])
    fig.update_layout(height=380, margin=dict(t=10), xaxis_title="")
    st.plotly_chart(fig, width="stretch")

    best_test = ab["test_macro_f1"].idxmax()
    if best_test != cfg["best_setup"]:
        st.warning(
            f"**ข้อสังเกต:** CV บน train เลือก *{cfg['best_setup']}* (CV {ab.loc[cfg['best_setup'],'cv_macro_f1']:.3f}) "
            f"แต่บน KDDTest+ setup ที่ดีที่สุดคือ *{best_test}* (Macro F1 {ab.loc[best_test,'test_macro_f1']:.3f}). "
            "CV แบบสุ่มบน train มักสูงเกินจริง เพราะแถวใน train คล้ายกันมาก และ test มี attack subtype ใหม่ที่ train ไม่เคยเห็น "
            "จึงควรรายงานทั้งสองค่า และระบุเป็นข้อจำกัด ไม่ควรเปลี่ยนไปเลือกด้วย test เพราะจะเกิด leakage")

    st.subheader("Model Comparison (ใช้ setup ที่ชนะ)")
    st.dataframe(art["comparison"].round(3), width="stretch")
    st.caption("Scaler ไม่ช่วย Random Forest (tree ไม่ไวต่อ scale) จึงใช้ scaler เฉพาะ Logistic Regression")

    st.subheader("โมเดลสุดท้าย: Random Forest (no scaler)")
    c1, c2 = st.columns(2)
    labels = P.LABELS
    cm = art["confusion"]
    with c1:
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="True"), title="Confusion Matrix (KDDTest+)")
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")
    with c2:
        norm = cm / cm.sum(axis=1, keepdims=True)
        fig = px.imshow(norm, x=labels, y=labels, text_auto=".2f", color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="True"), title="Normalized by true class (= recall)")
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")

    imp = art["importance"].head(15).iloc[::-1]
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", title="Top 15 feature importances",
                 color_discrete_sequence=["#0f766e"])
    fig.update_layout(height=450, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(art["report"].round(3), width="stretch")

    st.subheader("Error Analysis: R2L และ U2R")
    ea = art["error"]
    sub = ea[ea["Attack Type"].isin(["R2L", "U2R"])]
    by_seen = sub.groupby(["Attack Type", "seen_in_train"]).agg(n=("correct", "size"), recall=("correct", "mean")).round(3)
    st.markdown("**Recall แยกตาม subtype ที่เคย/ไม่เคยเห็นใน train**")
    st.dataframe(by_seen, width="stretch")
    st.markdown("**R2L ถูกทายเป็นอะไร**")
    st.dataframe(ea.loc[ea["Attack Type"] == "R2L", "pred"].value_counts().rename("count"), width="stretch")


# =============================================================== HOME (ทำนาย)
else:
    st.title("🛡️ ตรวจจับการโจมตีเครือข่าย (NSL-KDD)")
    st.caption("อัปโหลดข้อมูล traffic หรือลองสุ่มตัวอย่าง แล้วให้โมเดลทำนายว่าเป็น Normal หรือการโจมตีประเภทไหน")

    src = st.radio("แหล่งข้อมูล", ["สุ่มจาก KDDTest+", "อัปโหลดไฟล์ CSV/TXT"], horizontal=True)
    df = None
    if src.startswith("สุ่ม"):
        tr, te = get_data()
        c1, c2, c3 = st.columns(3)
        n = c1.slider("จำนวนแถว", 10, 2000, 200)
        only = c2.selectbox("เฉพาะ class", ["ทั้งหมด"] + P.LABELS)
        seed = c3.number_input("seed", 0, 9999, 1)
        pool = te if only == "ทั้งหมด" else te[te["Attack Type"] == only]
        df = pool.sample(min(n, len(pool)), random_state=int(seed)).reset_index(drop=True)
    else:
        up = st.file_uploader("ไฟล์ NSL-KDD (41/42/43 คอลัมน์ มีหรือไม่มี header ก็ได้)", type=["csv", "txt"])
        if up is not None:
            try:
                df = P.read_uploaded(up)
            except Exception as e:
                st.error(f"อ่านไฟล์ไม่ได้: {e}")
        else:
            st.info("ยังไม่ได้อัปโหลด — ลองดาวน์โหลดตัวอย่างจากแหล่งสุ่มก่อนก็ได้")

    if df is not None:
        try:
            issues, X, y_true = P.check_upload(df)
        except Exception as e:
            st.error(f"Validation ไม่ผ่าน: {e}")
            st.stop()

        st.success(f"Validation ผ่าน: {len(X):,} แถว × {X.shape[1]} features")
        for i in issues:
            st.warning(i)

        Xt = P.apply_transform(X, cfg, cfg["best_idx"])
        proba = art["model"].predict_proba(Xt)
        classes = art["model"].classes_
        out = X.copy()
        out.insert(0, "prediction", classes[proba.argmax(axis=1)])
        out.insert(1, "confidence", proba.max(axis=1).round(3))

        m1, m2, m3 = st.columns(3)
        n_att = int((out["prediction"] != "Normal").sum())
        m1.metric("แถวทั้งหมด", f"{len(out):,}")
        m2.metric("ทำนายว่าเป็นการโจมตี", f"{n_att:,}", f"{n_att/len(out):.1%}", delta_color="off")
        if y_true is not None and y_true.notna().all():
            m3.metric("Accuracy (เทียบ label ในไฟล์)", f"{(out['prediction'] == y_true).mean():.3f}")

        c1, c2 = st.columns(2)
        with c1:
            vc = out["prediction"].value_counts().reindex(P.LABELS).fillna(0).reset_index()
            vc.columns = ["class", "count"]
            fig = px.bar(vc, x="class", y="count", color="class", color_discrete_map=CLASS_COLORS, text_auto=True,
                         title="จำนวนตามผลทำนาย")
            fig.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig, width="stretch")
        with c2:
            if y_true is not None and y_true.notna().all():
                cm = pd.crosstab(y_true, out["prediction"]) \
                       .reindex(index=P.LABELS, columns=P.LABELS, fill_value=0)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                labels=dict(x="Predicted", y="True"), title="Confusion matrix (ไฟล์นี้)")
                fig.update_layout(height=360)
                st.plotly_chart(fig, width="stretch")
            else:
                fig = px.histogram(out, x="confidence", nbins=20, color_discrete_sequence=["#0f766e"], title="Confidence ของผลทำนาย")
                fig.update_layout(height=360)
                st.plotly_chart(fig, width="stretch")

        st.subheader("ผลทำนายรายแถว")
        shown = out.copy()
        if y_true is not None:
            shown.insert(2, "true", y_true.values)
        st.dataframe(shown, width="stretch", height=350)
        st.download_button("ดาวน์โหลดผลทำนาย (CSV)", shown.to_csv(index=False).encode("utf-8-sig"),
                           "predictions.csv", "text/csv")
