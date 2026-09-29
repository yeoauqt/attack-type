"""
pipeline.py — Data Engineering Pipeline สำหรับ NSL-KDD
ย้ายโค้ดจาก attack_type_nslkdd_DE.ipynb มาเป็นฟังก์ชัน เพื่อให้ Streamlit เรียกใช้ซ้ำได้

หลักการ: ทุกการตัดสินใจ (const_cols, corr_drop, top_services, เลือก setup) ใช้ train เท่านั้น
"""
import os, urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report

SEED = 42
N_EST = 100
BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
"hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
"num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
"serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
"dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
"dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
FEATURES = COLS[:41]
CAT = ["protocol_type", "service", "flag"]
DROP = ["label", "difficulty", "Attack Type"]
LABELS = ["Normal", "DoS", "Probe", "R2L", "U2R"]

_dos   = "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb".split()
_probe = "ipsweep nmap portsweep satan mscan saint".split()
_r2l   = "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster xlock xsnoop snmpguess snmpgetattack httptunnel sendmail named".split()
_u2r   = "buffer_overflow loadmodule perl rootkit ps sqlattack xterm".split()

SETUP_NAMES = ["0 raw (no engineering)", "1 + dedup + drop constant",
               "2 + drop correlated (>0.95)", "3 + engineered features"]


# ---------------------------------------------------------------- 1) Extraction
def ensure_data():
    """ดาวน์โหลด KDDTrain+/KDDTest+ ถ้ายังไม่มีในโฟลเดอร์ data/"""
    os.makedirs(DATA_DIR, exist_ok=True)
    for local, remote in [("KDDTrain.txt", "KDDTrain%2B.txt"), ("KDDTest.txt", "KDDTest%2B.txt")]:
        path = os.path.join(DATA_DIR, local)
        if not os.path.exists(path):
            urllib.request.urlretrieve(BASE + remote, path)


def to_category(l):
    if l == "normal": return "Normal"
    if l in _dos:   return "DoS"
    if l in _probe: return "Probe"
    if l in _r2l:   return "R2L"
    if l in _u2r:   return "U2R"
    raise ValueError(f"label ไม่รู้จัก: {l}")


def load_raw():
    ensure_data()
    train = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain.txt"), names=COLS)
    test = pd.read_csv(os.path.join(DATA_DIR, "KDDTest.txt"), names=COLS)
    assert train.shape[1] == test.shape[1] == 43, "จำนวนคอลัมน์ไม่ตรง schema"
    for d in (train, test):
        d["Attack Type"] = d["label"].map(to_category)
    return train, test


# ---------------------------------------------------------------- 2) Validation
def validate(step, Xtr, ytr, Xte, yte, step_log):
    assert Xtr.isna().sum().sum() == 0, f"[{step}] train มี missing"
    assert Xte.isna().sum().sum() == 0, f"[{step}] test มี missing"
    assert len(Xtr) == len(ytr) and len(Xte) == len(yte), f"[{step}] จำนวนแถว X/y ไม่ตรงกัน"
    assert list(Xtr.columns) == list(Xte.columns), f"[{step}] schema train/test ไม่ตรงกัน"
    assert set(CAT).issubset(Xtr.columns), f"[{step}] คอลัมน์ categorical หาย"
    assert set(ytr.unique()).issubset(LABELS), f"[{step}] y_train มี label แปลก"
    step_log.append(dict(step=step, train_rows=len(Xtr), test_rows=len(Xte), n_features=Xtr.shape[1]))


# ---------------------------------------------------------------- 3) Transform
def add_features(d, top_services):
    d = d.copy()
    d["bytes_ratio"]    = d["src_bytes"] / (d["dst_bytes"] + 1)
    d["src_bytes_log"]  = np.log1p(d["src_bytes"])
    d["dst_bytes_log"]  = np.log1p(d["dst_bytes"])
    d["duration_log"]   = np.log1p(d["duration"])
    d["err_rate_mean"]  = d[["serror_rate", "rerror_rate"]].mean(axis=1)
    d["srv_ratio"]      = d["srv_count"] / (d["count"] + 1)
    d["host_srv_ratio"] = d["dst_host_srv_count"] / (d["dst_host_count"] + 1)
    sus = [c for c in ["hot","num_failed_logins","num_compromised","root_shell",
                       "su_attempted","num_file_creations","num_shells","num_access_files"] if c in d.columns]
    d["suspicious_cnt"] = d[sus].sum(axis=1)
    d["service"] = d["service"].where(d["service"].isin(top_services), "other")
    return d


def profile(X):
    """Data profiling ของ feature ชุดหนึ่ง"""
    num = X.select_dtypes(include="number")
    return dict(
        n_rows=len(X), n_cols=X.shape[1],
        missing=int(X.isna().sum().sum()),
        duplicates=int(X.duplicated().sum()),
        constant=[c for c in X.columns if X[c].nunique() == 1],
        skew=num.skew().abs().sort_values(ascending=False),
    )


def high_corr_pairs(X, thr=0.95):
    corr = X.select_dtypes(include="number").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > thr).any()]
    pairs = upper.stack().reset_index()
    pairs.columns = ["feature_A", "feature_B", "abs_corr"]
    pairs = pairs[pairs["abs_corr"] > thr].sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return drop, pairs, corr


def apply_transform(X, cfg, upto=3):
    """ใช้ transform ที่เรียนรู้จาก train (cfg) กับข้อมูลใหม่ — upto = ขั้นที่ 0..3 ตาม SETUP_NAMES"""
    if upto >= 1:
        X = X.drop(columns=[c for c in cfg["const_cols"] if c in X.columns])
    if upto >= 2:
        X = X.drop(columns=[c for c in cfg["corr_drop"] if c in X.columns])
    if upto >= 3:
        X = add_features(X, cfg["top_services"])
    return X


# ---------------------------------------------------------------- 4) Models
def make_model(X, kind="rf", scale=None):
    if scale is None:
        scale = (kind == "lr")
    num = [c for c in X.columns if c not in CAT]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler() if scale else "passthrough", num),
    ])
    if kind == "dummy":
        clf = DummyClassifier(strategy="most_frequent")
    elif kind == "lr":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    else:
        clf = RandomForestClassifier(n_estimators=N_EST, max_depth=8, min_samples_leaf=20,
                                     class_weight="balanced", n_jobs=-1, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(name, Xtr, ytr, Xte, yte, kind="rf", scale=None):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    m = make_model(Xtr, kind, scale)
    cv = cross_val_score(m, Xtr, ytr, cv=skf, scoring="f1_macro", n_jobs=1).mean()
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    rec = recall_score(yte, p, labels=LABELS, average=None, zero_division=0)
    return dict(setup=name, n_features=Xtr.shape[1], cv_macro_f1=cv,
                train_acc=accuracy_score(ytr, m.predict(Xtr)), test_acc=accuracy_score(yte, p),
                test_macro_f1=f1_score(yte, p, average="macro", zero_division=0),
                recall_R2L=rec[3], recall_U2R=rec[4])


# ---------------------------------------------------------------- 5) Run everything
def run_all(progress=None):
    """รันทั้ง pipeline แล้วคืน artifacts ทั้งหมดที่หน้าเว็บต้องใช้"""
    say = progress or (lambda *_: None)
    say("โหลดข้อมูล", 0.05)
    train, test = load_raw()
    step_log = []

    X_train_raw, y_train = train.drop(columns=DROP), train["Attack Type"]
    X_test_raw, y_test = test.drop(columns=DROP), test["Attack Type"]
    validate("0 raw", X_train_raw, y_train, X_test_raw, y_test, step_log)

    prof = profile(X_train_raw)
    n_dup_test = int(X_test_raw.duplicated().sum())
    const_cols = prof["constant"]

    keep = ~X_train_raw.duplicated()
    X_train_de, y_train_de = X_train_raw[keep].drop(columns=const_cols), y_train[keep]
    X_test_de = X_test_raw.drop(columns=const_cols)
    validate("1 dedup + drop constant", X_train_de, y_train_de, X_test_de, y_test, step_log)

    corr_drop, pairs, corr = high_corr_pairs(X_train_de)
    X_train_sel, X_test_sel = X_train_de.drop(columns=corr_drop), X_test_de.drop(columns=corr_drop)
    validate("2 drop correlated (>0.95)", X_train_sel, y_train_de, X_test_sel, y_test, step_log)

    top_services = list(X_train_sel["service"].value_counts().head(15).index)
    X_train_fe, X_test_fe = add_features(X_train_sel, top_services), add_features(X_test_sel, top_services)
    validate("3 + engineered features", X_train_fe, y_train_de, X_test_fe, y_test, step_log)

    unseen = sorted(set(test["label"]) - set(train["label"]))
    cfg = dict(const_cols=const_cols, corr_drop=corr_drop, top_services=top_services)

    setups = {
        SETUP_NAMES[0]: (X_train_raw, y_train, X_test_raw),
        SETUP_NAMES[1]: (X_train_de, y_train_de, X_test_de),
        SETUP_NAMES[2]: (X_train_sel, y_train_de, X_test_sel),
        SETUP_NAMES[3]: (X_train_fe, y_train_de, X_test_fe),
    }
    results = []
    for i, (n, (Xtr, ytr, Xte)) in enumerate(setups.items()):
        say(f"Ablation: {n}", 0.15 + 0.15 * i)
        results.append(evaluate(n, Xtr, ytr, Xte, y_test))
    ablation = pd.DataFrame(results).set_index("setup")
    best_setup = ablation["cv_macro_f1"].idxmax()          # เลือกจาก CV บน train เท่านั้น
    best_idx = SETUP_NAMES.index(best_setup)
    Xtr_f, ytr_f, Xte_f = setups[best_setup]
    cfg["best_setup"], cfg["best_idx"] = best_setup, best_idx
    cfg["dedup_train"] = best_idx >= 1

    say("เปรียบเทียบโมเดล", 0.78)
    comparison = pd.DataFrame([
        evaluate("Dummy (most frequent)", Xtr_f, ytr_f, Xte_f, y_test, kind="dummy"),
        evaluate("Logistic Regression + scaler", Xtr_f, ytr_f, Xte_f, y_test, kind="lr"),
        evaluate("Random Forest (no scaler)", Xtr_f, ytr_f, Xte_f, y_test, kind="rf", scale=False),
        evaluate("Random Forest + scaler", Xtr_f, ytr_f, Xte_f, y_test, kind="rf", scale=True),
    ]).set_index("setup")

    say("เทรนโมเดลสุดท้าย", 0.92)
    final = make_model(Xtr_f, "rf", scale=False).fit(Xtr_f, ytr_f)
    pred = final.predict(Xte_f)
    names = final.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(final.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)

    ea = test[["label", "Attack Type"]].copy()
    ea["pred"] = pred
    ea["correct"] = ea["pred"] == ea["Attack Type"]
    ea["seen_in_train"] = ~ea["label"].isin(unseen)

    min_class = y_train.value_counts().idxmin()
    quality_report = pd.DataFrame([
        ["Missing value", f"{prof['missing']}", "ไม่ต้องทำอะไร"],
        ["Duplicate rows", f"train {prof['duplicates']:,} | test {n_dup_test:,}", "ลบจาก train (เก็บ test ตามต้นฉบับ)"],
        ["Constant feature", f"{len(const_cols)}: {const_cols}", "ลบ (ไม่มีข้อมูล)"],
        ["Highly correlated (>0.95)", f"{len(corr_drop)}: {corr_drop}", "ลบ (ซ้ำซ้อน)"],
        ["Categorical feature", f"{len(CAT)}: {CAT}", "One-hot encoding (handle_unknown=ignore)"],
        ["Skewed numeric", f"{prof['skew'].index[0]} skew={prof['skew'].iloc[0]:.0f}", "ทดลอง log-transform ใน feature engineering"],
        ["Class imbalance", f"{min_class}={int(y_train.value_counts().min()):,} vs Normal={int(y_train.value_counts()['Normal']):,}",
         "class_weight='balanced' + รายงาน Macro F1"],
        ["Unseen attack in test", f"{len(unseen)} subtypes ({int(test['label'].isin(unseen).sum()):,} แถว)",
         "รายงานเป็นข้อจำกัด + error analysis"],
    ], columns=["Data Quality Check", "Result", "Action"])

    counts = pd.DataFrame({"train": train["Attack Type"].value_counts(),
                           "test": test["Attack Type"].value_counts()}).loc[LABELS]

    cm = confusion_matrix(y_test, pred, labels=LABELS)
    report = pd.DataFrame(classification_report(y_test, pred, labels=LABELS, digits=3,
                                                zero_division=0, output_dict=True)).T
    say("เสร็จแล้ว", 1.0)
    return dict(
        cfg=cfg, model=final, step_log=pd.DataFrame(step_log), quality_report=quality_report,
        pairs=pairs, corr=corr, skew=prof["skew"], counts=counts, ablation=ablation,
        comparison=comparison, importance=imp, confusion=cm, report=report, error=ea, unseen=unseen,
        train_acc=accuracy_score(ytr_f, final.predict(Xtr_f)), test_acc=accuracy_score(y_test, pred),
        test_macro_f1=f1_score(y_test, pred, average="macro", zero_division=0),
        n_dup_train=prof["duplicates"], n_dup_test=n_dup_test,
    )


# ---------------------------------------------------------------- 6) Inference on uploaded data
def read_uploaded(file):
    """อ่านไฟล์ CSV/TXT ที่ผู้ใช้อัปโหลด รองรับทั้งมี header และไม่มี header (41/42/43 คอลัมน์)"""
    df = pd.read_csv(file, header=None, dtype=str)
    first = [str(v).strip() for v in df.iloc[0].tolist()]
    if "duration" in first and "protocol_type" in first:
        file.seek(0)
        return pd.read_csv(file)
    n = df.shape[1]
    if n not in (41, 42, 43):
        raise ValueError(f"ไฟล์มี {n} คอลัมน์ แต่ต้องมี 41 (features) / 42 (+label) / 43 (+label+difficulty) คอลัมน์")
    file.seek(0)
    return pd.read_csv(file, names=COLS[:n])


def check_upload(df):
    """Data validation ของไฟล์ที่อัปโหลด — คืน (issues, cleaned_X, y_true|None)"""
    issues = []
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ขาดคอลัมน์: {missing_cols}")
    X = df[FEATURES].copy()
    num_cols = [c for c in FEATURES if c not in CAT]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    n_bad = int(X[num_cols].isna().sum().sum())
    if n_bad:
        issues.append(f"พบค่า numeric ที่แปลงไม่ได้/ว่าง {n_bad} ช่อง → เติมด้วย median ของแต่ละคอลัมน์")
        X[num_cols] = X[num_cols].fillna(X[num_cols].median()).fillna(0)
    n_dup = int(X.duplicated().sum())
    if n_dup:
        issues.append(f"มีแถวซ้ำ {n_dup} แถว (ยังคงไว้ เพื่อให้จำนวนผลทำนายตรงกับไฟล์)")
    y = None
    if "label" in df.columns:
        try:
            y = df["label"].map(to_category)
        except ValueError as e:
            issues.append(f"label บางค่าไม่รู้จัก ({e}) → ไม่คำนวณ metric")
    return issues, X, y
