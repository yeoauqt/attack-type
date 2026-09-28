"""
pipeline.py  —  Data Engineering Pipeline สำหรับ NSL-KDD
(ย้ายโค้ดจาก attack_type_nslkdd_DE.ipynb มาเป็นฟังก์ชัน เพื่อให้ Streamlit เรียกใช้ได้)

หลักการ: ทุกการตัดสินใจ (dedup / constant / correlation / top services) คำนวณจาก train เท่านั้น
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label", "difficulty"]

CAT = ["protocol_type", "service", "flag"]
LABELS = ["Normal", "DoS", "Probe", "R2L", "U2R"]
DROP = ["label", "difficulty", "Attack Type"]

DOS = "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb".split()
PROBE = "ipsweep nmap portsweep satan mscan saint".split()
R2L = ("ftp_write guess_passwd imap multihop phf spy warezclient warezmaster xlock xsnoop snmpguess "
       "snmpgetattack httptunnel sendmail named").split()
U2R = "buffer_overflow loadmodule perl rootkit ps sqlattack xterm".split()

PIPELINE_STEPS = [
    ("INPUT", "NSL-KDD\nKDDTrain+ / KDDTest+"),
    ("1 Extraction", "โหลด + ใส่ schema\n(43 คอลัมน์)"),
    ("2 Quality Check", "Missing / Duplicate\n/ Constant / Skew"),
    ("3 Cleaning", "ลบ duplicate\nลบ constant"),
    ("4 Feature Selection", "correlation > 0.95"),
    ("5 Feature Eng.", "ทดลอง + ablation"),
    ("6 Preprocessing", "One-hot (+ Scaler\nเฉพาะ LR)"),
    ("7 Validation", "assert หลังทุกขั้น"),
    ("OUTPUT", "Clean data → Model\n→ Evaluation"),
]


# ------------------------------------------------------------------ 1) Extraction
def _download(fname: str, dest: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    urllib.request.urlretrieve(BASE_URL + fname.replace("+", "%2B"), dest)


def load_raw(train_src=None, test_src=None):
    """โหลด train/test  (ลำดับ: ไฟล์ที่อัปโหลด → data/ → ดาวน์โหลดจาก GitHub)"""
    if train_src is None:
        if not os.path.exists(TRAIN_FILE):
            _download("KDDTrain+.txt", TRAIN_FILE)
        train_src = TRAIN_FILE
    if test_src is None:
        if not os.path.exists(TEST_FILE):
            _download("KDDTest+.txt", TEST_FILE)
        test_src = TEST_FILE
    train = pd.read_csv(train_src, names=COLS)
    test = pd.read_csv(test_src, names=COLS)
    # Validation ของขั้น extraction
    assert train.shape[1] == test.shape[1] == 43, "จำนวนคอลัมน์ไม่ตรง schema"
    assert list(train.columns) == list(test.columns)
    return train, test


def to_category(l: str) -> str:
    if l == "normal":
        return "Normal"
    if l in DOS:
        return "DoS"
    if l in PROBE:
        return "Probe"
    if l in R2L:
        return "R2L"
    if l in U2R:
        return "U2R"
    raise ValueError(f"label ไม่รู้จัก: {l}")


def add_attack_type(train: pd.DataFrame, test: pd.DataFrame):
    train, test = train.copy(), test.copy()
    for d in (train, test):
        d["Attack Type"] = d["label"].map(to_category)
    return train, test


# ------------------------------------------------------------------ 2) Validation + step log
def validate(step, Xtr, ytr, Xte, yte, step_log: list):
    assert Xtr.isna().sum().sum() == 0, f"[{step}] train มี missing"
    assert Xte.isna().sum().sum() == 0, f"[{step}] test มี missing"
    assert len(Xtr) == len(ytr), f"[{step}] len(X_train) != len(y_train)"
    assert len(Xte) == len(yte), f"[{step}] len(X_test) != len(y_test)"
    assert list(Xtr.columns) == list(Xte.columns), f"[{step}] schema train/test ไม่ตรงกัน"
    assert set(CAT).issubset(Xtr.columns), f"[{step}] คอลัมน์ categorical หาย"
    assert set(ytr.unique()).issubset(LABELS), f"[{step}] y_train มี label แปลก"
    step_log.append(dict(step=step, train_rows=len(Xtr), test_rows=len(Xte), n_features=Xtr.shape[1]))


# ------------------------------------------------------------------ 3) Feature engineering
def add_features(d: pd.DataFrame, top_services: set) -> pd.DataFrame:
    d = d.copy()
    d["bytes_ratio"] = d["src_bytes"] / (d["dst_bytes"] + 1)
    d["src_bytes_log"] = np.log1p(d["src_bytes"])
    d["dst_bytes_log"] = np.log1p(d["dst_bytes"])
    d["duration_log"] = np.log1p(d["duration"])
    d["err_rate_mean"] = d[["serror_rate", "rerror_rate"]].mean(axis=1)
    d["srv_ratio"] = d["srv_count"] / (d["count"] + 1)
    d["host_srv_ratio"] = d["dst_host_srv_count"] / (d["dst_host_count"] + 1)
    sus = [c for c in ["hot", "num_failed_logins", "num_compromised", "root_shell",
                       "su_attempted", "num_file_creations", "num_shells", "num_access_files"] if c in d.columns]
    d["suspicious_cnt"] = d[sus].sum(axis=1)
    d["service"] = d["service"].where(d["service"].isin(top_services), "other")
    return d


# ------------------------------------------------------------------ 4) Run whole DE pipeline
def run_de(train: pd.DataFrame, test: pd.DataFrame, corr_threshold: float = 0.95) -> dict:
    """รันทุกขั้นตอน DE แล้วคืนผลลัพธ์ทุกอย่างเป็น dict"""
    step_log: list = []
    X_train_raw, y_train = train.drop(columns=DROP), train["Attack Type"]
    X_test_raw, y_test = test.drop(columns=DROP), test["Attack Type"]
    validate("0 raw", X_train_raw, y_train, X_test_raw, y_test, step_log)

    # --- profiling
    n_missing = int(X_train_raw.isna().sum().sum())
    n_dup_train = int(X_train_raw.duplicated().sum())
    n_dup_test = int(X_test_raw.duplicated().sum())
    const_cols = [c for c in X_train_raw.columns if X_train_raw[c].nunique() == 1]
    skew = X_train_raw.select_dtypes(include="number").skew().abs().sort_values(ascending=False)

    # --- cleaning
    keep = ~X_train_raw.duplicated()
    X_train_de, y_train_de = X_train_raw[keep], y_train[keep]
    X_train_de = X_train_de.drop(columns=const_cols)
    X_test_de = X_test_raw.drop(columns=const_cols)
    validate("1 dedup + drop constant", X_train_de, y_train_de, X_test_de, y_test, step_log)

    # --- correlation
    corr = X_train_de.select_dtypes(include="number").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_drop = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
    pairs = upper.stack().reset_index()
    pairs.columns = ["feature_A", "feature_B", "abs_corr"]
    pairs = pairs[pairs["abs_corr"] > corr_threshold].sort_values("abs_corr", ascending=False).reset_index(drop=True)
    X_train_sel = X_train_de.drop(columns=corr_drop)
    X_test_sel = X_test_de.drop(columns=corr_drop)
    validate(f"2 drop correlated (>{corr_threshold})", X_train_sel, y_train_de, X_test_sel, y_test, step_log)

    # --- feature engineering
    top_services = set(X_train_sel["service"].value_counts().head(15).index)
    X_train_fe = add_features(X_train_sel, top_services)
    X_test_fe = add_features(X_test_sel, top_services)
    validate("3 + engineered features", X_train_fe, y_train_de, X_test_fe, y_test, step_log)

    # --- class distribution / unseen
    counts = pd.DataFrame({"train": train["Attack Type"].value_counts(),
                           "test": test["Attack Type"].value_counts()}).reindex(LABELS).fillna(0).astype(int)
    unseen = sorted(set(test["label"]) - set(train["label"]))
    n_unseen_rows = int(test["label"].isin(unseen).sum())
    min_class = y_train.value_counts().idxmin()

    quality_report = pd.DataFrame([
        ["Missing value", f"{n_missing}", "ไม่ต้องทำอะไร"],
        ["Duplicate rows", f"train {n_dup_train} | test {n_dup_test}", "ลบจาก train (เก็บ test ตามต้นฉบับ)"],
        ["Constant feature", f"{len(const_cols)}: {const_cols}", "ลบ (ไม่มีข้อมูล)"],
        [f"Highly correlated (>{corr_threshold})", f"{len(corr_drop)}: {corr_drop}", "ลบ (ซ้ำซ้อน)"],
        ["Categorical feature", f"{len(CAT)}: {CAT}", "One-hot encoding (handle_unknown=ignore)"],
        ["Skewed numeric", f"{', '.join(skew.index[:3])}, ... (skew > {skew.iloc[4]:.0f})",
         "ทดลอง log-transform ใน feature engineering"],
        ["Class imbalance", f"{min_class}={int(y_train.value_counts().min())} vs "
                            f"Normal={int(y_train.value_counts()['Normal'])}",
         "class_weight='balanced' + รายงาน Macro F1"],
        ["Unseen attack in test", f"{len(unseen)} subtypes ({n_unseen_rows} แถว)",
         "รายงานเป็นข้อจำกัด + error analysis"],
    ], columns=["Data Quality Check", "Result", "Action"])

    step_df = pd.DataFrame(step_log).set_index("step")
    step_df["features_change"] = step_df["n_features"].diff().fillna(0).astype(int)
    step_df["train_rows_change"] = step_df["train_rows"].diff().fillna(0).astype(int)

    return dict(
        setups={
            "0 raw (no engineering)": (X_train_raw, y_train, X_test_raw),
            "1 + dedup + drop constant": (X_train_de, y_train_de, X_test_de),
            "2 + drop correlated": (X_train_sel, y_train_de, X_test_sel),
            "3 + engineered features": (X_train_fe, y_train_de, X_test_fe),
        },
        y_test=y_test, counts=counts, unseen=unseen, n_unseen_rows=n_unseen_rows,
        n_missing=n_missing, n_dup_train=n_dup_train, n_dup_test=n_dup_test,
        const_cols=const_cols, corr_drop=corr_drop, corr_pairs=pairs, skew=skew,
        quality_report=quality_report, step_df=step_df, top_services=top_services,
        corr_matrix=corr,
    )


# ------------------------------------------------------------------ 5) Model
def make_model(X: pd.DataFrame, kind="rf", scale=None, n_est=100) -> Pipeline:
    """kind: 'dummy' | 'lr' | 'rf'.  scale=None -> lr ใช้ scaler, rf/dummy ไม่ใช้"""
    if scale is None:
        scale = kind == "lr"
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
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=8, min_samples_leaf=20,
                                     class_weight="balanced", n_jobs=-1, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(name, Xtr, ytr, Xte, yte, kind="rf", scale=None, n_est=100, cv_splits=3) -> dict:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    m = make_model(Xtr, kind, scale, n_est)
    cv = cross_val_score(m, Xtr, ytr, cv=skf, scoring="f1_macro", n_jobs=1).mean()  # เลือกค่าด้วย train
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    rec = recall_score(yte, p, labels=LABELS, average=None, zero_division=0)
    return dict(setup=name, n_features=Xtr.shape[1], cv_macro_f1=cv,
                train_acc=accuracy_score(ytr, m.predict(Xtr)), test_acc=accuracy_score(yte, p),
                test_macro_f1=f1_score(yte, p, average="macro", zero_division=0),
                recall_R2L=rec[3], recall_U2R=rec[4])


def run_ablation(de: dict, n_est=100) -> pd.DataFrame:
    rows = [evaluate(n, Xtr, ytr, Xte, de["y_test"], n_est=n_est) for n, (Xtr, ytr, Xte) in de["setups"].items()]
    return pd.DataFrame(rows).set_index("setup").round(3)


def run_comparison(de: dict, best_setup: str, n_est=100) -> pd.DataFrame:
    Xtr, ytr, Xte = de["setups"][best_setup]
    yte = de["y_test"]
    rows = [
        evaluate("Dummy (most frequent)", Xtr, ytr, Xte, yte, kind="dummy", n_est=n_est),
        evaluate("Logistic Regression + scaler", Xtr, ytr, Xte, yte, kind="lr", n_est=n_est),
        evaluate("Random Forest (no scaler)", Xtr, ytr, Xte, yte, kind="rf", scale=False, n_est=n_est),
        evaluate("Random Forest + scaler", Xtr, ytr, Xte, yte, kind="rf", scale=True, n_est=n_est),
    ]
    return pd.DataFrame(rows).set_index("setup").round(3)


def error_analysis(test: pd.DataFrame, pred, unseen: list):
    ea = test[["label", "Attack Type"]].copy()
    ea["pred"] = np.asarray(pred)
    ea["correct"] = ea["pred"] == ea["Attack Type"]
    ea["seen_in_train"] = ~ea["label"].isin(unseen)
    rare = ea[ea["Attack Type"].isin(["R2L", "U2R"])]
    sub = (rare.groupby(["Attack Type", "label", "seen_in_train"])
           .agg(n=("correct", "size"), recall=("correct", "mean")).round(3)
           .sort_values("n", ascending=False).reset_index())
    by_seen = (rare.groupby(["Attack Type", "seen_in_train"])
               .agg(n=("correct", "size"), recall=("correct", "mean")).round(3).reset_index())
    r2l_pred = ea.loc[ea["Attack Type"] == "R2L", "pred"].value_counts()
    return sub, by_seen, r2l_pred
