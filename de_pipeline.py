"""
NetGuard - Data Engineering pipeline for NSL-KDD (204426 group project).

All decisions (duplicates, constant columns, correlated features, top services)
are learned from the TRAIN set only, then re-applied to test / new traffic.
This module has no Streamlit dependency so it can be tested on its own.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
CORR_THRESHOLD = 0.95
BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"

FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CAT = ["protocol_type", "service", "flag"]
NUM = [c for c in FEATURES if c not in CAT]
RATE_COLS = [c for c in NUM if c.endswith("_rate")]
LABELS = ["Normal", "DoS", "Probe", "R2L", "U2R"]

_DOS = "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb".split()
_PROBE = "ipsweep nmap portsweep satan mscan saint".split()
_R2L = ("ftp_write guess_passwd imap multihop phf spy warezclient warezmaster xlock xsnoop "
        "snmpguess snmpgetattack httptunnel sendmail named").split()
_U2R = "buffer_overflow loadmodule perl rootkit ps sqlattack xterm".split()

SETUP_NAMES = [
    "0 Raw (no engineering)",
    "1 Dedup + drop constant",
    "2 + Drop correlated (>0.95)",
    "3 + Engineered features",
]


def to_category(label: str) -> str:
    label = str(label).strip().lower().rstrip(".")
    if label == "normal":
        return "Normal"
    if label in _DOS:
        return "DoS"
    if label in _PROBE:
        return "Probe"
    if label in _R2L:
        return "R2L"
    if label in _U2R:
        return "U2R"
    return "Unknown"


# --------------------------------------------------------------------------- #
# 1. Extraction
# --------------------------------------------------------------------------- #
def read_nsl(src) -> pd.DataFrame:
    """Read a KDD-style file (41 features [+ label [+ difficulty]]) and attach the schema."""
    df = pd.read_csv(src, header=None, low_memory=False, skipinitialspace=True)
    if str(df.iloc[0, 0]).strip().lower() == "duration":
        df = df.iloc[1:].reset_index(drop=True)
    n = df.shape[1]
    if n == 43:
        df.columns = FEATURES + ["label", "difficulty"]
    elif n == 42:
        df.columns = FEATURES + ["label"]
        df["difficulty"] = np.nan
    else:
        raise ValueError(f"Expected 42 or 43 columns (41 features + label [+ difficulty]); found {n}.")
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df["label"] = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
    return df


def load_one(upload: bytes | None, fname: str) -> tuple[pd.DataFrame, str]:
    """Priority: uploaded file -> data/ folder in the repo -> download from GitHub."""
    if upload is not None:
        return read_nsl(io.BytesIO(upload)), "uploaded file"
    local = os.path.join("data", fname)
    if os.path.exists(local):
        return read_nsl(local), "data/ folder"
    return read_nsl(BASE_URL + fname), "downloaded from GitHub"


# --------------------------------------------------------------------------- #
# 2. Feature helpers
# --------------------------------------------------------------------------- #
def corr_pairs(corr: pd.DataFrame, thr: float) -> pd.DataFrame:
    cols = list(corr.columns)
    iu = np.triu_indices(len(cols), k=1)
    vals = corr.values[iu]
    m = vals > thr
    out = pd.DataFrame({
        "feature_A": [cols[i] for i in iu[0][m]],
        "feature_B": [cols[j] for j in iu[1][m]],
        "abs_corr": vals[m],
    })
    return out.sort_values("abs_corr", ascending=False).reset_index(drop=True)


def corr_drop_list(corr: pd.DataFrame, thr: float) -> list[str]:
    """Drop the later column of every pair whose |corr| exceeds the threshold."""
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if (upper[c] > thr).any()]


def add_features(d: pd.DataFrame, top_services: set) -> pd.DataFrame:
    d = d.copy()
    has = lambda *cs: all(c in d.columns for c in cs)  # noqa: E731
    if has("src_bytes", "dst_bytes"):
        d["bytes_ratio"] = d["src_bytes"] / (d["dst_bytes"] + 1)
    for c in ("src_bytes", "dst_bytes", "duration"):
        if c in d.columns:
            d[f"{c}_log"] = np.log1p(d[c])
    if has("serror_rate", "rerror_rate"):
        d["err_rate_mean"] = d[["serror_rate", "rerror_rate"]].mean(axis=1)
    if has("srv_count", "count"):
        d["srv_ratio"] = d["srv_count"] / (d["count"] + 1)
    if has("dst_host_srv_count", "dst_host_count"):
        d["host_srv_ratio"] = d["dst_host_srv_count"] / (d["dst_host_count"] + 1)
    sus = [c for c in ["hot", "num_failed_logins", "num_compromised", "root_shell", "su_attempted",
                       "num_file_creations", "num_shells", "num_access_files"] if c in d.columns]
    if sus:
        d["suspicious_cnt"] = d[sus].sum(axis=1)
    d["service"] = d["service"].where(d["service"].isin(top_services), "other")
    return d


ENGINEERED_TABLE = pd.DataFrame([
    ["bytes_ratio", "src_bytes / (dst_bytes + 1)", "Direction of traffic (upload vs download heavy)"],
    ["src_bytes_log, dst_bytes_log, duration_log", "log(1 + x)", "Reduce extreme right skew"],
    ["err_rate_mean", "mean(serror_rate, rerror_rate)", "Single signal for connection errors"],
    ["srv_ratio", "srv_count / (count + 1)", "Share of connections to the same service"],
    ["host_srv_ratio", "dst_host_srv_count / (dst_host_count + 1)", "Same idea at destination-host level"],
    ["suspicious_cnt", "sum of hot, failed logins, compromised, ...", "Aggregate of R2L / U2R indicators"],
    ["service (grouped)", "keep top-15 services, rest = 'other'", "Reduce one-hot width and rare categories"],
], columns=["Feature", "Formula", "Purpose"])


# --------------------------------------------------------------------------- #
# 3. Pipeline preparation (fit on TRAIN only)
# --------------------------------------------------------------------------- #
def validate_step(step, Xtr, ytr, Xte, yte, log: list):
    assert Xtr.isna().sum().sum() == 0, f"[{step}] train has missing values"
    assert Xte.isna().sum().sum() == 0, f"[{step}] test has missing values"
    assert len(Xtr) == len(ytr), f"[{step}] len(X_train) != len(y_train)"
    assert len(Xte) == len(yte), f"[{step}] len(X_test) != len(y_test)"
    assert list(Xtr.columns) == list(Xte.columns), f"[{step}] train/test schema mismatch"
    assert set(CAT).issubset(Xtr.columns), f"[{step}] categorical columns missing"
    assert set(ytr.unique()).issubset(LABELS), f"[{step}] unexpected labels in y_train"
    log.append(dict(step=step, train_rows=len(Xtr), test_rows=len(Xte), n_features=Xtr.shape[1]))


def prepare(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    train, test = train.copy(), test.copy()
    for d in (train, test):
        d["Attack Type"] = d["label"].map(to_category)
    unknown = sorted(set(train.loc[train["Attack Type"] == "Unknown", "label"])
                     | set(test.loc[test["Attack Type"] == "Unknown", "label"]))
    if unknown:
        raise ValueError(f"Unknown attack labels found: {unknown}")

    drop = ["label", "difficulty", "Attack Type"]
    X_train_raw, y_train = train.drop(columns=drop), train["Attack Type"]
    X_test_raw, y_test = test.drop(columns=drop), test["Attack Type"]
    log: list = []
    validate_step("0 Raw", X_train_raw, y_train, X_test_raw, y_test, log)

    # profiling (train)
    n_missing = int(X_train_raw.isna().sum().sum())
    n_dup_train = int(X_train_raw.duplicated().sum())
    n_dup_test = int(X_test_raw.duplicated().sum())
    const_cols = [c for c in X_train_raw.columns if X_train_raw[c].nunique() == 1]
    skew = X_train_raw[NUM].skew().abs().sort_values(ascending=False)

    # cleaning
    keep = ~X_train_raw.duplicated()
    X_train_de, y_train_de = X_train_raw[keep].drop(columns=const_cols), y_train[keep]
    X_test_de = X_test_raw.drop(columns=const_cols)
    validate_step("1 Dedup + drop constant", X_train_de, y_train_de, X_test_de, y_test, log)

    # correlation filter (train only)
    corr = X_train_de.select_dtypes("number").corr().abs()
    corr_drop = corr_drop_list(corr, CORR_THRESHOLD)
    pairs = corr_pairs(corr, CORR_THRESHOLD)
    X_train_sel, X_test_sel = X_train_de.drop(columns=corr_drop), X_test_de.drop(columns=corr_drop)
    validate_step("2 Drop correlated", X_train_sel, y_train_de, X_test_sel, y_test, log)

    # feature engineering
    top_services = set(X_train_sel["service"].value_counts().head(15).index)
    X_train_fe, X_test_fe = add_features(X_train_sel, top_services), add_features(X_test_sel, top_services)
    validate_step("3 Engineered features", X_train_fe, y_train_de, X_test_fe, y_test, log)

    counts = (pd.DataFrame({"train": y_train.value_counts(), "test": y_test.value_counts()})
              .reindex(LABELS).fillna(0).astype(int))
    unseen = sorted(set(test["label"]) - set(train["label"]))

    return dict(
        train=train, test=test, counts=counts, unseen=unseen,
        X_train_raw=X_train_raw, y_train=y_train, X_test_raw=X_test_raw, y_test=y_test,
        X_train_de=X_train_de, y_train_de=y_train_de, X_test_de=X_test_de,
        X_train_sel=X_train_sel, X_test_sel=X_test_sel,
        X_train_fe=X_train_fe, X_test_fe=X_test_fe,
        n_missing=n_missing, n_dup_train=n_dup_train, n_dup_test=n_dup_test,
        const_cols=const_cols, corr=corr, corr_drop=corr_drop, pairs=pairs, skew=skew,
        top_services=top_services, step_log=pd.DataFrame(log),
        cats={c: set(train[c].unique()) for c in CAT},
    )


def quality_report(b: dict) -> pd.DataFrame:
    tr = b["counts"]["train"]
    top_skew = ", ".join(f"{k} ({v:.1f})" for k, v in b["skew"].head(3).items())
    rows = [
        ["Missing values", f"{b['n_missing']}", "None required (checked by validation after every step)"],
        ["Duplicate rows", f"train {b['n_dup_train']} | test {b['n_dup_test']}",
         "Remove from train; keep test as published"],
        ["Constant features", f"{len(b['const_cols'])}: {', '.join(b['const_cols']) or '-'}", "Drop (no information)"],
        [f"Highly correlated (>{CORR_THRESHOLD})", f"{len(b['corr_drop'])}: {', '.join(b['corr_drop']) or '-'}",
         "Drop the redundant column of each pair (fit on train only)"],
        ["Categorical features", f"{len(CAT)}: {', '.join(CAT)}", "One-hot encoding, handle_unknown='ignore'"],
        ["Skewed numeric features", top_skew, "Test log(1+x) inside the ablation study"],
        ["Class imbalance", f"U2R = {tr['U2R']:,} vs Normal = {tr['Normal']:,}",
         "class_weight='balanced' and report macro F1"],
        ["Train / test drift", f"{len(b['unseen'])} attack sub-types appear only in test",
         "Report per sub-type recall in Error Analysis"],
    ]
    return pd.DataFrame(rows, columns=["Issue", "Finding", "Action"])


def transform_for_setup(X_raw: pd.DataFrame, b: dict, setup: str):
    """Apply the fitted cleaning steps to any raw traffic frame. Returns (frame, step list)."""
    X = X_raw[FEATURES].copy()
    steps = [("Validated raw schema", X.shape[1])]
    if setup == SETUP_NAMES[0]:
        return X, steps
    X = X.drop(columns=b["const_cols"])
    steps.append(("Drop constant columns", X.shape[1]))
    if setup == SETUP_NAMES[1]:
        return X, steps
    X = X.drop(columns=b["corr_drop"])
    steps.append((f"Drop correlated columns (>{CORR_THRESHOLD})", X.shape[1]))
    if setup == SETUP_NAMES[2]:
        return X, steps
    X = add_features(X, b["top_services"])
    steps.append(("Add engineered features", X.shape[1]))
    return X, steps


def setup_frames(b: dict) -> dict:
    return {
        SETUP_NAMES[0]: (b["X_train_raw"], b["y_train"], b["X_test_raw"]),
        SETUP_NAMES[1]: (b["X_train_de"], b["y_train_de"], b["X_test_de"]),
        SETUP_NAMES[2]: (b["X_train_sel"], b["y_train_de"], b["X_test_sel"]),
        SETUP_NAMES[3]: (b["X_train_fe"], b["y_train_de"], b["X_test_fe"]),
    }


# --------------------------------------------------------------------------- #
# 4. Upload validation (used by the Prediction page)
# --------------------------------------------------------------------------- #
def validate_upload(raw: bytes, b: dict):
    """Return (X, y_true_or_None, checks). X is None when a blocking check fails."""
    checks: list[dict] = []

    def add(name, status, detail):
        checks.append(dict(Check=name, Status=status, Detail=detail))

    try:
        df = pd.read_csv(io.BytesIO(raw), header=None, low_memory=False, skipinitialspace=True)
    except Exception as e:  # noqa: BLE001
        add("File parsing", "FAIL", str(e)[:200])
        return None, None, checks
    if df.empty:
        add("File parsing", "FAIL", "File contains no rows")
        return None, None, checks
    add("File parsing", "PASS", f"{len(df):,} rows, {df.shape[1]} columns")

    if str(df.iloc[0, 0]).strip().lower() == "duration":
        df = df.iloc[1:].reset_index(drop=True)
        add("Header row", "PASS", "Header detected and removed")

    n = df.shape[1]
    if n not in (41, 42, 43):
        add("Column count", "FAIL", f"Expected 41 (features), 42 (+ label) or 43 (+ label, difficulty); found {n}")
        return None, None, checks
    add("Column count", "PASS", f"{n} columns matched to the NSL-KDD schema")
    df.columns = FEATURES + (["label"] if n >= 42 else []) + (["difficulty"] if n == 43 else [])

    coerced = df[NUM].apply(pd.to_numeric, errors="coerce")
    bad = int((coerced.isna() & df[NUM].notna()).sum().sum())
    if bad:
        add("Numeric types", "FAIL", f"{bad:,} non-numeric values in numeric columns")
        return None, None, checks
    add("Numeric types", "PASS", f"{len(NUM)} numeric columns parsed")

    missing = int(coerced.isna().sum().sum() + df[CAT].isna().sum().sum())
    if missing:
        add("Missing values", "FAIL", f"{missing:,} missing cells")
        return None, None, checks
    add("Missing values", "PASS", "No missing cells")

    unk = {c: sorted(set(df[c].astype(str)) - b["cats"][c]) for c in CAT}
    unk = {c: v for c, v in unk.items() if v}
    if unk:
        add("Unseen categories", "WARN",
            "; ".join(f"{c}: {', '.join(v[:5])}" for c, v in unk.items()) + " (one-hot ignores them)")
    else:
        add("Unseen categories", "PASS", "All categorical values were seen in training")

    rate_bad = int(((coerced[RATE_COLS] < 0) | (coerced[RATE_COLS] > 1)).sum().sum())
    add("Rate range [0, 1]", "WARN" if rate_bad else "PASS",
        f"{rate_bad:,} rate values outside [0, 1]" if rate_bad else "All rate columns within range")

    X = df[FEATURES].copy()
    X[NUM] = coerced
    ndup = int(X.duplicated().sum())
    add("Duplicate rows", "WARN" if ndup else "PASS",
        f"{ndup:,} duplicates (reported, not removed so output rows match the input)" if ndup else "No duplicates")

    y = None
    if "label" in df.columns:
        mapped = df["label"].astype(str).map(to_category)
        if (mapped == "Unknown").any():
            add("Ground-truth labels", "WARN", "Unrecognised labels found; accuracy will not be computed")
        else:
            y = mapped
            add("Ground-truth labels", "PASS", "Labels mapped to 5 categories for evaluation")
    return X, y, checks


# --------------------------------------------------------------------------- #
# 5. Models and evaluation
# --------------------------------------------------------------------------- #
def make_model(X, kind="rf", scale=None, n_est=60):
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
        clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    else:
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=8, min_samples_leaf=20,
                                     class_weight="balanced", n_jobs=-1, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(name, Xtr, ytr, Xte, yte, kind="rf", scale=None, n_est=60, cv=True) -> dict:
    row = {"setup": name, "n_features": Xtr.shape[1]}
    if cv:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        row["cv_macro_f1"] = float(cross_val_score(make_model(Xtr, kind, scale, n_est), Xtr, ytr,
                                                   cv=skf, scoring="f1_macro").mean())
    model = make_model(Xtr, kind, scale, n_est).fit(Xtr, ytr)
    pred = model.predict(Xte)
    rec = recall_score(yte, pred, labels=LABELS, average=None, zero_division=0)
    row.update(test_acc=accuracy_score(yte, pred),
               test_macro_f1=f1_score(yte, pred, average="macro", labels=LABELS, zero_division=0),
               R2L_recall=rec[LABELS.index("R2L")], U2R_recall=rec[LABELS.index("U2R")])
    return row


def run_ablation(b: dict, n_est: int, cv: bool = False) -> pd.DataFrame:
    rows = [evaluate(n, Xtr, ytr, Xte, b["y_test"], n_est=n_est, cv=cv)
            for n, (Xtr, ytr, Xte) in setup_frames(b).items()]
    return pd.DataFrame(rows).set_index("setup")


def run_comparison(b: dict, setup: str, n_est: int) -> pd.DataFrame:
    Xtr, ytr, Xte = setup_frames(b)[setup]
    yte = b["y_test"]
    rows = [
        evaluate("Dummy (most frequent)", Xtr, ytr, Xte, yte, kind="dummy", cv=False),
        evaluate("Logistic Regression + scaler", Xtr, ytr, Xte, yte, kind="lr", cv=False),
        evaluate("Random Forest (no scaler)", Xtr, ytr, Xte, yte, kind="rf", scale=False, n_est=n_est, cv=False),
        evaluate("Random Forest + scaler", Xtr, ytr, Xte, yte, kind="rf", scale=True, n_est=n_est, cv=False),
    ]
    return pd.DataFrame(rows).set_index("setup")


def build_final(b: dict, setup: str, n_est: int) -> dict:
    Xtr, ytr, Xte = setup_frames(b)[setup]
    model = make_model(Xtr, "rf", False, n_est).fit(Xtr, ytr)
    pred = model.predict(Xte)
    names = [n.split("__", 1)[-1] for n in model.named_steps["pre"].get_feature_names_out()]
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    return dict(setup=setup, model=model, pred=pred, importance=imp.head(15),
                train_acc=accuracy_score(ytr, model.predict(Xtr)),
                test_acc=accuracy_score(b["y_test"], pred))


def error_tables(b: dict, pred) -> dict:
    yte = b["y_test"]
    report = pd.DataFrame(classification_report(yte, pred, labels=LABELS, output_dict=True, zero_division=0)).T
    report = report.loc[LABELS + ["macro avg", "weighted avg"], ["precision", "recall", "f1-score", "support"]].round(3)
    cm = confusion_matrix(yte, pred, labels=LABELS)
    cm_norm = confusion_matrix(yte, pred, labels=LABELS, normalize="true")

    ea = b["test"][["label", "Attack Type"]].copy()
    ea["pred"] = pred
    ea["correct"] = ea["pred"] == ea["Attack Type"]
    ea["seen_in_train"] = ~ea["label"].isin(b["unseen"])
    rare = ea[ea["Attack Type"].isin(["R2L", "U2R"])]
    sub = (rare.groupby(["Attack Type", "label", "seen_in_train"])
           .agg(n=("correct", "size"), recall=("correct", "mean")).round(3)
           .sort_values("n", ascending=False).reset_index())
    by_seen = (rare.groupby(["Attack Type", "seen_in_train"])
               .agg(n=("correct", "size"), recall=("correct", "mean")).round(3).reset_index())
    r2l_as = ea.loc[ea["Attack Type"] == "R2L", "pred"].value_counts().reindex(LABELS).fillna(0).astype(int)
    return dict(report=report, cm=cm, cm_norm=cm_norm, sub=sub, by_seen=by_seen, r2l_as=r2l_as)
