"""Train the NSL-KDD classifier and export model.joblib + sample_test.csv.
Mirrors the notebook: dedup -> drop constant -> drop correlated (>0.95) -> one-hot -> Random Forest."""
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
        "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
        "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label","difficulty"]
RAW, CAT = COLS[:41], ["protocol_type", "service", "flag"]
LABELS = ["Normal", "DoS", "Probe", "R2L", "U2R"]
GROUPS = {
    "DoS": "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb",
    "Probe": "ipsweep nmap portsweep satan mscan saint",
    "R2L": "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster xlock xsnoop snmpguess snmpgetattack httptunnel sendmail named",
    "U2R": "buffer_overflow loadmodule perl rootkit ps sqlattack xterm",
}
LOOKUP = {a: g for g, s in GROUPS.items() for a in s.split()}


def category(label):
    if label == "normal":
        return "Normal"
    return LOOKUP[label]


def main(seed=42):
    for remote, local in (("KDDTrain%2B.txt", "KDDTrain.txt"), ("KDDTest%2B.txt", "KDDTest.txt")):
        urllib.request.urlretrieve(BASE + remote, local)
    tr = pd.read_csv("KDDTrain.txt", names=COLS)
    te = pd.read_csv("KDDTest.txt", names=COLS)
    for d in (tr, te):
        d["Attack Type"] = d["label"].map(category)

    # Cleaning (train only): duplicates, constant columns, highly correlated columns
    keep = ~tr[RAW].duplicated()
    Xtr, ytr = tr.loc[keep, RAW], tr.loc[keep, "Attack Type"]
    const = [c for c in RAW if Xtr[c].nunique() == 1]
    corr = Xtr.select_dtypes("number").drop(columns=const, errors="ignore").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = const + [c for c in upper.columns if (upper[c] > 0.95).any()]
    Xtr_s, Xte_s = Xtr.drop(columns=drop), te[RAW].drop(columns=drop)

    num = [c for c in Xtr_s.columns if c not in CAT]
    model = Pipeline([
        ("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                                   ("num", "passthrough", num)])),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=20,
                                       class_weight="balanced", n_jobs=-1, random_state=seed)),
    ]).fit(Xtr_s, ytr)

    pred = model.predict(Xte_s)
    yte = te["Attack Type"]
    metrics = {
        "acc": accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "recall": dict(zip(LABELS, recall_score(yte, pred, labels=LABELS, average=None, zero_division=0))),
        "support": yte.value_counts().reindex(LABELS).to_dict(),
        "n_train": len(Xtr_s), "n_features": Xtr_s.shape[1],
    }

    # Feature importance aggregated back to the original columns
    imp = {}
    for n, v in zip(model.named_steps["pre"].get_feature_names_out(), model.named_steps["clf"].feature_importances_):
        n = n.split("__", 1)[1]
        base = next((c for c in CAT if n.startswith(c + "_")), n)
        imp[base] = imp.get(base, 0.0) + float(v)

    defaults = {c: (str(Xtr[c].mode()[0]) if c in CAT else float(Xtr[c].median())) for c in RAW}
    opts = {c: sorted(set(tr[c]) | set(te[c])) for c in CAT}
    joblib.dump({"model": model, "raw": RAW, "drop": drop, "defaults": defaults, "opts": opts,
                 "importance": imp, "metrics": metrics}, "model.joblib", compress=3)
    te.sample(3000, random_state=seed)[RAW + ["Attack Type"]].to_csv("sample_test.csv", index=False)


if __name__ == "__main__":
    main()
