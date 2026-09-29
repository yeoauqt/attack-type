# NetGuard — NSL-KDD Intrusion Detection Dashboard

A Streamlit dashboard built for course 204426 (Data Engineering group project). It classifies
NSL-KDD network connections into **Normal / DoS / Probe / R2L / U2R** and, more importantly,
makes the *data engineering* work itself visible: schema validation, deduplication, correlation
filtering and feature engineering, all fitted on the training set only and re-applied to new
traffic on the Prediction page.

## Pages

| Page | What it shows |
|---|---|
| Upload & Data Quality | Dataset overview, per-class counts, and profiling findings (missing values, duplicates, constant columns, skew, correlation heatmap) |
| Pipeline Explorer | The three cleaning steps with row/feature counts before and after each one, the engineered-feature table, and an ablation study that isolates what each step actually contributes |
| Predict & Visualize | Upload new traffic (with schema validation) or edit a single record and score it end to end, plus confusion matrix, feature importance, and a baseline comparison against Dummy/Logistic Regression |

## Project structure

```
netguard/
├── app.py                 # entry point: theme, sidebar, page routing
├── ui.py                   # shared chrome (topbar, cards, bundle loading)
├── de_pipeline.py          # all data engineering + modelling logic (no Streamlit dependency)
├── views/                  # one module per page
├── assets/style.css        # theme
├── data/                   # optional: put KDDTrain+.txt / KDDTest+.txt here
├── .streamlit/config.toml  # color theme
└── requirements.txt
```

If `data/KDDTrain+.txt` and `data/KDDTest+.txt` are not present and nothing is uploaded from the
sidebar, the app downloads the standard NSL-KDD release from a public GitHub mirror
(`defcom17/NSL_KDD`) at first run.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy from GitHub (Streamlit Community Cloud)

1. Push this folder to a new GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click
   **New app**.
3. Pick the repository/branch and set **Main file path** to `app.py`.
4. Deploy. The first load takes a little longer while the dataset downloads and the Random
   Forests for the Ablation/Models/Error Analysis pages are fitted.

## Notes for the write-up (บทที่ 3)

`de_pipeline.py` is written so every technique used in the report maps to one function:

- `read_nsl`, `validate_upload` → schema validation / extraction
- `prepare` (dedup + constant-column removal) → data cleaning
- `corr_drop_list` / `corr_pairs` → correlation-based feature selection (fit on train only)
- `add_features` → feature engineering (ratios, log-transforms, service grouping)
- `run_ablation` → ablation study proving which step actually helps
- `error_tables` → seen-vs-unseen attack sub-type analysis for R2L/U2R
