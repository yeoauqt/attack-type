# Network Intrusion Detection (NSL-KDD)

Streamlit front end for the NSL-KDD data engineering pipeline and Random Forest classifier.

## Run locally
```
pip install -r requirements.txt
python train_model.py      # optional: creates model.joblib and sample_test.csv
streamlit run app.py
```
If `model.joblib` is missing, the app trains it automatically on first launch (needs internet, about 1 minute).

## Deploy from GitHub
1. Push this folder to a GitHub repository (`app.py` at the repository root).
2. Go to share.streamlit.io, sign in with GitHub, choose the repository, set main file to `app.py`, and deploy.
3. To skip the first-run training, run `python train_model.py` locally and commit `model.joblib` and
   `sample_test.csv`. Pin your local `scikit-learn` version in `requirements.txt` so the model loads correctly.
