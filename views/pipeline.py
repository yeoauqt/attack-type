import streamlit as st

from ui import ensure_bundle, page_header

STEP_INFO = [
    ("1. Extraction", "Read KDDTrain+ / KDDTest+ (41 features + label [+ difficulty]); "
                        "coerce numeric columns and validate the schema."),
    ("2. Cleaning", "Drop exact-duplicate rows (train only) and columns with a single constant "
                     "value (no information for any model)."),
    ("3. Correlation filter", "Compute the correlation matrix on the training set only, then drop one "
                                "column of every pair with |corr| > 0.95 to avoid redundant, collinear inputs."),
    ("4. Feature engineering", "Add ratios and log-transforms (bytes_ratio, *_log), aggregate suspicious-activity "
                                 "counts, and group rare services into 'other' to control one-hot width."),
]


def render():
    b = ensure_bundle()
    page_header("Pipeline", "Every step below is fitted on the training set only and re-applied "
                             "identically to the test set and to any new traffic uploaded on the Prediction page.")

    cols = st.columns(len(STEP_INFO))
    for col, (title, desc) in zip(cols, STEP_INFO):
        with col:
            st.markdown(
                f"<div class='ng-card' style='min-height:170px'>"
                f"<div class='ng-card-title'>{title}</div>"
                f"<div class='ng-card-sub' style='margin-top:8px; line-height:1.5;'>{desc}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='ng-sep'>", unsafe_allow_html=True)
    st.markdown("#### Row and feature counts at every step")
    log = b["step_log"].rename(columns={
        "step": "Step", "train_rows": "Train rows", "test_rows": "Test rows", "n_features": "Features",
    })
    st.dataframe(log, hide_index=True, use_container_width=True)

    st.markdown("#### What changed between steps")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='ng-card'><div class='ng-card-title'>Rows removed</div>"
            f"<div class='ng-card-value' style='font-size:22px; margin-top:10px'>"
            f"{b['n_dup_train']} duplicate training rows</div>"
            f"<div class='ng-card-sub'>Test rows are never dropped, so the reported test accuracy "
            f"stays comparable across every setup.</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        dropped = b["const_cols"] + b["corr_drop"]
        st.markdown(
            f"<div class='ng-card'><div class='ng-card-title'>Columns removed</div>"
            f"<div class='ng-card-value' style='font-size:22px; margin-top:10px'>{len(dropped)} of 41</div>"
            f"<div class='ng-card-sub'>{', '.join(dropped)}</div></div>",
            unsafe_allow_html=True,
        )

    st.download_button(
        "Download step log (CSV)", log.to_csv(index=False).encode(),
        file_name="pipeline_step_log.csv", mime="text/csv",
    )
