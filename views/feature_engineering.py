import streamlit as st

from de_pipeline import ENGINEERED_TABLE
from ui import ensure_bundle, page_header


def render():
    b = ensure_bundle()
    page_header("Feature Engineering", "New columns added after the correlation filter, "
                                        "each with the formula used and the reasoning behind it.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Features before engineering", b["X_train_sel"].shape[1])
    c2.metric("Features after engineering", b["X_train_fe"].shape[1])
    c3.metric("Services kept", len(b["top_services"]), help="Remaining services are grouped into 'other'.")

    st.dataframe(ENGINEERED_TABLE, hide_index=True, use_container_width=True)

    st.markdown("<hr class='ng-sep'>", unsafe_allow_html=True)
    st.markdown("#### Preview of the engineered training set")
    new_cols = [c for c in b["X_train_fe"].columns if c not in b["X_train_sel"].columns]
    st.dataframe(b["X_train_fe"][["service"] + new_cols].head(8), hide_index=True, use_container_width=True)

    st.markdown("#### Services kept vs. grouped as 'other'")
    kept = ", ".join(sorted(b["top_services"]))
    st.caption(f"Kept ({len(b['top_services'])}): {kept}")
