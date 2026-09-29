import streamlit as st

from ui import ensure_bundle, inject_css, render_logo, render_settings_panel, render_topbar
from views import pipeline, predict, upload_quality

st.set_page_config(
    page_title="NetGuard | NSL-KDD Intrusion Detection",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_logo()

pages = {
    "Pages": [
        st.Page(upload_quality.render, title="Upload & Data Quality", icon=":material/fact_check:",
                url_path="data-quality", default=True),
        st.Page(pipeline.render, title="Pipeline Explorer", icon=":material/account_tree:", url_path="pipeline"),
        st.Page(predict.render, title="Predict & Visualize", icon=":material/sensors:", url_path="predict"),
    ]
}
pg = st.navigation(pages)

render_settings_panel()
ensure_bundle()
render_topbar()
pg.run()
