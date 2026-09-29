import streamlit as st

from ui import ensure_bundle, inject_css, render_logo, render_settings_panel, render_topbar
from views import (ablation, data_quality, dashboard, error_analysis, feature_engineering,
                    models, pipeline, prediction)

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
        st.Page(dashboard.render, title="Dashboard", icon=":material/space_dashboard:",
                url_path="dashboard", default=True),
        st.Page(pipeline.render, title="Pipeline", icon=":material/account_tree:", url_path="pipeline"),
        st.Page(data_quality.render, title="Data Quality", icon=":material/fact_check:", url_path="data-quality"),
        st.Page(feature_engineering.render, title="Feature Engineering", icon=":material/tune:",
                url_path="feature-engineering"),
        st.Page(ablation.render, title="Ablation Study", icon=":material/science:", url_path="ablation"),
        st.Page(models.render, title="Models", icon=":material/model_training:", url_path="models"),
        st.Page(error_analysis.render, title="Error Analysis", icon=":material/troubleshoot:",
                url_path="error-analysis"),
        st.Page(prediction.render, title="Prediction", icon=":material/sensors:", url_path="prediction"),
    ]
}
pg = st.navigation(pages)

render_settings_panel()
ensure_bundle()
render_topbar()
pg.run()
