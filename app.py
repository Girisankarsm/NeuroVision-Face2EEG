import streamlit as st

from neurovision.app.about_tab import render_about_tab
from neurovision.app.live_tab import render_live_tab
from neurovision.app.results_tab import render_results_tab
from neurovision.app.styles import apply_design_system


def main():
    st.set_page_config(
        page_title="NeuroVision (Face2EEG)",
        page_icon="N",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_design_system()

    st.title("NeuroVision Research Dashboard")

    tab1, tab2, tab3 = st.tabs(["Live Feed", "Results & Experiments", "About & Limits"])

    with tab1:
        render_live_tab()

    with tab2:
        render_results_tab()

    with tab3:
        render_about_tab()


if __name__ == "__main__":
    main()
