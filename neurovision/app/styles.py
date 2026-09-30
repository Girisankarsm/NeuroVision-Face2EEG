import streamlit as st

def apply_design_system():
    """Apply the custom dark scientific design system to Streamlit."""

    # CSS Customization for Streamlit
    css = """
    <style>
    /* Base background and text */
    .stApp {
        background-color: #0B1220;
        color: #E6EDF7;
        font-family: 'Inter', system-ui, sans-serif;
    }

    [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {
        background: #121B2E;
        border: 1px solid #1E2A44;
        border-radius: 12px;
        padding: 0.65rem;
    }

    /* Fix header padding */
    .block-container {
        padding-top: 3.5rem !important;
    }

    /* Headers and Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #E6EDF7 !important;
        font-family: 'Inter', system-ui, sans-serif;
    }

    /* Metrics and Data cards */
    div[data-testid="stMetricValue"] {
        color: #4CC9F0;
    }
    div[data-testid="stMetricLabel"] {
        color: #8FA3C2;
    }

    /* Sidebar styling if used */
    section[data-testid="stSidebar"] {
        background-color: #121B2E;
        border-right: 1px solid #1E2A44;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: transparent;
        color: #8FA3C2;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4CC9F0 !important;
        box-shadow: inset 0 -2px #4CC9F0;
    }

    /* Status Badges */
    .status-error {
        background-color: rgba(239, 71, 111, 0.1);
        color: #EF476F;
        padding: 6px 10px;
        border-radius: 6px;
        border: 1px solid rgba(239, 71, 111, 0.3);
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-warning {
        background-color: rgba(244, 185, 66, 0.1);
        color: #F4B942;
        padding: 6px 10px;
        border-radius: 6px;
        border: 1px solid rgba(244, 185, 66, 0.3);
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-neutral {
        background-color: rgba(143, 163, 194, 0.1);
        color: #8FA3C2;
        padding: 6px 10px;
        border-radius: 6px;
        border: 1px solid rgba(143, 163, 194, 0.3);
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-success {
        background-color: rgba(46, 196, 182, 0.1);
        color: #2EC4B6;
        padding: 6px 10px;
        border-radius: 6px;
        border: 1px solid rgba(46, 196, 182, 0.3);
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* DataFrame/Table styling */
    .stDataFrame {
        background-color: #121B2E;
        border: 1px solid #1E2A44;
        border-radius: 12px;
    }

    /* Code blocks */
    pre {
        background-color: #121B2E !important;
        border: 1px solid #1E2A44;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
