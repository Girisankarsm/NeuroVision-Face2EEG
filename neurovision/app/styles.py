import streamlit as st

def apply_design_system():
    """Apply the custom dark scientific design system to Streamlit."""

    # CSS Customization for Streamlit
    css = """
    <style>
    :root {
        --nv-background: #0B1220;
        --nv-surface: #121B2E;
        --nv-border: #1E2A44;
        --nv-text: #E6EDF7;
        --nv-muted: #8FA3C2;
        --nv-accent: #4CC9F0;
    }

    /* Base background and text */
    .stApp {
        background-color: var(--nv-background);
        color: var(--nv-text);
        font-family: 'Inter', system-ui, sans-serif;
    }

    [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--nv-surface);
        border: 1px solid var(--nv-border);
        border-radius: 8px;
        padding: 0.65rem;
    }

    /* Fix header padding */
    .block-container {
        padding-top: 3.5rem !important;
    }

    /* Headers and Titles */
    h1, h2, h3, h4, h5, h6 {
        color: var(--nv-text) !important;
        font-family: 'Inter', system-ui, sans-serif;
    }

    /* Metrics and Data cards */
    div[data-testid="stMetricValue"] {
        color: var(--nv-accent);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--nv-muted);
    }

    /* Sidebar styling if used */
    section[data-testid="stSidebar"] {
        background-color: var(--nv-surface);
        border-right: 1px solid var(--nv-border);
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
        border-radius: 4px;
        border: 1px solid rgba(239, 71, 111, 0.3);
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-warning {
        background-color: rgba(244, 185, 66, 0.1);
        color: #F4B942;
        padding: 6px 10px;
        border-radius: 4px;
        border: 1px solid rgba(244, 185, 66, 0.3);
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-neutral {
        background-color: rgba(143, 163, 194, 0.1);
        color: #8FA3C2;
        padding: 6px 10px;
        border-radius: 4px;
        border: 1px solid rgba(143, 163, 194, 0.3);
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-success {
        background-color: rgba(46, 196, 182, 0.1);
        color: #2EC4B6;
        padding: 6px 10px;
        border-radius: 4px;
        border: 1px solid rgba(46, 196, 182, 0.3);
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* DataFrame/Table styling */
    .stDataFrame {
        background-color: var(--nv-surface);
        border: 1px solid var(--nv-border);
        border-radius: 8px;
    }

    /* Code blocks */
    pre {
        background-color: var(--nv-surface) !important;
        border: 1px solid var(--nv-border);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
