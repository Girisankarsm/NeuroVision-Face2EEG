import streamlit as st

def render_about_tab():
    st.header("ℹ️ About NeuroVision")

    st.markdown("""
    **NeuroVision (Face2EEG)** is a research instrument designed to investigate whether temporal facial dynamics captured from a standard webcam can predict correlated electroencephalogram (EEG) neural oscillations.

    ### How it Works
    1. **Face Tracking**: Extracts 468 3D landmarks and action-unit intensities using MediaPipe.
    2. **Feature Extraction**: Computes a compact ~60-dimensional vector (EAR, MAR, head pose, blink rate, AU intensities) per frame.
    3. **Prediction**: A machine learning model processes a sliding window of these features to predict log band power (Delta, Theta, Alpha, Beta, Gamma) of the corresponding EEG segment.
    """)

    st.warning("""
    ### ⚠️ Scientific & Ethical Disclaimer

    **This is a research instrument, NOT a medical device.**

    - Camera-only outputs represent AI-predicted statistical estimates based on facial muscle movements and ocular artifacts.
    - They are **NOT** direct recordings of neural activity.
    - Model confidence metrics reflect mathematical uncertainty, **not biological certainty**.
    - If no trained model is loaded, the application explicitly displays `EEG MODEL NOT LOADED` and will never fabricate fake neural activity.
    """)
