import os
from pathlib import Path
import json

import pandas as pd
import streamlit as st


def render_results_tab():
    st.header("📊 Experiment Results")

    results_dir = Path("results")
    if not results_dir.exists():
        st.warning("No results found. Run `python main.py experiment --dataset data.npz` to generate results.")
        return

    # Load baselines
    if (results_dir / "baselines.csv").exists():
        st.subheader("Baseline Comparison")
        df_base = pd.read_csv(results_dir / "baselines.csv")
        summary = df_base.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("r2", ascending=False)
        st.dataframe(summary, use_container_width=True)

    # Load controls
    if (results_dir / "controls.json").exists():
        st.subheader("Controls & Rigor")
        with (results_dir / "controls.json").open() as f:
            controls = json.load(f)

        col1, col2 = st.columns(2)
        with col1:
            p_val = controls.get("permutation_test", {}).get("p_value", "N/A")
            st.metric("Permutation Test p-value", p_val)
            st.caption("H0: No relationship between face and EEG. p > 0.05 means we failed to reject random chance.")

        with col2:
            drop = controls.get("blink_ablation", {}).get("drop", "N/A")
            if isinstance(drop, float):
                st.metric("Blink Ablation (R² Drop)", f"{drop:.4f}")
            else:
                st.metric("Blink Ablation (R² Drop)", str(drop))
            st.caption("How much performance drops when blink features are zeroed out.")

    # Load figures if available
    st.subheader("Visualizations")
    figs = ["prediction_vs_ground_truth.png", "per_subject_metrics.png", "band_powers.png", "residuals.png"]

    for fig in figs:
        fig_path = results_dir / fig
        if fig_path.exists():
            st.image(str(fig_path), caption=fig.replace("_", " ").title().replace(".Png", ""))
