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

    st.subheader("Clustering")
    clustering_dir = results_dir / "clustering"
    clustering_json = clustering_dir / "cluster_results.json"
    if not clustering_json.exists():
        st.info("No clustering output yet. Run `python main.py cluster --dataset <dataset.npz> --out results/clustering`.")
    else:
        with clustering_json.open(encoding="utf-8") as stream:
            clustering = json.load(stream)
        if clustering.get("synthetic"):
            st.warning("SYNTHETIC SANITY CHECK · These outputs validate code paths only; they are not real-subject findings.")
        aggregate = clustering.get("aggregate", {})
        eeg_results = aggregate.get("cluster_eeg_statistics", {})
        alpha_results = eeg_results.get("alpha", {})
        perm_results = aggregate.get("alpha_permutation", {})
        metric_col, p_col, effect_col = st.columns(3)
        metric_col.metric("Held-out alpha KW p", f"{alpha_results.get('p_adjusted_holm', float('nan')):.4g}")
        p_col.metric("Alpha permutation p", f"{perm_results.get('p_value', float('nan')):.4g}")
        effect_col.metric("Alpha epsilon-squared", f"{alpha_results.get('epsilon_squared', float('nan')):.4f}")
        st.caption(aggregate.get("interpretation", "Facial-state clusters are descriptive and are not brain states."))
        st.caption("Window-level Kruskal-Wallis tests do not model repeated windows within subjects; compare with the subject-stratified permutation control.")
        table_path = clustering_dir / "cluster_table.csv"
        if table_path.exists():
            st.dataframe(pd.read_csv(table_path), use_container_width=True)
        with st.expander("Artifact sensitivity and subject checks"):
            st.json({
                "artifact_sensitivity": aggregate.get("artifact_sensitivity", {}),
                "per_subject_motion_alpha": aggregate.get("per_subject_motion_alpha", {}),
            })
        with st.expander("Regression feature comparison (subject bootstrap CIs)"):
            st.json(aggregate.get("regression_feature_comparison", {}))
        for fig in (
            "selection_curves.png", "heldout_tsne.png", "cluster_profiles.png",
            "alpha_by_cluster.png", "cluster_timeline.png",
        ):
            fig_path = clustering_dir / fig
            if fig_path.exists():
                st.image(str(fig_path), caption=fig.replace("_", " ").replace(".png", "").title())

    # Load existing supervised experiment figures if available
    st.subheader("Visualizations")
    figs = ["prediction_vs_ground_truth.png", "per_subject_metrics.png", "band_powers.png", "residuals.png"]

    for fig in figs:
        fig_path = results_dir / fig
        if fig_path.exists():
            st.image(str(fig_path), caption=fig.replace("_", " ").title().replace(".Png", ""))
