from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def generate_report(results_dir: str | Path) -> str:
    """Auto-generate a markdown Results section from the experiment output."""
    results_dir = Path(results_dir)

    if not (results_dir / "baselines.csv").exists():
        return "No results found. Run `python main.py experiment` first."

    df_base = pd.read_csv(results_dir / "baselines.csv")
    base_summary = df_base.groupby("model")[["mae", "rmse", "r2"]].mean().sort_values("r2", ascending=False)

    cv_summary = ""
    if (results_dir / "cv_results.csv").exists():
        df_cv = pd.read_csv(results_dir / "cv_results.csv")
        cv_summary = df_cv.groupby("model")[["mae", "rmse", "r2"]].mean().to_markdown()

    controls_summary = ""
    if (results_dir / "controls.json").exists():
        with (results_dir / "controls.json").open() as f:
            controls = json.load(f)
        p_val = controls.get("permutation_test", {}).get("p_value", "N/A")
        ablation_drop = controls.get("blink_ablation", {}).get("drop", "N/A")
        controls_summary = f"- **Permutation Test p-value**: {p_val}\n- **Blink Ablation R² Drop**: {ablation_drop}"

    report = f"""## Results (Auto-Generated)

### Baseline Models
{base_summary.to_markdown()}

### Deep Models (Cross-Validation)
{cv_summary if cv_summary else "N/A"}

### Controls & Rigor
{controls_summary if controls_summary else "N/A"}

### Scientific Interpretation
- **Are we predicting real EEG?** A p-value > 0.05 indicates the model is NOT learning a meaningful signal better than random chance.
- **Is it just blinks?** If the blink ablation drop is large (e.g. > 0.05), the model is heavily relying on ocular artifacts rather than neural activity.
"""
    return report

def write_report_to_readme(results_dir: str | Path, readme_path: str | Path = "README.md") -> None:
    report = generate_report(results_dir)
    readme = Path(readme_path).read_text()

    start_marker = "<!-- RESULTS_START -->"
    end_marker = "<!-- RESULTS_END -->"

    if start_marker in readme and end_marker in readme:
        before = readme.split(start_marker)[0]
        after = readme.split(end_marker)[1]
        new_readme = f"{before}{start_marker}\n{report}\n{end_marker}{after}"
    else:
        new_readme = readme + f"\n\n{start_marker}\n{report}\n{end_marker}\n"

    Path(readme_path).write_text(new_readme)
