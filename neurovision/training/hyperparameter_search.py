from __future__ import annotations


def explain_optuna_status() -> str:
    return (
        "Optuna search is intentionally optional. Install optuna and call training.train_model "
        "with candidate configs produced by your study; only validation metrics should be used "
        "for model selection."
    )
