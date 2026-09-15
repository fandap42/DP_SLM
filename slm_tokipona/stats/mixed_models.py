"""
Statistical modeling module:
- Linear Mixed-Effects Models (LMEM)
- Factorial ANOVA and interaction analysis
- Variance decomposition (eta-squared)
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_mixed_effects_model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits Linear Mixed-Effects Model:
      BPC ~ log_D + log_N + C(tokenizer, Treatment(reference='word'))
            + log_D:C(tokenizer) + (1 | seed)

    Quantifies the main effects of data scale, parameter scale, tokenization type,
    and their statistical interactions, while accounting for random seed variance.
    """
    df_clean = df.copy()
    df_clean["log_D"] = np.log(df_clean["data_words"] if "data_words" in df_clean.columns else df_clean["data_fraction"])
    df_clean["log_N"] = np.log(df_clean["non_embedding_params"] if "non_embedding_params" in df_clean.columns else df_clean["total_params"])

    # Formula with interactions and reference level 'word'
    formula = "val_bpc ~ log_D + log_N + C(tokenizer) + log_D:C(tokenizer)"

    # Fallback to OLS if only 1 seed or if mixedlm fails convergence
    use_mixed = len(df_clean["seed"].unique()) > 1

    model_summary_str = ""
    coef_dict = {}
    pvalues_dict = {}
    conf_int_dict = {}

    if use_mixed:
        try:
            model = smf.mixedlm(formula, df_clean, groups=df_clean["seed"])
            result = model.fit(reml=False)
            model_summary_str = str(result.summary())
            for k, v in result.params.items():
                coef_dict[k] = float(v)
            for k, v in result.pvalues.items():
                pvalues_dict[k] = float(v)
            ci = result.conf_int()
            for idx, row in ci.iterrows():
                conf_int_dict[idx] = (float(row[0]), float(row[1]))
        except Exception as e:
            use_mixed = False
            model_summary_str = f"MixedLM failed ({e}), falling back to OLS."

    if not use_mixed:
        ols_model = smf.ols(formula, data=df_clean)
        result = ols_model.fit()
        model_summary_str = str(result.summary())
        for k, v in result.params.items():
            coef_dict[k] = float(v)
        for k, v in result.pvalues.items():
            pvalues_dict[k] = float(v)
        ci = result.conf_int()
        for idx, row in ci.iterrows():
            conf_int_dict[idx] = (float(row[0]), float(row[1]))

    return {
        "model_type": "MixedLM" if use_mixed else "OLS",
        "formula": formula,
        "summary_text": model_summary_str,
        "coefficients": coef_dict,
        "p_values": pvalues_dict,
        "confidence_intervals": conf_int_dict,
    }


def compute_factorial_anova(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Factorial ANOVA table with Sum of Squares and Effect Sizes (eta-squared)
    for Data Fraction, Model Size, Tokenizer, and their 2-way interactions.
    """
    df_clean = df.copy()
    formula = "val_bpc ~ C(data_fraction) + C(model_size) + C(tokenizer) + C(data_fraction):C(tokenizer) + C(model_size):C(tokenizer)"

    ols_model = smf.ols(formula, data=df_clean).fit()
    anova_table = sm.stats.anova_lm(ols_model, typ=2)

    total_ss = anova_table["sum_sq"].sum()
    anova_table["eta_sq"] = anova_table["sum_sq"] / total_ss
    anova_table["eta_sq_percent"] = anova_table["eta_sq"] * 100.0

    return anova_table
