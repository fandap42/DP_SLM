"""
Statistical analysis module for SLM experiments:
- Linear Mixed-Effects Models (LMEM)
- Factorial ANOVA and interaction analysis
- Empirical scaling laws (Kaplan / Chinchilla)
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit


def fit_mixed_effects_model(df: pd.DataFrame) -> Dict[str, Any]:
    df_clean = df.copy()
    df_clean["log_D"] = np.log(df_clean["data_words"] if "data_words" in df_clean.columns else df_clean["data_fraction"])
    df_clean["log_N"] = np.log(df_clean["non_embedding_params"] if "non_embedding_params" in df_clean.columns else df_clean["total_params"])

    formula = "val_bpc ~ log_D + log_N + C(tokenizer) + log_D:C(tokenizer)"
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
            model_summary_str = f"MixedLM fallback ({e})"

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
    df_clean = df.copy()
    formula = "val_bpc ~ C(data_fraction) + C(model_size) + C(tokenizer) + C(data_fraction):C(tokenizer) + C(model_size):C(tokenizer)"
    ols_model = smf.ols(formula, data=df_clean).fit()
    anova_table = sm.stats.anova_lm(ols_model, typ=2)

    total_ss = anova_table["sum_sq"].sum()
    anova_table["eta_sq"] = anova_table["sum_sq"] / total_ss
    anova_table["eta_sq_percent"] = anova_table["eta_sq"] * 100.0
    return anova_table


def power_law_kaplan(X: Tuple[np.ndarray, np.ndarray], E: float, A: float, B: float, alpha: float, beta: float) -> np.ndarray:
    N, D = X
    return E + (A / (N ** alpha)) + (B / (D ** beta))


def fit_scaling_law(df: pd.DataFrame) -> Dict[str, any]:
    df_clean = df.copy().dropna(subset=["val_bpc"])
    N = df_clean["non_embedding_params"].values.astype(float)
    D = df_clean["data_words"].values.astype(float) if "data_words" in df_clean.columns else df_clean["data_fraction"].values.astype(float)
    L = df_clean["val_bpc"].values.astype(float)

    min_loss = float(np.min(L))
    p0 = [min_loss * 0.7, 100.0, 100.0, 0.15, 0.25]
    bounds = ([0.0, 1e-4, 1e-4, 0.01, 0.01], [min_loss * 0.99, 1e6, 1e6, 1.5, 1.5])

    try:
        popt, pcov = curve_fit(power_law_kaplan, (N, D), L, p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        E, A, B, alpha, beta = popt
        pred_L = power_law_kaplan((N, D), *popt)
        ss_res = np.sum((L - pred_L) ** 2)
        ss_tot = np.sum((L - np.mean(L)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

        return {
            "success": True,
            "params": {
                "E_irreducible_loss": float(E),
                "A_param_scale": float(A),
                "B_data_scale": float(B),
                "alpha_param_exponent": float(alpha),
                "beta_data_exponent": float(beta),
            },
            "std_errors": {
                "E_err": float(perr[0]),
                "A_err": float(perr[1]),
                "B_err": float(perr[2]),
                "alpha_err": float(perr[3]),
                "beta_err": float(perr[4]),
            },
            "r_squared": float(r2),
            "equation": f"BPC = {E:.3f} + ({A:.2f} / N^{alpha:.3f}) + ({B:.2f} / D^{beta:.3f})",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def fit_tokenizer_scaling_curves(df: pd.DataFrame) -> Dict[str, Dict]:
    results = {}
    for tok, group in df.groupby("tokenizer"):
        d_vals = group["data_words"].values.astype(float) if "data_words" in group.columns else group["data_fraction"].values.astype(float)
        bpc_vals = group["val_bpc"].values.astype(float)
        if len(d_vals) < 3:
            continue
        try:
            log_d = np.log(d_vals)
            log_bpc = np.log(bpc_vals)
            slope, intercept = np.polyfit(log_d, log_bpc, deg=1)
            b = -slope
            a = np.exp(intercept)
            r2 = float(np.corrcoef(log_d, log_bpc)[0, 1] ** 2)
            results[str(tok)] = {
                "a": float(a),
                "scaling_exponent_beta": float(b),
                "r_squared": r2,
                "formula": f"BPC = {a:.3f} * D^(-{b:.3f})",
            }
        except Exception as e:
            results[str(tok)] = {"error": str(e)}
    return results
