"""
Empirical Scaling Laws estimation (Kaplan / Chinchilla / Hoffmann et al.)
adapted for Small Language Models on Low-Resource / Small Data regime.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def power_law_kaplan(X: Tuple[np.ndarray, np.ndarray], E: float, A: float, B: float, alpha: float, beta: float) -> np.ndarray:
    """
    Kaplan et al. / Chinchilla parametric loss function:
    L(N, D) = E + A * N^(-alpha) + B * D^(-beta)
    """
    N, D = X
    return E + (A / (N ** alpha)) + (B / (D ** beta))


def fit_scaling_law(df: pd.DataFrame) -> Dict[str, any]:
    """
    Fits empirical scaling law parameters (E, A, B, alpha, beta) from experiment dataframe.
    """
    df_clean = df.copy().dropna(subset=["val_bpc"])

    N = df_clean["non_embedding_params"].values.astype(float)
    D = df_clean["data_words"].values.astype(float) if "data_words" in df_clean.columns else df_clean["data_fraction"].values.astype(float)
    L = df_clean["val_bpc"].values.astype(float)

    # Initial parameter guesses
    min_loss = float(np.min(L))
    p0 = [
        min_loss * 0.7,   # E (irreducible entropy)
        100.0,            # A
        100.0,            # B
        0.15,             # alpha (param exponent, typically 0.05-0.35)
        0.25,             # beta (data exponent, typically 0.1-0.4)
    ]
    bounds = (
        [0.0, 1e-4, 1e-4, 0.01, 0.01],
        [min_loss * 0.99, 1e6, 1e6, 1.5, 1.5],
    )

    try:
        popt, pcov = curve_fit(
            power_law_kaplan,
            (N, D),
            L,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
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
        # Fallback to simple 1D power law fits per variable
        return {
            "success": False,
            "error": str(e),
            "fallback": "Non-linear joint fit did not converge with current data points.",
        }


def fit_tokenizer_scaling_curves(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Fits separate 1D power law curves for each tokenizer:
    BPC = a * D^(-b)
    """
    results = {}
    for tok, group in df.groupby("tokenizer"):
        d_vals = group["data_words"].values.astype(float) if "data_words" in group.columns else group["data_fraction"].values.astype(float)
        bpc_vals = group["val_bpc"].values.astype(float)

        if len(d_vals) < 3:
            continue

        try:
            log_d = np.log(d_vals)
            log_bpc = np.log(bpc_vals)
            # Linear regression on log-log: log(BPC) = log(a) - b * log(D)
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
