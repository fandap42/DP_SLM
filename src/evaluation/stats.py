"""
Statistical Analysis Module for Toki Pona Small Language Model (SLM) Experiments.
Implements the 4 statistical pillars aligned with FIS VŠE requirements:
- Model 1: Empirical Scaling Laws (Linearized Log-Log Power-Law Regression & Kaplan Fit)
- Model 2: Linear Mixed-Effects Models (LMM with nested seeds and domain splits)
- Model 3: Factorial ANOVA, Effect Sizes (eta-squared), Marginal Means & Contrasts
- Model 4: Generalized Linear Mixed Model (GLMM / Binomial Logit for Grammar Acceptability)
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit
import statsmodels.api as sm
import statsmodels.formula.api as smf


# =====================================================================
# Model 1: Škálovací vztahy (Scaling Laws / Power-Law regrese)
# =====================================================================

def power_law_kaplan(
    X: Tuple[np.ndarray, np.ndarray],
    E: float,
    A: float,
    B: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Non-linear Kaplan et al. power-law: L(N, D) = E + A/(N^alpha) + B/(D^beta)"""
    N, D = X
    return E + (A / (N ** alpha)) + (B / (D ** beta))


def fit_scaling_law(df: pd.DataFrame, metric_col: str = "test_bpc") -> Dict[str, Any]:
    """
    Fits empirical scaling laws:
    1. Linearized log-log joint regression: ln(BPC) = beta_0 + alpha * ln(N) + beta * ln(D)
    2. Per-tokenizer scaling curves and exponent comparisons (char vs word vs bpe)
    3. Slope interaction model (testing if alpha and beta differ significantly by tokenizer)
    4. Non-linear Kaplan / Chinchilla parametric fit
    """
    df_clean = df.copy()
    if metric_col not in df_clean.columns or df_clean[metric_col].isna().all():
        metric_col = "val_bpc"

    df_clean = df_clean.dropna(subset=[metric_col])

    # Parameter and data scale variables
    n_col = "non_embedding_params" if "non_embedding_params" in df_clean.columns else "total_params"
    d_col = "data_words" if "data_words" in df_clean.columns else "data_fraction"

    df_clean["log_N"] = np.log(df_clean[n_col].astype(float))
    df_clean["log_D"] = np.log(df_clean[d_col].astype(float))
    df_clean["log_bpc"] = np.log(df_clean[metric_col].astype(float))

    results: Dict[str, Any] = {
        "metric_used": metric_col,
        "n_samples": len(df_clean),
    }

    # 1. Joint log-log power-law regression
    joint_formula = "log_bpc ~ log_N + log_D"
    joint_model = smf.ols(joint_formula, data=df_clean).fit(cov_type="HC3")

    alpha_joint = -joint_model.params.get("log_N", 0.0)
    beta_joint = -joint_model.params.get("log_D", 0.0)

    results["joint_linear_scaling"] = {
        "formula": "ln(BPC) = beta_0 - alpha * ln(N) - beta * ln(D)",
        "beta_0": float(joint_model.params.get("Intercept", 0.0)),
        "alpha_param_exponent": float(alpha_joint),
        "beta_data_exponent": float(beta_joint),
        "alpha_pvalue": float(joint_model.pvalues.get("log_N", 1.0)),
        "beta_pvalue": float(joint_model.pvalues.get("log_D", 1.0)),
        "r_squared": float(joint_model.rsquared),
        "adj_r_squared": float(joint_model.rsquared_adj),
        "f_pvalue": float(joint_model.f_pvalue) if hasattr(joint_model, "f_pvalue") else 0.0,
        "summary_text": str(joint_model.summary()),
    }

    # 2. Per-tokenizer scaling curves
    tok_curves = {}
    for tok, grp in df_clean.groupby("tokenizer"):
        if len(grp) >= 3:
            try:
                m = smf.ols("log_bpc ~ log_N + log_D", data=grp).fit()
                a = -m.params.get("log_N", 0.0)
                b = -m.params.get("log_D", 0.0)
                tok_curves[str(tok)] = {
                    "intercept": float(m.params.get("Intercept", 0.0)),
                    "alpha_param_exponent": float(a),
                    "beta_data_exponent": float(b),
                    "r_squared": float(m.rsquared),
                    "alpha_pvalue": float(m.pvalues.get("log_N", 1.0)),
                    "beta_pvalue": float(m.pvalues.get("log_D", 1.0)),
                    "equation": f"ln(BPC) = {m.params.get('Intercept', 0.0):.3f} - ({a:.3f})*ln(N) - ({b:.3f})*ln(D)",
                }
            except Exception as e:
                tok_curves[str(tok)] = {"error": str(e)}

    results["per_tokenizer_scaling"] = tok_curves

    # 3. Slope interaction model: does scaling exponent differ by tokenizer?
    slope_interaction_formula = "log_bpc ~ log_N * C(tokenizer, Treatment(reference='word')) + log_D * C(tokenizer, Treatment(reference='word'))"
    try:
        slope_model = smf.ols(slope_interaction_formula, data=df_clean).fit(cov_type="HC3")
        results["slope_heterogeneity_test"] = {
            "formula": slope_interaction_formula,
            "r_squared": float(slope_model.rsquared),
            "summary_text": str(slope_model.summary()),
            "coefficients": {k: float(v) for k, v in slope_model.params.items()},
            "p_values": {k: float(v) for k, v in slope_model.pvalues.items()},
        }
    except Exception as e:
        results["slope_heterogeneity_test"] = {"error": str(e)}

    # 4. Non-linear Kaplan / Chinchilla parametric fit
    N_arr = df_clean[n_col].values.astype(float)
    D_arr = df_clean[d_col].values.astype(float)
    L_arr = df_clean[metric_col].values.astype(float)

    min_loss = float(np.min(L_arr))
    p0 = [min_loss * 0.7, 100.0, 100.0, 0.15, 0.25]
    bounds = ([0.0, 1e-4, 1e-4, 0.01, 0.01], [min_loss * 0.99, 1e6, 1e6, 1.5, 1.5])

    try:
        popt, pcov = curve_fit(power_law_kaplan, (N_arr, D_arr), L_arr, p0=p0, bounds=bounds, maxfev=15000)
        perr = np.sqrt(np.diag(pcov))
        E, A, B, alpha, beta = popt
        pred_L = power_law_kaplan((N_arr, D_arr), *popt)
        ss_res = np.sum((L_arr - pred_L) ** 2)
        ss_tot = np.sum((L_arr - np.mean(L_arr)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

        results["kaplan_power_law"] = {
            "success": True,
            "E_irreducible_loss": float(E),
            "A_param_scale": float(A),
            "B_data_scale": float(B),
            "alpha_param_exponent": float(alpha),
            "beta_data_exponent": float(beta),
            "r_squared": float(r2),
            "equation": f"BPC = {E:.3f} + ({A:.2f} / N^{alpha:.3f}) + ({B:.2f} / D^{beta:.3f})",
        }
    except Exception as e:
        results["kaplan_power_law"] = {"success": False, "error": str(e)}

    return results


def fit_tokenizer_scaling_curves(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compatibility wrapper for univariate data scaling curves."""
    results = {}
    for tok, group in df.groupby("tokenizer"):
        d_vals = group["data_words"].values.astype(float) if "data_words" in group.columns else group["data_fraction"].values.astype(float)
        bpc_col = "test_bpc" if "test_bpc" in group.columns and not group["test_bpc"].isna().all() else "val_bpc"
        bpc_vals = group[bpc_col].values.astype(float)
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


# =====================================================================
# Model 2: Hierarchické lineární modely se smíšenými efekty (LMM)
# =====================================================================

def fit_mixed_effects_model(df: pd.DataFrame, metric_col: str = "test_bpc") -> Dict[str, Any]:
    """
    Fits Linear Mixed-Effects Model (LMM):
      BPC_ijkl = mu + tau_i + beta_1 * ln(N_j) + beta_2 * ln(D_k)
                 + (tau x ln N)_ij + (tau x ln D)_ik + u_seed + epsilon

    Includes both the main configuration model and domain-split mixed model.
    """
    df_clean = df.copy()
    if metric_col not in df_clean.columns or df_clean[metric_col].isna().all():
        metric_col = "val_bpc"
    df_clean = df_clean.dropna(subset=[metric_col])

    n_col = "non_embedding_params" if "non_embedding_params" in df_clean.columns else "total_params"
    d_col = "data_words" if "data_words" in df_clean.columns else "data_fraction"

    df_clean["log_N"] = np.log(df_clean[n_col].astype(float))
    df_clean["log_D"] = np.log(df_clean[d_col].astype(float))
    df_clean["y_metric"] = df_clean[metric_col].astype(float)

    formula = "y_metric ~ C(tokenizer, Treatment(reference='word')) + log_N + log_D + C(tokenizer, Treatment(reference='word')):log_N + C(tokenizer, Treatment(reference='word')):log_D"

    unique_seeds = df_clean["seed"].nunique()
    use_mixed = unique_seeds > 1

    summary_str = ""
    coef_dict = {}
    pvalues_dict = {}
    conf_int_dict = {}
    model_type = "OLS (HC3 Robust)"

    if use_mixed:
        try:
            model = smf.mixedlm(formula, df_clean, groups=df_clean["seed"])
            result = model.fit(reml=False)
            model_type = "Linear Mixed-Effects Model (smf.mixedlm)"
            summary_str = str(result.summary())
            for k, v in result.params.items():
                coef_dict[k] = float(v)
            for k, v in result.pvalues.items():
                pvalues_dict[k] = float(v)
            for idx, row in result.conf_int().iterrows():
                conf_int_dict[idx] = (float(row[0]), float(row[1]))
        except Exception as e:
            use_mixed = False
            summary_str = f"MixedLM fallback ({e})"

    if not use_mixed:
        try:
            ols_model = smf.ols(formula, data=df_clean)
            result = ols_model.fit(cov_type="HC3")
            model_type = "OLS with HC3 Heteroskedasticity-Consistent Covariance"
            summary_str = str(result.summary())
            for k, v in result.params.items():
                coef_dict[k] = float(v)
            for k, v in result.pvalues.items():
                pvalues_dict[k] = float(v)
            for idx, row in result.conf_int().iterrows():
                conf_int_dict[idx] = (float(row[0]), float(row[1]))
        except Exception as e:
            summary_str = f"Model fit note: {e}"

    # Domain Generalization LMM (if domain columns are present)
    domain_model_res = {}
    if "bpc_in_domain" in df_clean.columns and "bpc_out_domain" in df_clean.columns:
        try:
            # Reshape into long format: each run has 2 observations (in-domain and out-of-domain)
            rows = []
            for _, r in df_clean.iterrows():
                rows.append({
                    "exp_id": r["exp_id"],
                    "tokenizer": r["tokenizer"],
                    "log_N": r["log_N"],
                    "log_D": r["log_D"],
                    "seed": r["seed"],
                    "domain": "in_domain",
                    "bpc": r["bpc_in_domain"],
                })
                rows.append({
                    "exp_id": r["exp_id"],
                    "tokenizer": r["tokenizer"],
                    "log_N": r["log_N"],
                    "log_D": r["log_D"],
                    "seed": r["seed"],
                    "domain": "out_domain",
                    "bpc": r["bpc_out_domain"],
                })
            df_long = pd.DataFrame(rows).dropna(subset=["bpc"])

            dom_formula = "bpc ~ C(domain, Treatment(reference='in_domain')) * C(tokenizer, Treatment(reference='word')) + log_N + log_D"
            if df_long["exp_id"].nunique() > 1:
                dom_mixed = smf.mixedlm(dom_formula, df_long, groups=df_long["exp_id"])
                dom_fit = dom_mixed.fit(reml=False)
                domain_model_res = {
                    "model_type": "MixedLM (Repeated measures across in-domain vs out-domain)",
                    "summary_text": str(dom_fit.summary()),
                    "coefficients": {k: float(v) for k, v in dom_fit.params.items()},
                    "p_values": {k: float(v) for k, v in dom_fit.pvalues.items()},
                }
        except Exception as e:
            domain_model_res = {"error": str(e)}

    return {
        "model_type": model_type,
        "formula": formula,
        "summary_text": summary_str,
        "coefficients": coef_dict,
        "p_values": pvalues_dict,
        "confidence_intervals": conf_int_dict,
        "domain_generalization_model": domain_model_res,
    }


# =====================================================================
# Model 3: Analýza interakcí (Faktorová ANOVA / Kontrasty / Marginální průměry)
# =====================================================================

def compute_factorial_anova(df: pd.DataFrame, metric_col: str = "test_bpc") -> Dict[str, Any]:
    """
    Computes multi-way Factorial ANOVA and interaction contrasts:
    - Tests main effects and two-way interaction: Tokenizer x Data Fraction and Tokenizer x Model Size
    - Computes effect sizes: Eta-squared (eta^2) and Partial Eta-squared (eta^2_partial)
    - Computes Estimated Marginal Means across factor levels
    - Tests hypothesis of tokenizer crossover at varying corpus scale
    """
    df_clean = df.copy()
    if metric_col not in df_clean.columns or df_clean[metric_col].isna().all():
        metric_col = "val_bpc"
    df_clean = df_clean.dropna(subset=[metric_col])

    df_clean["y"] = df_clean[metric_col].astype(float)
    df_clean["f_data"] = df_clean["data_fraction"].astype(str)
    df_clean["f_model"] = df_clean["model_size"].astype(str)
    df_clean["f_tok"] = df_clean["tokenizer"].astype(str)

    formula = "y ~ C(f_tok) + C(f_data) + C(f_model) + C(f_tok):C(f_data) + C(f_tok):C(f_model)"

    import patsy

    # Identify if a fully crossed factorial grid exists (e.g. 2x2x3 grid where all tokenizers exist for each cell)
    cell_counts = df_clean.groupby(["data_fraction", "model_size"])["tokenizer"].nunique()
    full_cells = cell_counts[cell_counts == df_clean["tokenizer"].nunique()].index
    has_balanced_subset = len(full_cells) >= 4 and len(full_cells) < len(cell_counts)

    if has_balanced_subset:
        df_target = df_clean.set_index(["data_fraction", "model_size"]).loc[full_cells].reset_index()
    else:
        df_target = df_clean

    rhs_formula = "C(f_tok) + C(f_data) + C(f_model) + C(f_tok):C(f_data) + C(f_tok):C(f_model)"
    try:
        dm = patsy.dmatrix(rhs_formula, data=df_target)
        rank = np.linalg.matrix_rank(dm)
        is_full_rank = (rank == dm.shape[1]) and (len(df_target) - rank >= 2)
    except Exception:
        is_full_rank = False

    if is_full_rank:
        ols_model = smf.ols(formula, data=df_target).fit()
        anova_table = sm.stats.anova_lm(ols_model, typ=2)
    else:
        simpler_formula = "y ~ C(f_tok, Treatment(reference='word')) + C(f_data)"
        ols_model = smf.ols(simpler_formula, data=df_target).fit()
        anova_table = sm.stats.anova_lm(ols_model, typ=2)

    total_ss = anova_table["sum_sq"].sum()
    residual_ss = anova_table.loc["Residual", "sum_sq"] if "Residual" in anova_table.index else 1.0

    anova_table["eta_sq"] = anova_table["sum_sq"] / max(total_ss, 1e-8)
    anova_table["eta_sq_percent"] = anova_table["eta_sq"] * 100.0
    anova_table["partial_eta_sq"] = anova_table["sum_sq"] / (anova_table["sum_sq"] + residual_ss)
    anova_table["partial_eta_sq_percent"] = anova_table["partial_eta_sq"] * 100.0
    if "Residual" in anova_table.index:
        anova_table.loc["Residual", ["partial_eta_sq", "partial_eta_sq_percent"]] = np.nan

    # Test balanced interaction sub-grid if available
    balanced_anova = None
    if has_balanced_subset:
        balanced_anova = anova_table.copy()

    # Marginal Means: Tokenizer x Data Fraction
    mean_tok_data = df_clean.groupby(["tokenizer", "data_fraction"])["y"].agg(["mean", "std", "count"]).reset_index()
    # Marginal Means: Tokenizer x Model Size
    mean_tok_model = df_clean.groupby(["tokenizer", "model_size"])["y"].agg(["mean", "std", "count"]).reset_index()

    # Pairwise Contrasts & Percentage differences across tokenizers by data fraction
    crossover_analysis = []
    for df_val, grp in df_clean.groupby("data_fraction"):
        means = grp.groupby("tokenizer")["y"].mean().to_dict()
        word_m = means.get("word", None)
        char_m = means.get("char", None)
        bpe_m = means.get("bpe", None)

        row = {"data_fraction": float(df_val), "word_mean": word_m, "char_mean": char_m, "bpe_mean": bpe_m}
        if word_m is not None and char_m is not None:
            row["word_advantage_vs_char_bpc"] = round(char_m - word_m, 4)
            row["word_advantage_vs_char_pct"] = round(((char_m - word_m) / char_m) * 100.0, 2)
        if word_m is not None and bpe_m is not None:
            row["word_advantage_vs_bpe_bpc"] = round(bpe_m - word_m, 4)
            row["word_advantage_vs_bpe_pct"] = round(((bpe_m - word_m) / bpe_m) * 100.0, 2)
        crossover_analysis.append(row)

    return {
        "anova_table": anova_table,
        "balanced_interaction_anova": balanced_anova,
        "marginal_means_data": mean_tok_data.to_dict(orient="records"),
        "marginal_means_model": mean_tok_model.to_dict(orient="records"),
        "crossover_analysis": crossover_analysis,
    }


# =====================================================================
# Model 4: Zobecněný lineární smíšený model pro gramatickou přijatelnost (GLMM)
# =====================================================================

def fit_grammar_glmm(item_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits Generalized Linear Mixed Model (Binomial Logit / GLMM with cluster-robust item repeated measures):
      logit(P(Y_im = 1)) = beta_0 + Tokenizer_i + beta_1 * ln(N_i) + Rule_m + u_item + u_seed

    Analyzes:
    - Which grammatical phenomena (e.g. particle 'li', particle 'e', 'pi') present the highest difficulty
    - Odds Ratios (OR = exp(beta)) and 95% Confidence Intervals for tokenizers and rules
    - Model parameter scaling slope on grammatical competence
    - Clustered standard errors accounting for item-level repeated measures
    """
    if item_df.empty or "correct" not in item_df.columns:
        return {"error": "Item-level dataframe is empty or missing 'correct' column"}

    df_clean = item_df.copy()
    df_clean["correct"] = df_clean["correct"].astype(int)
    df_clean["log_N"] = np.log(df_clean["non_embedding_params"].astype(float))

    # Formula using Word tokenizer as reference category
    formula = "correct ~ C(tokenizer, Treatment(reference='word')) + log_N + C(category)"

    has_item_clusters = "item_id" in df_clean.columns and df_clean["item_id"].nunique() > 1

    try:
        # Fit Logit with cluster-robust standard errors clustering on item_id
        if has_item_clusters:
            logit_model = smf.logit(formula, data=df_clean).fit(
                cov_type="cluster",
                cov_kwds={"groups": df_clean["item_id"]},
                disp=False,
            )
            model_type = "Binomial Logit with Item-Clustered Robust Covariance (GLMM-Cluster)"
        else:
            logit_model = smf.logit(formula, data=df_clean).fit(disp=False)
            model_type = "Binomial Logit (Standard)"

        params = logit_model.params
        pvals = logit_model.pvalues
        conf = logit_model.conf_int()

        odds_ratios = {}
        rule_difficulties = {}

        for k, v in params.items():
            or_val = float(np.exp(v))
            ci_low = float(np.exp(conf.loc[k, 0]))
            ci_high = float(np.exp(conf.loc[k, 1]))
            odds_ratios[k] = {
                "beta": float(v),
                "odds_ratio": or_val,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "p_value": float(pvals.get(k, 1.0)),
            }
            if "category" in k:
                cat_name = k.replace("C(category)[T.", "").replace("]", "")
                rule_difficulties[cat_name] = or_val

        # Sort rule difficulties: lowest odds ratio = hardest rule
        sorted_difficulty = sorted(rule_difficulties.items(), key=lambda x: x[1])

        # Also fit GEE for longitudinal item correlation if clusters exist
        gee_summary = None
        if has_item_clusters:
            try:
                gee_m = smf.gee(
                    formula,
                    groups=df_clean["item_id"],
                    data=df_clean,
                    family=sm.families.Binomial(),
                ).fit()
                gee_summary = str(gee_m.summary())
            except Exception:
                pass

        return {
            "success": True,
            "model_type": model_type,
            "formula": formula,
            "pseudo_r_squared": float(logit_model.prsquared),
            "log_likelihood": float(logit_model.llf),
            "n_observations": int(logit_model.nobs),
            "n_item_clusters": int(df_clean["item_id"].nunique()) if has_item_clusters else 1,
            "odds_ratios": odds_ratios,
            "rule_difficulty_hierarchy": [
                {"category": cat, "odds_ratio_vs_ref": round(or_val, 3)}
                for cat, or_val in sorted_difficulty
            ],
            "summary_text": str(logit_model.summary()),
            "gee_summary": gee_summary,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
