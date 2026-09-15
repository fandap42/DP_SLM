"""
Comprehensive Statistical Analysis & Visualization of SLM Factorial Experiments.
Outputs:
- Mixed-effects regression coefficients, standard errors, p-values
- Factorial ANOVA table with effect sizes (eta-squared)
- Empirical scaling law fits (Kaplan / Chinchilla parameters)
- Publication-quality charts in figures/
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from slm_tokipona.stats.mixed_models import fit_mixed_effects_model, compute_factorial_anova
from slm_tokipona.stats.scaling_laws import fit_scaling_law, fit_tokenizer_scaling_curves


def run_statistical_analysis(results_csv: Path | str, output_dir: Path | str, figures_dir: Path | str):
    results_csv = Path(results_csv)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    print(f"[Analysis] Loaded {len(df)} experiment rows from {results_csv}")

    # Set matplotlib style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    })

    # =========================================================================
    # 1. Linear Mixed-Effects Model & OLS
    # =========================================================================
    print("\n[Analysis] Fitting Mixed-Effects Model / OLS Regression...")
    lmem_results = fit_mixed_effects_model(df)
    with open(output_dir / "mixed_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(lmem_results["summary_text"])

    # =========================================================================
    # 2. Factorial ANOVA & Effect Sizes
    # =========================================================================
    print("[Analysis] Computing Factorial ANOVA & Effect Sizes (eta-squared)...")
    try:
        anova_table = compute_factorial_anova(df)
        anova_table.to_csv(output_dir / "anova_table.csv")
        anova_str = anova_table.to_string()
    except Exception as e:
        anova_str = f"ANOVA computation note: {e}"

    # =========================================================================
    # 3. Empirical Scaling Laws (Kaplan / Chinchilla)
    # =========================================================================
    print("[Analysis] Fitting Empirical Scaling Laws...")
    scaling_law_results = fit_scaling_law(df)
    tokenizer_curves = fit_tokenizer_scaling_curves(df)

    with open(output_dir / "scaling_laws.json", "w", encoding="utf-8") as f:
        json.dump({
            "joint_scaling_law": scaling_law_results,
            "tokenizer_scaling_curves": tokenizer_curves,
        }, f, indent=2)

    # =========================================================================
    # 4. Visualization & Publication Charts
    # =========================================================================
    print("[Analysis] Generating publication-quality charts...")

    # Color mapping for tokenizers
    colors = {"char": "#2b5c8f", "bpe": "#2a9d8f", "word": "#e76f51"}
    markers = {"char": "o", "bpe": "s", "word": "^"}

    # Chart 1: Data Scaling (BPC vs Data Fraction / Words)
    fig, ax = plt.subplots(figsize=(7, 5))
    for tok, group in df.groupby("tokenizer"):
        group_mean = group.groupby("data_fraction")["val_bpc"].mean().reset_index()
        group_std = group.groupby("data_fraction")["val_bpc"].std().fillna(0).reset_index()

        ax.errorbar(
            group_mean["data_fraction"] * 100,
            group_mean["val_bpc"],
            yerr=group_std["val_bpc"],
            label=f"Tokenizer: {tok.upper()}",
            color=colors.get(tok, "black"),
            marker=markers.get(tok, "o"),
            linewidth=2,
            capsize=4,
        )

    ax.set_xlabel("Corpus Size (% of Full Training Set)")
    ax.set_ylabel("Validation Bits-Per-Character (BPC) ↓")
    ax.set_title("Toki Pona SLM: Data Scaling by Tokenization Strategy")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "scaling_data_bpc.png")
    plt.close(fig)

    # Chart 2: Parameter Scaling (BPC vs Non-embedding Parameters)
    fig, ax = plt.subplots(figsize=(7, 5))
    for tok, group in df.groupby("tokenizer"):
        group_mean = group.groupby("non_embedding_params")["val_bpc"].mean().reset_index()
        ax.plot(
            group_mean["non_embedding_params"],
            group_mean["val_bpc"],
            label=f"Tokenizer: {tok.upper()}",
            color=colors.get(tok, "black"),
            marker=markers.get(tok, "o"),
            linewidth=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Non-Embedding Parameters (log scale)")
    ax.set_ylabel("Validation Bits-Per-Character (BPC) ↓")
    ax.set_title("Toki Pona SLM: Parameter Scaling Laws")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "scaling_params_bpc.png")
    plt.close(fig)

    # Chart 3: Interaction Plot (Data Fraction x Tokenizer)
    fig, ax = plt.subplots(figsize=(7, 5))
    pivot_data = df.pivot_table(index="data_fraction", columns="tokenizer", values="val_bpc", aggfunc="mean")
    for tok in pivot_data.columns:
        ax.plot(
            pivot_data.index * 100,
            pivot_data[tok],
            marker=markers.get(tok, "o"),
            color=colors.get(tok, "black"),
            linewidth=2,
            label=f"{tok.upper()}",
        )
    ax.set_xlabel("Corpus Fraction (%)")
    ax.set_ylabel("Mean Validation BPC ↓")
    ax.set_title("Statistical Interaction: Corpus Size × Tokenizer")
    ax.legend(title="Tokenizer")
    fig.tight_layout()
    fig.savefig(figures_dir / "interaction_effects.png")
    plt.close(fig)

    # Chart 4: Grammatical Acceptability vs BPC
    fig, ax = plt.subplots(figsize=(7, 5))
    for tok, group in df.groupby("tokenizer"):
        ax.scatter(
            group["val_bpc"],
            group["grammar_acc"] * 100,
            label=tok.upper(),
            color=colors.get(tok, "black"),
            marker=markers.get(tok, "o"),
            s=80,
            alpha=0.85,
        )

    ax.set_xlabel("Validation Bits-Per-Character (BPC) ↓")
    ax.set_ylabel("Grammar Acceptability Accuracy (%) ↑")
    ax.set_title("Grammatical Competence vs. Information-Theoretic BPC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "grammar_vs_bpc.png")
    plt.close(fig)

    # =========================================================================
    # 5. Markdown Report Synthesis
    # =========================================================================
    report_md = f"""# Statistická analýza výsledků: SLM Toki Pona na malých datech

Tento report shrnuje empirické výsledky faktorového experimentu zkoumajícího vliv **velikosti korpusu ($D$)**, **velikosti modelu ($N$)** a **tokenizace ($T$)** na jazykové modelování minimalistického jazyka toki pona.

---

## 1. Popis experimentálního souboru
- **Celkový počet běhů:** {len(df)}
- **Faktory:**
  - Tokenizace: `{', '.join(df['tokenizer'].unique())}`
  - Modely: `{', '.join(df['model_size'].unique())}`
  - Frakce korpusu: `{', '.join(str(int(x*100)) + '%' for x in sorted(df['data_fraction'].unique()))}`
  - Semínka (Seeds): `{', '.join(str(s) for s in df['seed'].unique())}`

---

## 2. Průměrné výsledky dle konfigurace (BPC & Gramatika)

```
{df.groupby(['tokenizer', 'model_size', 'data_fraction'])[['val_bpc', 'grammar_acc']].mean().to_string()}
```

---

## 3. Výsledky lineárního modelu / Mixed-Effects Regression
Model specifikace: `{lmem_results.get('formula')}`
Metoda odhadu: `{lmem_results.get('model_type')}`

### Odhadnuté koeficienty a signifikance:
| Proměnná / Prediktor | Koeficient ($\beta$) | p-hodnota | 95% Interval spolehlivosti |
| :--- | :--- | :--- | :--- |
"""
    for var, coef in lmem_results["coefficients"].items():
        pval = lmem_results["p_values"].get(var, float("nan"))
        ci = lmem_results["confidence_intervals"].get(var, (float("nan"), float("nan")))
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        report_md += f"| `{var}` | {coef:.4f} {sig} | {pval:.4e} | [{ci[0]:.4f}, {ci[1]:.4f}] |\n"

    report_md += f"""
*Poznámka: *** p < 0.001, ** p < 0.01, * p < 0.05*

---

## 4. Analýza rozptylu (ANOVA) a velikosti účinku ($\\eta^2$)

```
{anova_str}
```

---

## 5. Odhad škálovacích zákonů (Scaling Laws)
- **Kaplan / Chinchilla rovnice:**
  `{scaling_law_results.get('equation', 'N/A')}`
- **Koeficient determinace ($R^2$):** `{scaling_law_results.get('r_squared', 'N/A')}`

### Škálovací křivky dle tokenizéru ($BPC = a \cdot D^{{-\\beta}}$):
"""
    for tok, c_data in tokenizer_curves.items():
        if "formula" in c_data:
            report_md += f"- **{tok.upper()}:** `{c_data['formula']}` ($R^2 = {c_data['r_squared']:.3f}$)\n"

    report_md += f"""
---

## 6. Generované grafy
Grafy byly uloženy do složky `figures/`:
1. `figures/scaling_data_bpc.png` – Škálování podle velikosti korpusu
2. `figures/scaling_params_bpc.png` – Škálování podle počtu parametrů
3. `figures/interaction_effects.png` – Interakční graf Tokenizér × Data
4. `figures/grammar_vs_bpc.png` – Vztah mezi BPC a gramatickou přijatelností
"""

    with open(output_dir / "statistical_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[Analysis] Complete! Statistical report saved to {output_dir / 'statistical_report.md'}")
    return report_md
