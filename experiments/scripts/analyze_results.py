"""
Convenience entry point for statistical analysis and visualization.
Reads from results/metrics/results.csv and exports charts to results/figures/.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from src.evaluation.stats import fit_mixed_effects_model, compute_factorial_anova, fit_scaling_law, fit_tokenizer_scaling_curves


def run_statistical_analysis(results_csv: Path | str, output_dir: Path | str, figures_dir: Path | str):
    results_csv = Path(results_csv)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    print(f"[Analysis] Loaded {len(df)} experiment rows from {results_csv}")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    })

    # 1. MixedLM & OLS
    print("\n[Analysis] Fitting Mixed-Effects Model / OLS Regression...")
    lmem_results = fit_mixed_effects_model(df)
    with open(output_dir / "mixed_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(lmem_results["summary_text"])

    # 2. ANOVA
    print("[Analysis] Computing Factorial ANOVA & Effect Sizes (eta-squared)...")
    try:
        anova_table = compute_factorial_anova(df)
        anova_table.to_csv(output_dir / "anova_table.csv")
        anova_str = anova_table.to_string()
    except Exception as e:
        anova_str = f"ANOVA computation note: {e}"

    # 3. Scaling Laws
    print("[Analysis] Fitting Empirical Scaling Laws...")
    scaling_law_results = fit_scaling_law(df)
    tokenizer_curves = fit_tokenizer_scaling_curves(df)

    with open(output_dir / "scaling_laws.json", "w", encoding="utf-8") as f:
        json.dump({
            "joint_scaling_law": scaling_law_results,
            "tokenizer_scaling_curves": tokenizer_curves,
        }, f, indent=2)

    # 4. Publication Charts
    print("[Analysis] Generating publication-quality charts...")
    colors = {"char": "#2b5c8f", "bpe": "#2a9d8f", "word": "#e76f51"}
    markers = {"char": "o", "bpe": "s", "word": "^"}

    # Chart 1: Data Scaling
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
    fig.savefig(figures_dir / "scaling_data_bpc.pdf")
    plt.close(fig)

    # Chart 2: Parameter Scaling
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
    fig.savefig(figures_dir / "scaling_params_bpc.pdf")
    plt.close(fig)

    # Chart 3: Interaction Plot
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
    fig.savefig(figures_dir / "interaction_effects.pdf")
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
    fig.savefig(figures_dir / "grammar_vs_bpc.pdf")
    plt.close(fig)

    # Markdown report
    report_md = f"""# Statistická analýza výsledků: SLM Toki Pona na malých datech

- **Celkový počet běhů:** {len(df)}
- **Tokenizace:** `{', '.join(df['tokenizer'].unique())}`
- **Modely:** `{', '.join(df['model_size'].unique())}`
- **Frakce korpusu:** `{', '.join(str(int(x*100)) + '%' for x in sorted(df['data_fraction'].unique()))}`

---

## 1. Průměrné výsledky dle konfigurace (BPC & Gramatika)

```
{df.groupby(['tokenizer', 'model_size', 'data_fraction'])[['val_bpc', 'grammar_acc']].mean().to_string()}
```

---

## 2. Analýza rozptylu (ANOVA) a velikosti účinku ($\\eta^2$)

```
{anova_str}
```

---

## 3. Odhad škálovacích zákonů (Scaling Laws)
- **Kaplan / Chinchilla rovnice:**
  `{scaling_law_results.get('equation', 'N/A')}`
- **Koeficient determinace ($R^2$):** `{scaling_law_results.get('r_squared', 'N/A')}`
"""
    with open(output_dir / "statistical_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[Analysis] Report saved to {output_dir / 'statistical_report.md'}")


if __name__ == "__main__":
    run_statistical_analysis(
        root / "results" / "metrics" / "results.csv",
        root / "results" / "metrics",
        root / "results" / "figures",
    )
