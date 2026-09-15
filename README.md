# Diplomka Monorepo: Small Language Models (SLM) na malých datech

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch ROCm](https://img.shields.io/badge/PyTorch-ROCm%20%2F%20CUDA-red.svg)](https://pytorch.org/)
[![VŠE FIS](https://img.shields.io/badge/V%C5%A0E-FIS-darkblue.svg)](https://fis.vse.cz/)

> **Diplomová práce:** Small Language Models (SLM) na malých datech: statistická analýza výkonu a škálování  
> **Autor:** Student FIS VŠE  
> **Vedoucí práce:** Ing. Karel Šafr, Ph.D.  
> **Garantující pracoviště:** Katedra statistiky a pravděpodobnosti, Fakulta informatiky a statistiky (FIS), Vysoká škola ekonomická v Praze

---

## 🏛️ Struktura monorepositoráře

```text
diplomka-slm/
├── .gitignore                  # Přesně definovaná pravidla (ignoruje data/, váhy *.pt)
├── README.md                   # Kompletní průvodce a dokumentace projektu
├── pyproject.toml              # Moderní Python build & package definice
│
├── data/                       # IGNOROVAT V GITU! (pouze .gitkeep)
│   ├── raw/                    # Původní nestrukturovaný text toki pona (Parquet)
│   ├── processed/              # Vyčištěný, deduplikovaný text (JSONL, metadata)
│   └── tokenized/              # Připravená tokenizovaná data
│
├── src/                        # Samotný modulární Python balíček
│   ├── data/                   # Stahování, čištění, normalizace, PyTorch Dataset
│   ├── tokenization/           # Znaková, BPE podslovní a celoslovní tokenizace
│   ├── models/                 # Architektura Decoder-Only Transformer (RMSNorm, SwiGLU)
│   ├── training/               # Trénovací smyčka s podporou ROCm, AMP a schedulingu
│   └── evaluation/             # Výpočet BPC, BLiMP testy gramatiky, ANOVA a LMEM
│
├── experiments/                # Orchestrace faktorového pokusu
│   ├── configs/                # Konfigurace běhů (poc_quick.json, factorial_full.json)
│   ├── scripts/                # Spouštěcí bash/powershell/slurm skripty pro lokál i cluster
│   └── notebooks/              # Jupyter notebooky čistě na EDA a tvorbu grafů
│
├── results/                    # Výstupy
│   ├── figures/                # Finální grafy (PNG, PDF), které jdou do práce
│   ├── metrics/                # CSV/JSON soubory s výsledky běhů (bpc, loss, časy, tokenizéry)
│   └── models/                 # IGNOROVAT V GITU! (checkpointy modelů na disku)
│
├── docs/                       # Zdroje a poznámky
│   ├── notes/                  # Metodické poznámky ke škálovacím zákonům
│   └── toki_pona_grammar/      # Lingvistická pravidla pro tvorbu evaluace minimal pairs
│
└── thesis/                     # Samotný text práce (LaTeX šablona FIS VŠE)
    ├── main.tex                # Hlavní řídicí LaTeX soubor
    ├── references.bib          # BibTeX citace (Kaplan, Chinchilla, Vaswani, Lang)
    ├── chapters/               # Jednotlivé kapitoly (01_intro.tex až 06_conclusion.tex)
    └── figures/                # Kopie grafů z results/figures/ pro kompilaci PDF
```

---

## 🚀 Rychlý start a spuštění experimentů

### 1. Příprava virtuálního prostředí
```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Spuštění faktorového pokusu
Experiment lze spustit buď přes nainstalovaný CLI příkaz, nebo přímo skriptem:

```powershell
# Možnost A: CLI příkaz (dostupný odkudkoliv ve venvu)
slm-run --config experiments/configs/poc_quick.json

# Možnost B: Spouštěcí skript
python experiments/scripts/run_experiments.py --config experiments/configs/poc_quick.json
```

### 3. Spuštění na výpočetním clusteru (SLURM)
Pro spuštění plného výpočetního rastru (např. 144 běhů) na školním clusteru FIS VŠE či MetaCentrum:
```bash
sbatch experiments/scripts/run_slurm.sh
```

### 4. Spuštění statistické analýzy a přegenerování grafů
```powershell
python analyze_results.py
```
Vygeneruje aktualizované grafy v `results/figures/` a statistický report v `results/metrics/statistical_report.md`.

### 5. Kompilace textu diplomové práce (LaTeX)
Text práce je připraven v adresáři `thesis/`:
```bash
cd thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📊 Přehled výsledků z PoC běhu

| Tokenizér | Model | Frakce korpusu | Val BPC ↓ | PPL (char) ↓ | Gramatická přesnost (BLiMP) ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CHAR** | micro (~116k) | 25 % | 1.8495 | 3.60 | 39.0 % |
| **CHAR** | mini (~498k)  | 75 % | **1.0728** | **2.10** | 85.4 % |
| **BPE**  | micro (~120k) | 25 % | 1.6536 | 3.15 | 75.6 % |
| **BPE**  | mini (~504k)  | 75 % | 1.2183 | 2.33 | 92.7 % |
| **WORD** | micro (~130k) | 25 % | 1.2848 | 2.44 | 85.4 % |
| **WORD** | mini (~884k)  | 75 % | **1.1185** | **2.17** | **100.0 %** |

**Empirický Kaplanův zákon pro Toki Pona:**
$$\text{BPC}(N, D) = 0.555 + \frac{77.05}{N^{0.462}} + \frac{93.78}{D^{0.390}}$$
ANOVA prokazuje statistickou signifikanci modelů, dat i tokenizace ($p < 0.05$) s vysvětlením **99.25 % rozptylu**.
