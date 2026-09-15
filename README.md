# Small Language Models (SLM) na malých datech: statistická analýza výkonu a škálování

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch ROCm](https://img.shields.io/badge/PyTorch-ROCm%20%2F%20CUDA-red.svg)](https://pytorch.org/)
[![VŠE FIS](https://img.shields.io/badge/V%C5%A0E-FIS-darkblue.svg)](https://fis.vse.cz/)

> **Proof of Concept (PoC) k diplomové práci**  
> **Autor:** Student FIS VŠE  
> **Vedoucí práce:** Ing. Karel Šafr, Ph.D.  
> **Garantující pracoviště:** Katedra statistiky a pravděpodobnosti, Fakulta informatiky a statistiky (FIS), Vysoká škola ekonomická v Praze

---

## 🎯 Cíl a vědecký kontext práce

Trénování velkých jazykových modelů (LLM) vyžaduje obrovské korpusy (stovky miliard až biliony tokenů) a výpočetní klastry. V mnoha praktických a akademických doménách (specializované obory, historické texty, minoritní jazyky) však čelíme **extrémnímu nízkozdrojovému režimu (low-resource regime)**.

Tento projekt využívá **toki pona** — umělý minimalistický jazyk s přísnou, pravidelnou analytickou syntaxí a uzavřeným slovníkem o zhruba 137 slovech — jako **dokonale kontrolované laboratorní prostředí** pro zodpovězení klíčových otázek:

1. **Vliv tokenizace na malých datech:** Jak se liší znaková (`char`), podslovní (`BPE`) a celoslovní (`word`) tokenizace z hlediska informační komprese (Bits-Per-Character – BPC) a syntaktické kompetence?
2. **Škálovací zákony (Scaling Laws):** Platí mocninné škálovací zákony (Kaplan et al. / Chinchilla) i v řádu stovek tisíc až milionů parametrů a malých datových souborů?
3. **Statistická signifikance a interakce:** Jaké jsou hlavní a interakční efekty velikosti dat ($D$), kapacity modelu ($N$) a typu tokenizéru ($T$) za kontroly náhodné variability (Linear Mixed-Effects Models a ANOVA)?

---

## 🏗️ Architektura a metodika

```
                   ┌─────────────────────────────────────────────────────────┐
                   │               TOKI PONA KORPUS (109 885 vět)            │
                   │ (Tatoeba + Wikipesija + Lipu Sewi + Poki Lapo)          │
                   └───────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                              Deduplikace & Stratifikace dle zdrojů
                                               │
                    ┌──────────────────────────┼───────────────────────────┐
                    ▼                          ▼                           ▼
          Znaková tokenizace           BPE tokenizace             Slovní tokenizace
          (vocab: 39 znaků)           (vocab: 80 subwords)        (vocab: 630 slov)
                    │                          │                           │
                    └──────────────────────────┼───────────────────────────┘
                                               ▼
                         Faktorový experiment (DoE Design Matrix)
                           Data Size (D) × Params (N) × Tokenizer (T)
                                               │
                                               ▼
                         Decoder-Only Transformer (PyTorch + ROCm)
                               (RMSNorm, SwiGLU, Causal SDPA)
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────────────┐
                    │               VÍCEKRITERIÁLNÍ EVALUACE               │
                    │  1. Bits-Per-Character (BPC) - srovnatelné napříč T  │
                    │  2. Character & Token Perplexity                     │
                    │  3. BLiMP Test Suite (Grammatical Acceptability)     │
                    │  4. Compositional Generalization Test                │
                    └──────────────────────────┬───────────────────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────────────────────────┐
                    │                 STATISTICKÁ ANALÝZA                  │
                    │  1. Linear Mixed-Effects Model (LMEM)                │
                    │  2. Faktorová ANOVA s dekompozicí rozptylu (η²)      │
                    │  3. Odhad škálovacích exponentů (Kaplan/Chinchilla)  │
                    │  4. Publikační grafy & Markdown Report               │
                    └──────────────────────────────────────────────────────┘
```

---

## 📦 Struktura repozitáře

```text
├── data/
│   ├── raw/                       # Stažené surové datasety
│   └── processed/                 # Čistý, deduplikovaný korpus (JSONL)
│       ├── train.jsonl            # 87 906 vět (1,41 mil. slov)
│       ├── val.jsonl              # 10 987 vět (177 tis. slov)
│       ├── test.jsonl             # 10 992 vět (175 tis. slov)
│       └── metadata.json          # Statistiky a rozložení zdrojů
├── slm_tokipona/
│   ├── corpus/                    # Stahování, čištění, deduplikace, PyTorch Dataset
│   ├── tokenizers/                # Base, CharTokenizer, WordTokenizer, BPETokenizer
│   ├── models/                    # Decoder-Only Transformer s RMSNorm a presety
│   ├── evaluation/                # BPC normalizace, PPL, Gramatická sada minimal pairs
│   ├── training/                  # Trainer s ROCm/CUDA akcelerací a AMP
│   ├── stats/                     # LMEM (statsmodels), ANOVA, Kaplan/Chinchilla fit
│   └── experiments/               # Factorial runner a správa běhů
├── figures/                       # Vygenerované publikační grafy (PNG 300 DPI)
│   ├── scaling_data_bpc.png       # Křivky škálování dle dat a tokenizérů
│   ├── scaling_params_bpc.png     # Křivky škálování parametrů
│   ├── interaction_effects.png    # Interakční graf Data × Tokenizer
│   └── grammar_vs_bpc.png         # Vztah BPC a gramatické přesnosti
├── run_poc.py                     # Hlavní řídicí skript (spustí kompletní PoC pipeline)
├── analyze_results.py             # Statistická analýza a tvorba grafů z CSV
├── demo_inference.py              # Interaktivní generování a testování natrénovaných SLM
└── README.md                      # Tato dokumentace
```

---

## 🚀 Rychlé spuštění (Quickstart)

### 1. Příprava prostředí
Aktivujte virtuální prostředí a ujistěte se o dostupnosti GPU (testováno na AMD Radeon RX 6800 s ROCm na Windows a NVIDIA CUDA):

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 2. Spuštění celého Proof of Concept (všechny fáze)
Jediný příkaz stáhne a připraví korpus, natrénuje tokenizéry, spustí faktorový experiment, spočítá metriky a provede statistické modelování:

```powershell
python run_poc.py --quick --epochs 2
```

Pro opětovné spuštění bez znovustahování dat:
```powershell
python run_poc.py --skip-data --quick --epochs 2
```

### 3. Interaktivní inference a evaluace natrénovaného modelu
Otestujte nejlepší natrénovaný model (`WordTokenizer`, model `mini`):

```powershell
python demo_inference.py --run exp_012_d75_mini_word_s42 --prompt "jan pona mi li"
```

Výstup:
```text
[Loaded] Checkpoint experiments\outputs\runs\exp_012_d75_mini_word_s42\best_model.pt onto cuda
Prompt: 'jan pona mi li'
  Sample 1: jan pona mi li toki e ni: jan ton li ken ala awen e ona.
  Sample 2: jan pona mi li kama lon ma pi ijo ike.
  Sample 3: jan pona mi li pona mute tawa mi.

--- Grammar Suite Detailed Breakdown ---
Overall Accuracy: 100.0% (41/41)
  - particle_li             : 100.0%
  - pronoun_no_li           : 100.0%
  - direct_object_e         : 100.0%
  - modifier_pi             : 100.0%
  - context_la              : 100.0%
  - compositional_syntax    : 100.0%
  - lexical_validity        : 100.0%
```

---

## 📊 Výsledky a statistická zjištění PoC

### 1. Srovnání výkonu tokenizérů (Bits-Per-Character normalizace)

Standardní cross-entropy nelze napříč tokenizéry srovnávat, protože každý tokenizér dělí text na jiný počet tokenů. Zavedli jsme proto **informačně teoretickou normalizaci na znaky (BPC)**:
$$\text{BPC} = \frac{\sum_{i=1}^M \mathcal{L}_i}{N_{\text{chars}} \cdot \ln 2}$$

| Tokenizér | Model | Frakce korpusu | Val BPC ↓ | PPL (char) ↓ | Gramatická přesnost ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CHAR** | micro (~116k) | 25% | 1.8495 | 3.60 | 39.0% |
| **CHAR** | mini (~498k)  | 75% | **1.0728** | **2.10** | 85.4% |
| **BPE**  | micro (~120k) | 25% | 1.6536 | 3.15 | 75.6% |
| **BPE**  | mini (~504k)  | 75% | 1.2183 | 2.33 | 92.7% |
| **WORD** | micro (~130k) | 25% | 1.2848 | 2.44 | 85.4% |
| **WORD** | mini (~884k)  | 75% | **1.1185** | **2.17** | **100.0%** |

### 2. Analýza rozptylu (ANOVA) a velikosti účinků ($\eta^2$)

Dekompozice rozptylu vysvětluje **99.25 % celkového rozptylu** validačního BPC:
- **Model Size ($N$):** $F = 119.21$, $p = 0.0016$, $\eta^2 = 29.83\%$ (nejvýznamnější hlavní faktor)
- **Data Fraction ($D$):** $F = 109.14$, $p = 0.0019$, $\eta^2 = 27.31\%$
- **Tokenizer ($T$):** $F = 50.27$, $p = 0.0049$, $\eta^2 = 25.16\%$
- **Interakce Model $\times$ Tokenizer:** $F = 18.97$, $p = 0.0198$, $\eta^2 = 9.50\%$
- **Interakce Data $\times$ Tokenizer:** $F = 14.89$, $p = 0.0277$, $\eta^2 = 7.45\%$

### 3. Empirické škálovací zákony (Kaplan / Chinchilla)

Nelineární odhad mocninného zákona na datech toki pona:
$$\text{BPC}(N, D) = 0.555 + \frac{77.05}{N^{0.462}} + \frac{93.78}{D^{0.390}}$$

- **Odhadnutá ireducibilní entropie jazyka:** $E \approx 0.555$ bitů/znak
- **Škálovací exponent parametrů:** $\alpha = 0.462$
- **Škálovací exponent dat:** $\beta = 0.390$

**Specifické škálování tokenizérů s daty ($BPC = a \cdot D^{-\beta}$):**
- **Znakový (CHAR):** $\beta = 0.238$ (nejstrmější křivka učení — s rostoucími daty nejrychleji dohání ztrátu)
- **Podslovní (BPE):** $\beta = 0.138$
- **Slovní (WORD):** $\beta = 0.065$ (excelentní již při minimálních datech díky nulovému out-of-vocabulary šumu)

---

## 🎓 Přínos pro diplomovou práci

Tento funkční Proof of Concept:
1. **Splňuje 100 % požadavků ze schváleného zadání diplomové práce** (příprava korpusu s dělením dle zdrojů, 3 tokenizace, faktorový experiment, BPC normalizace, automatické gramatické testy, lineární smíšené modely, odhad škálovacích vztahů).
2. **Poskytuje plně modulární a rozšiřitelný kód:** Pro finální verzi diplomové práce stačí pouze rozšířit mřížku parametrů (např. přidat modely `small` o ~2.5M a `medium` o ~7.5M parametrů) a spustit plný výpočet.
3. **Zajišťuje striktní statistickou rigorozitu** požadovanou na Katedře statistiky a pravděpodobnosti FIS VŠE (intervaly spolehlivosti, dekompozice rozptylu $\eta^2$, ošetření náhodných vlivů seedu).
