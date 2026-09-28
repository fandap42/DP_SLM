# Diplomka Monorepo: Small Language Models (SLM) na malých datech

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch ROCm](https://img.shields.io/badge/PyTorch-ROCm%20%2F%20CUDA-red.svg)](https://pytorch.org/)
[![VŠE FIS](https://img.shields.io/badge/V%C5%A0E-FIS-darkblue.svg)](https://fis.vse.cz/)

> **Diplomová práce:** Small Language Models (SLM) na malých datech: statistická analýza výkonu a škálování  
> **Autor:** František Pavlík  
> **Vedoucí práce:** Ing. Karel Šafr, Ph.D.  
> **Garantující pracoviště:** Katedra statistiky a pravděpodobnosti, Fakulta informatiky a statistiky (FIS), Vysoká škola ekonomická v Praze

---

## 🏛️ Čtyři pilíře evaluace a statistické syntézy

Podle zadání diplomové práce na Katedře statistiky a pravděpodobnosti FIS VŠE je hodnocení postaveno na čtyřech provázaných pilířích:

### 1. Pilíř: Standardní testovací sada a normalizovaná křížová entropie (BPC)
- **Normalizace na znaky (Bits per Character – BPC):**
  $$\text{Loss}_{\text{char}} = \frac{-\sum_{t} \ln P(x_t \mid x_{<t})}{N_{\text{znaky}}}, \quad \text{BPC} = \frac{\text{Loss}_{\text{char}}}{\ln(2)}$$
  Zajišťuje matematicky rigorózní a spravedlivé srovnání napříč znakovou, slovní a subword (BPE) tokenizací.
- **Doménová generalizace (Domain Split):**
  Vyhodnocení BPC odděleně pro data z domény viděné v tréninku (*in-domain*: Tatoeba) a neviděných domén (*out-of-domain*: Wikipesija, Poki Lapo, Lipu Sewi) s vyčíslením doménové mezery $\Delta\text{BPC}_{\text{domain}}$.

### 2. Pilíř: Automaticky generované testy gramatické přijatelnosti (BLiMP Toki Pona)
- **Přístup minimálních párů:**
  Model porovnává logaritmickou pravděpodobnost gramaticky správné věty ($S_{\text{correct}}$) vůči větě se syntaktickou chybou ($S_{\text{incorrect}}$). Úspěch nastává, pokud $P(S_{\text{correct}}) > P(S_{\text{incorrect}})$.
- **Automatický generátor pravidel:**
  Syntetizuje 447 minimálních párů pokrývajících stěžejní pravidla toki pona:
  1. *Pravidlo částice li:* po *mi/sina* se *li* nepíše, u ostatních podmětů je povinné.
  2. *Pravidlo předmětu e:* přímý předmět po tranzitivním slovese vyžaduje částici *e*.
  3. *Slovosled a modifikátory:* přívlastek stojí vždy za podstatným jménem (*tomo suli* vs. *suli tomo*).
  4. *Závorkovací částice pi:* používá se výhradně pro skupinu 2+ modifikátorů (*tomo suli* vs. *tomo pi suli*).
  5. *Kontextová částice la* a *lexikální validita* (slova toki pona vs. pseudononsensy).

### 3. Pilíř: Testy kompozičního zobecnění (Hold-out Compounding)
- **Metodika Hold-Out skládání:**
  20 vybraných víceslovných konceptů (*telo nasa*, *tomo telo*, *jan utala*, *ilo moku*, *tomo sona*, ...) je testováno v syntakticky kontrolovaných větách vůči 20 známým kontrolním spojením (*jan pona*, *toki pona*, *tomo suli*, ...).
- **Metriky:**
  BPC na neviděných spojeních ($\text{BPC}_{\text{heldout}}$), na známých spojeních ($\text{BPC}_{\text{seen}}$), kompoziční generalizační mezera ($\text{Gap} = \text{BPC}_{\text{heldout}} - \text{BPC}_{\text{seen}}$) a přesnost v kompozičních minimálních párech (pořadí slov ve složenině).

### 4. Pilíř: Statistická analýza na úrovni FIS VŠE
- **Model 1: Škálovací zákony (Power-Law Regrese):**
  Linearizovaný log-log model $\ln(\text{BPC}) = \beta_0 + \alpha \ln(N) + \beta \ln(D) + \varepsilon$, srovnání směrnic $\alpha, \beta$ pro jednotlivé tokenizace a nelineární Kaplanova křivka.
- **Model 2: Hierarchické lineární modely se smíšenými efekty (LMM):**
  Model zachycující strukturu experimentu, náhodné efekty seedu a vícenásobná měření domén.
- **Model 3: Analýza rozptylu (Faktorová ANOVA & Kontrasty):**
  Testování hlavních efektů a interakcí ($\text{Tokenizace} \times \text{Data}$, $\text{Tokenizace} \times \text{Model}$) s výpočtem $\eta^2$ a marginálních průměrů.
- **Model 4: Zobecněný lineární model pro gramatiku (GLMM):**
  Logistická regrese $\text{logit}(P(Y=1)) \sim \text{Tokenizace} + \ln(N) + \text{Pravidlo} + u_{\text{seed}}$ nad 5 811 položkami testu. Odhaduje poměry šancí (OR) a hierarchii obtížnosti gramatických jevů.
- **R replikační skript:**
  `experiments/scripts/analysis_models.R` pro přímé spuštění v prostředí R (`lme4`, `emmeans`, `car`, `ggplot2`).

---

## 🏛️ Struktura monorepositoráře

```text
DP_SLM/
├── .gitignore                  # Přesně definovaná pravidla (ignoruje data/, váhy *.pt, build PDF)
├── README.md                   # Kompletní průvodce a dokumentace projektu
├── pyproject.toml              # Moderní Python build & package definice
│
├── data/                       # IGNOROVAT V GITU! (pouze .gitkeep)
│   ├── raw/                    # Původní nestrukturovaný text toki pona (Parquet)
│   ├── processed/              # Vyčištěný, deduplikovaný text (JSONL, metadata)
│   ├── evaluation/             # Verzované testovací sady minimálních párů a kompozice
│   └── tokenized/              # Připravená tokenizovaná data
│
├── release_assets/             # Distribuční balíčky pro GitHub Releases
│   ├── toki_pona_corpus.tar.gz # Zkomprimovaný vyčištěný korpus
│   └── best_models.tar.gz      # Kontrolní body nejlepších modelů a tokenizéry
│
├── src/                        # Samotný modulární Python balíček
│   ├── data/                   # Stahování, čištění, normalizace, PyTorch Dataset
│   ├── tokenization/           # Znaková, BPE podslovní a celoslovní tokenizace
│   ├── models/                 # Architektura Decoder-Only Transformer (RMSNorm, SwiGLU)
│   ├── training/               # Trénovací smyčka s podporou ROCm, AMP a schedulingu
│   └── evaluation/             # BPC, doménový split, BLiMP testy, kompozice, ANOVA, LMM, GLMM
│
├── experiments/                # Orchestrace faktorového pokusu
│   ├── configs/                # Konfigurace běhů (poc_quick.json, factorial_full.json)
│   ├── scripts/                # Spouštěcí skripty pro lokál, cluster, evaluaci i R analýzu
│   │   ├── analyze_results.py  # Python syntéza 4 statistických modelů a tvorba grafů
│   │   ├── analysis_models.R   # R skript pro lme4, emmeans a car
│   │   └── run_experiments.py  # Trénovací pipeline faktorového pokusu
│   └── notebooks/              # Jupyter notebooky na EDA a tvorbu grafů
│
├── results/                    # Výstupy
│   ├── figures/                # 7 publikačních grafů (PNG 300 DPI + vektorové PDF)
│   ├── metrics/                # CSV/JSON výsledky, ANOVA tabulka, LMM a GLMM výstupy
│   └── models/                 # IGNOROVAT V GITU! (checkpointy modelů na disku)
│
├── tests/                      # Unit a integrační testy pro všechny 4 pilíře
│
└── thesis/                     # Samotný text práce (LaTeX šablona FIS VŠE)
    ├── main.tex                # Hlavní řídicí LaTeX soubor
    ├── kapitoly/               # Jednotlivé kapitoly práce (00uvod.tex až 05zaver.tex)
    └── figures/                # Schémata práce a grafy linkované z results/figures/
```

---

## 🚀 Rychlý start a spuštění příkazů

### 1. Příprava virtuálního prostředí
```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Spuštění faktorového pokusu
```powershell
# Spuštění tréninku (CLI příkaz)
slm-run --config experiments/configs/poc_quick.json
```

### 3. Komplexní evaluace modelů (všechny 4 pilíře)
```powershell
# Vyhodnocení všech natrénovaných checkpointů na testovacích datech a testech:
slm-eval --all

# Případně vyhodnocení konkrétního modelu:
slm-eval --run exp_012_d75_mini_word_s42
```

### 4. Regenerace syntetických lingvistických testů
```powershell
slm-generate-tests
```

### 5. Spuštění statistické analýzy a přegenerování grafů
```powershell
slm-analyze
```
Vygeneruje 7 publikačních grafů v `results/figures/` (PNG a PDF), GLMM analýzu, LMM a statistický report v `results/metrics/statistical_report.md`.

### 6. Spuštění statistické replikace v R
```powershell
Rscript experiments/scripts/analysis_models.R
```

---

## 📊 Přehled výsledků faktorového pokusu (13 běhů)

| Tokenizér | Model | Frakce korpusu | Test BPC ↓ | In-Domain BPC ↓ | Out-of-Domain BPC ↓ | Gramatika (BLiMP) ↑ | Kompozice (Pair Acc) ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CHAR** | micro (~98k) | 25 % | 1.8986 | 2.2032 | 1.7485 | 47.4 % | 64.4 % |
| **CHAR** | mini (~787k) | 25 % | 1.4274 | 1.5416 | 1.3712 | 66.7 % | 80.6 % |
| **CHAR** | micro (~98k) | 75 % | 1.4577 | 1.6069 | 1.3842 | 61.1 % | 76.2 % |
| **CHAR** | mini (~787k) | 75 % | **1.1034** | **1.1903** | **1.0607** | 96.6 % | 96.9 % |
| **BPE**  | micro (~98k) | 25 % | 1.6808 | 1.5282 | 1.7560 | 83.2 % | 84.4 % |
| **BPE**  | mini (~787k) | 25 % | 1.4304 | 1.2704 | 1.5092 | 94.4 % | 91.2 % |
| **BPE**  | micro (~98k) | 75 % | 1.4373 | 1.2800 | 1.5148 | 91.5 % | 91.9 % |
| **BPE**  | mini (~787k) | 75 % | 1.2429 | 1.1152 | 1.3059 | 98.0 % | 98.8 % |
| **WORD** | micro (~98k) | 25 % | 1.2974 | 1.2086 | 1.3412 | 97.8 % | 95.0 % |
| **WORD** | mini (~787k) | 25 % | 1.2127 | 1.1149 | 1.2608 | 96.4 % | 95.6 % |
| **WORD** | micro (~98k) | 75 % | 1.2088 | 1.1098 | 1.2576 | 97.8 % | 99.4 % |
| **WORD** | mini (~787k) | 75 % | 1.1357 | 1.0396 | 1.1831 | 98.7 % | 98.8 % |
| **WORD** | large (~37.7M)| 100 % | **1.0231** | **0.9412** | **1.0634** | **98.4 %** | **98.8 %** |

**Empirický Kaplanův zákon pro Toki Pona:**
$$\text{BPC}(N, D) = 0.645 + \frac{539.07}{N^{0.645}} + \frac{101.87}{D^{0.397}} \quad (R^2 = 0.622)$$

**Linearizovaný mocninný zákon:**
$$\ln(\text{BPC}) = \beta_0 - 0.052 \ln(N) - 0.128 \ln(D) \quad (R^2 = 0.609)$$

ANOVA prokazuje statistickou signifikanci modelů, dat i interakcí ($p < 0.05$). GLMM analýza 5 811 lingvistických testů prokazuje, že nejobtížnějším gramatickým pravidlem pro malé modely je závorkovací částice *pi* ($\text{OR} = 0.480$), zatímco celoslovní tokenizace signifikantně předčí znakovou i BPE reprezentaci v rychlosti osvojování syntaxe.
