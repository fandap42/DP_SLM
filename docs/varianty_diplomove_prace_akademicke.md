# Tři finální varianty diplomové práce (DP_SLM) – Kompletní akademická a technická specifikace

> **Garantující pracoviště:** Katedra statistiky a pravděpodobnosti (KSTP), Fakulta informatiky a statistiky VŠE v Praze  
> **Vedoucí práce:** Ing. Karel Šafr, Ph.D.  
> **Autor:** Bc. František Pavlík  
> **Účel dokumentu:** Kompletní vědecké, matematické, statistické a inženýrské zadání, podle kterého může člověk nebo autonomní AI agent okamžitě začít implementovat kód, spouštět faktorové experimenty, počítat statistické modely a psát text diplomové práce.

---

## 1. Společný vědecký základ všech tří variant (Proč Toki Pona $\to$ LLM)

U velkých jazykových modelů (např. 70B LLM na angličtině se slovníkem $|V| \approx 128\,000$ tokenů) tvoří matice vnoření (embeddingy) jen **~2 % parametrů** a **98 %** leží v hlubokých výpočetních vrstvách. U malých modelů (1M–10M parametrů) trénovaných na přirozených jazycích však slovník pohltí až 85 % celkového počtu parametrů (*Embedding Bottleneck*), což zcela zkresluje jejich dynamiku učení oproti velkým LLM.

Úměrným zmenšením slovníku na minimalistický jazyk **Toki Pona** ($|V| \approx 137$ kořenových slov) se **obnovuje přesná strukturální proporce velkého LLM (2 % embeddingy, 98 % sekvenční jádro)**. Toki Pona tak ve všech třech variantách slouží jako **měřítkově invariantní mikro-laboratoř (Scale-Invariant Proxy)**, která dovoluje na stovkách kontrolovaných běhů statisticky odvodit zákony platné pro předtrénink velkých LLM.

### Přehledové srovnání tří variant

| Kritérium | Varianta 1: Čistá statistická hloubka (InSIS Baseline+) | Varianta 2: Architektonický duel (Transformer vs. Mamba-3) | Varianta 3: Fyzika učení & Data Wall (Muon vs. AdamW & Kurikulum) |
| :--- | :--- | :--- | :--- |
| **Hlavní téma** | Hluboká statistická analýza škálování, tokenizace, křivek učení a fázových přechodů syntaxe | Srovnání globální pozornosti (Attention) a rekurzivní komprese stavu (SSM) na malých datech | Boj s vyčerpáním dat (*Data Wall*): spektrální optimalizace (Muon) a kurikulární řazení dat |
| **Architektura** | 1× (Decoder-Only Transformer) | 2× (Transformer vs. Mamba-3, *Iso-Parameter*) | 1× (Decoder-Only Transformer) |
| **Technická náročnost (kód)** | **Nízká** (100% čistý PyTorch, žádné cizí kernely) | **Střední–Vyšší** (integrace `mamba-ssm` / SSD a vybalancování vah) | **Nízká–Střední** (Muon má jen ~25 řádků maticového násobení v PyTorchi) |
| **Statistická hloubka (FIS VŠE)** | **Extrémní** (ANOVA, LMM, NLME křivky učení, GLMM, psychometrické IRT, Bootstrap) | **Velmi vysoká** (Faktorová ANOVA, LMM/GLMM s interakcemi $\text{Arch} \times \text{Token}$, škálovací zákony) | **Velmi vysoká** (Plochy odezvy RSM, analýza bodu obratu $R_{\text{crit}}$, LMM/GLMM, Mixture DoE) |
| **Soulad se zadáním v InSIS** | **100 % (Beze změny zadání)** | **95 % (Rozšíření o 1 srovnávací faktor)** | **95 % (Rozšíření o faktor optimalizace/dat)** |

---

## 2. Detailní specifikace jednotlivých variant (Abstrakty, Otázky, Modely)

---

### VARIANTA 1: Čistá statistická hloubka a vynoření syntaxe (InSIS Baseline+)

**Název:** *Small Language Models (SLM) na malých datech: statistická analýza výkonu, tokenizace a škálování*

#### 1.1 Základní specifika a experimentální design
* **Zaměření:** 100% soustředění na statistické modelování bez tříštění sil na vedlejší architektury. Zkoumá se jedna referenční architektura (**Decoder-Only Transformer**) v plném faktorovém experimentu:
  * **Faktor A – Kapacita modelu ($N$):** 4 úrovně pokrývající téměř 2 řády velikosti (`Micro` ~120k, `Mini` ~550k, `Small` ~2,5M, `Medium` ~8,0M parametrů).
  * **Faktor B – Granularita tokenizace ($T$):** 3 úrovně (`CHAR`, podslovní `BPE`, celoslovní `WORD`).
  * **Faktor C – Objem unikátních dat ($D$):** 4 frakce korpusu (10 %, 25 %, 50 %, 100 %).
  * **Faktor D – Více-epochové opakování ($E$):** Průběžné měření v 1., 4., 16. a 64. epoše (křivky učení).
  * **Replikace:** 3 náhodné inicializační seedy (`42, 1337, 2026`) $\Rightarrow 4 \times 3 \times 4 \times 3 = \mathbf{144 \text{ běhů}}$ ($\times 4$ epochové checkpointy $= 576$ vyhodnocených modelových stavů).

#### 1.2 Abstrakt (CZ / EN)
> **Abstrakt (CZ):** Umělý minimalistický jazyk toki pona, charakterizovaný pravidelnou analytickou syntaxí a uzavřeným jádrovým slovníkem o přibližně 137 základních slovech, slouží v této diplomové práci jako měřítkově invariantní mikro-laboratoř (scale-invariant proxy) pro výzkum malých jazykových modelů (SLM) a zobecnění jejich vlastností pro velké modely (LLM). Minimalistický lexikon eliminuje úzké hrdlo matice vnoření (embedding bottleneck), které u přirozených jazyků pohlcuje většinu parametrů malých sítí, a umožňuje alokovat přes 98 % kapacity přímo do výpočetních vrstev Transformeru. Cílem práce je statisticky popsat a kvantifikovat, jak objem trénovacích dat, počet více-epochových opakování (data-constrained regime), počet parametrů a granularita tokenizace (znaková, podslovní BPE a celoslovní) ovlivňují výkon, datovou efektivitu, dynamiku přefitování a škálování SLM. Nad deduplikovaným korpusem rozděleným stratifikovaně podle čtyř žánrových zdrojů je realizován opakovaný plný faktorový experiment napříč více náhodnými inicializacemi při srovnatelném výpočetním rozpočtu. Výkon modelů trénovaných od začátku je hodnocen v průběhu celých křivek učení pomocí křížové entropie normalizované na znaky (Bits-Per-Character, BPC), slovně normalizované perplexity, doménové generalizace a automaticky generovaných testů gramatické přijatelnosti (447 minimálních párů BLiMP-Toki Pona) a kompozičního zobecnění. Pomocí vícefaktorové analýzy rozptylu (ANOVA), lineárních a zobecněných lineárních smíšených modelů (LMM, GLMM), teorie odpovědi na položku (IRT) a nelineární regrese s bootstrapovými intervaly spolehlivosti práce odhaduje mocninné škálovací zákony pro opakovaná data, modeluje nelineární fázový přechod mezi křížovou entropií a osvojením syntaxe, dekomponuje rozptyl náhodných semínek a identifikuje optimální konfigurace pro učení z malých dat.
>
> **Abstract (EN):** The artificial minimalist language Toki Pona, characterized by regular analytical syntax and a closed core lexicon of approximately 137 root words, serves in this master's thesis as a scale-invariant proxy environment for investigating Small Language Models (SLMs) and abstracting their scaling properties to Large Language Models (LLMs). Its minimal vocabulary eliminates the embedding bottleneck that typically consumes most parameters in small networks trained on natural languages, allocating over 98% of parameter capacity directly to the Transformer layers. The objective of this study is to statistically model and quantify how training data volume, multi-epoch repetition in a data-constrained regime, parameter count, and tokenization granularity (character-level, subword BPE, and word-level) govern generative performance, data efficiency, overfitting dynamics, and scaling behavior. Using a deduplicated corpus stratified across four textual genres, we conduct a replicated full-factorial experiment across multiple random initialization seeds under a controlled compute budget. Models trained from scratch are evaluated along their full learning trajectories using character-normalized cross-entropy (Bits-Per-Character, BPC), word-normalized perplexity, domain generalization gaps, and automated benchmarks of grammatical acceptability (447 BLiMP-Toki Pona minimal pairs) and compositional generalization. Employing multi-factor ANOVA, linear and generalized linear mixed-effects models (LMM and GLMM), Item Response Theory (IRT), and nonlinear regression with bootstrap confidence intervals, we estimate data-constrained power-law scaling laws, model the nonlinear phase transition from cross-entropy loss to syntactic competence, quantify seed variance, and identify optimal SLM configurations for low-resource learning.

#### 1.3 Výzkumné otázky (VO)
* **VO1 (Interakce tokenizace a měřítka):** Jak závisí škálovací exponenty modelu ($\alpha_T$) a dat ($\beta_T$) na zvolené granularitě tokenizace $T \in \{\text{CHAR}, \text{BPE}, \text{WORD}\}$ a při jakém kritickém objemu dat a parametrů dochází k bodu překřížení (*crossover point*), kdy znaková/podslovní reprezentace překoná celoslovní tokenizaci?
* **VO2 (Škálování v datové tísni a přefitování):** Jak klesá efektivní informační přínos opakovaných dat ($D_{\text{eff}}$) při více-epochovém tréninku (1 až 64 epoch) a jak tokenizace ovlivňuje polohu kritického bodu přefitování ($R_{\text{crit}}$) na validační U-křivce?
* **VO3 (Fázový přechod k syntaxi a IRT):** Je vztah mezi spojitým poklesem křížové entropie (BPC) a osvojením diskrétních syntaktických pravidel (BLiMP) plynulý, nebo vykazuje nelineární fázový přechod (*emergence threshold* $\text{BPC}^*$) závislý na psychometrické obtížnosti daného gramatického jevu (*li, e, pi, la*)?
* **VO4 (Dekompozice nejistoty):** Jaký podíl celkového rozptylu BPC připadá na systematické faktory (kapacita, data, tokenizace) v porovnání s náhodnou variabilitou inicializačních semínek ($u_{\text{seed}}$) a mezidoménovým přenosem ($\Delta\text{BPC}_{\text{domain}}$)?

---

### VARIANTA 2: Architektonický duel (Transformer vs. Mamba-3 v datové tísni)

**Název:** *Small Language Models (SLM) na malých datech: statistická analýza výkonu a škálování architektur Transformer a Mamba*

#### 2.1 Základní specifika a experimentální design
* **Zaměření:** Rozšiřuje původní zadání o přímou konfrontaci dvou základních matematických principů zpracování sekvencí při striktní podmínce **Iso-Parameter ($N \pm 1\,\%$)** a shodném počtu vrstev:
  1. **Causal Transformer:** Globální paměť $\mathcal{O}(L^2)$ s přímým adresováním všech předchozích tokenů.
  2. **Mamba-3 (Selective SSM):** Rekurzivní komprese historie do konečného stavu $\mathcal{O}(1)$ pomocí exponenciálně-lichoběžníkové diskretizace a komplexních fázových rotací.
* **Faktorová mřížka:** $2 \text{ (Architektury: Transformer vs. Mamba-3)} \times 3 \text{ (Tokenizace: CHAR, BPE, WORD)} \times 4 \text{ (Velikosti: Micro, Mini, Small, Medium)} \times 4 \text{ (Frakce dat: 10, 25, 50, 100 \%)} \times 3 \text{ (Seedy)} = \mathbf{288 \text{ běhů}}$ (s vyhodnocením v 1., 4., 16. a 64. epoše).

#### 2.2 Abstrakt (CZ / EN)
> **Abstrakt (CZ):** Při trénování malých jazykových modelů (SLM) na přirozených jazycích pohlcuje matice vnoření (embedding bottleneck) kvůli rozsáhlým slovníkům většinu parametrického rozpočtu sítě, což znemožňuje čisté srovnání výpočetních architektur. Tato diplomová práce využívá umělý minimalistický jazyk toki pona s uzavřeným lexikonem přibližně 137 základních slov a pravidelnou analytickou syntaxí jako měřítkově invariantní mikro-laboratoř (scale-invariant proxy), v níž přes 98 % parametrů směřuje přímo do sekvenčního jádra sítě – tedy ve stejném poměru jako u velkých jazykových modelů (LLM). Cílem práce je v režimu omezených dat (data-constrained regime) statisticky popsat a porovnat, jak dva fundamentální matematické přístupy ke zpracování sekvencí – kauzální Transformer s globální pozorností a rekurzivní stavový model Mamba-3 – interagují s granularitou tokenizace (znaková, podslovní BPE a celoslovní), počtem parametrů a více-epochovým opakováním dat. Nad deduplikovaným korpusem rozděleným stratifikovaně podle čtyř žánrových zdrojů je realizován opakovaný plný faktorový experiment za přísné podmínky vyrovnaného počtu parametrů (Iso-Parameter). Výkon modelů je hodnocen pomocí křížové entropie normalizované na znaky (BPC), slovně normalizované perplexity, doménové generalizace a automaticky generovaných testů gramatické přijatelnosti (447 minimálních párů BLiMP-Toki Pona) a kompozičního zobecnění. Pomocí vícefaktorové analýzy rozptylu (ANOVA), lineárních a zobecněných lineárních smíšených modelů (LMM a GLMM) a nelineární regrese práce odhaduje architektonicky specifické škálovací zákony, testuje hypotézu rekurzivního stavu jako přirozeného regularizátoru proti přefitování ($R_{\text{crit}}$) a formuluje doporučení pro volbu sekvenční architektury.
>
> **Abstract (EN):** When training Small Language Models (SLMs) on natural languages, the embedding matrix consumes the vast majority of the parameter budget due to large vocabularies (the embedding bottleneck), obscuring clean architectural comparisons. This master's thesis leverages the artificial minimalist language Toki Pona—characterized by a closed lexicon of approximately 137 root words and regular analytical syntax—as a scale-invariant proxy environment where over 98% of parameters reside directly in the sequential core, mirroring the structural proportions of frontier LLMs. Operating in a data-constrained regime, the objective of this study is to statistically compare how two fundamental sequence modeling paradigms—the global-attention Causal Transformer and the recurrent Mamba-3 selective state space model—interact with tokenization granularity (character-level, subword BPE, and word-level), parameter count, and multi-epoch data repetition. Using a deduplicated corpus stratified across four textual genres, we conduct a replicated full-factorial experiment under strict iso-parameter constraints. Models trained from scratch are evaluated via character-normalized cross-entropy (Bits-Per-Character, BPC), word-normalized perplexity, domain generalization gaps, and automated benchmarks of grammatical acceptability (447 BLiMP-Toki Pona minimal pairs) and compositional generalization. Employing multi-factor ANOVA, linear and generalized linear mixed-effects models (LMM and GLMM), and nonlinear regression, we estimate architecture-specific scaling laws, test whether recurrent state compression acts as an implicit regularizer delaying multi-epoch overfitting ($R_{\text{crit}}$), and derive actionable insights for sequence backbone selection.

#### 2.3 Výzkumné otázky (VO)
* **VO1 (Interakce $\text{Architektura} \times \text{Tokenizace}$):** Jak se mění relativní výkonnost modelu Mamba-3 vůči Transformeru při přechodu od krátkých celoslovních sekvencí (`WORD`) k 5–7× delším znakovým sekvencím (`CHAR`), které kladou vysoké nároky na komprimační kapacitu rekurzivního stavu $h_t$?
* **VO2 (Rekurzivní hrdlo jako regularizátor v datové tísni):** Působí konečná paměť skrytého stavu Mamby-3 při více-epochovém tréninku (1 až 64 epoch) jako strukturální regularizátor, který oproti globální pozornosti Transformeru zpomaluje memorování trénovacích n-gramů a posouvá kritický bod přefitování ($R_{\text{crit}}$)?
* **VO3 (Škálovací exponenty architektur):** Liší se statisticky významně škálovací exponenty parametrů ($\alpha$) a dat ($\beta$) v Muennighoffově škálovacím zákoně mezi architekturou Transformer a Mamba-3?
* **VO4 (Syntaktická selektivita v BLiMP):** Vykazuje Mamba-3 v logistickém smíšeném modelu (GLMM) nižší poměr šancí (Odds Ratio) na správné vyhodnocení gramatických jevů vyžadujících syntaktické závorkování a kontextové oddělení (*pi*, *la*) oproti Transformeru?

---

### VARIANTA 3: Překonání datové zdi (Spektrální optimalizace Muon vs. AdamW & Kurikulum)

**Název:** *Small Language Models (SLM) v datové tísni: statistická analýza spektrální optimalizace, kurikulárního učení a více-epochového škálování*

#### 3.1 Základní specifika a experimentální design
* **Zaměření:** Zaměřuje se na vyčerpání unikátních dat (*Data Wall*) a maximalizaci datové efektivity. Architektura je zafixovaná (**Decoder-Only Transformer**), do faktorového pokusu vstupují řídicí faktory trénovacího procesu:
  * **Faktor A – Optimalizační paradigma (2 úrovně):** Standardní prvkový **AdamW** vs. maticový spektrální optimalizátor **Muon** (Newton-Schulzova iterace 5. stupně na 2D váhových maticích).
  * **Faktor B – Uspořádání dat (2 úrovně + Mixture DoE):** Náhodné míchání (*i.i.d. shuffling*) vs. **Kurikulární učení (*Syntactic Curriculum*)** podle syntaktické složitosti vět + míchání 4 žánrů korpusu na simplexu.
  * **Faktor C – Tokenizace (`CHAR`, `BPE`, `WORD`)** $\times$ **Faktor D – Kapacita ($N$) a opakování epoch (1 až 64 epoch)**.

#### 3.2 Abstrakt (CZ / EN)
> **Abstrakt (CZ):** V důsledku vyčerpání unikátních textových zdrojů naráží trénink velkých i malých jazykových modelů (LLM a SLM) na problém datové zdi (data wall), kdy je nutné opakovat omezený korpus ve více epochách, což vede k rychlému přefitování a degradaci výkonu. Tato diplomová práce využívá umělý minimalistický jazyk toki pona s uzavřeným slovníkem přibližně 137 základních slov jako měřítkově invariantní mikro-laboratoř (scale-invariant proxy), která eliminuje úzké hrdlo matice vnoření a umožňuje alokovat přes 98 % parametrů do skrytých vrstev Transformeru. Cílem práce je statisticky popsat a kvantifikovat, jak volba optimalizačního algoritmu – srovnání prvkového optimalizátoru AdamW se spektrálním maticovým optimalizátorem Muon využívajícím Newton-Schulzovu ortogonalizaci –, kurikulární řazení trénovacích vět podle syntaktické složitosti, granularita tokenizace (znaková, podslovní BPE, celoslovní) a počet parametrů ovlivňují tokenovou efektivitu, tvar křivek učení a odolnost vůči více-epochovému přefitování. Nad deduplikovaným korpusem čtyř žánrových zdrojů je realizován opakovaný faktorový experiment měřený napříč 1 až 64 epochami pomocí znakově normalizované křížové entropie (BPC), perplexity a syntaktických a kompozičních testů BLiMP-Toki Pona. Pomocí vícefaktorové analýzy rozptylu (ANOVA), nelineárních smíšených modelů (NLME), návrhu experimentů se směsmi (Mixture DoE) a regrese škálovacích zákonů práce kvantifikuje posun kritického bodu přefitování ($R_{\text{crit}}$) a formuluje statisticky podložené postupy pro maximální vytěžení omezených trénovacích dat.
>
> **Abstract (EN):** Due to the depletion of unique human text, training both Large and Small Language Models (LLMs and SLMs) increasingly faces the data wall, requiring multi-epoch repetition over finite corpora that triggers rapid memorization and performance degradation. This master's thesis utilizes the artificial minimalist language Toki Pona—featuring a closed lexicon of approximately 137 root words—as a scale-invariant proxy environment that eliminates the embedding bottleneck and allocates over 98% of parameters directly to Transformer hidden layers. The objective of this study is to statistically model and quantify how optimization geometry—comparing coordinate-wise AdamW against the spectral matrix optimizer Muon via fifth-order Newton-Schulz orthogonalization—, syntactic curriculum ordering, tokenization granularity (character-level, subword BPE, and word-level), and parameter scale govern token efficiency, learning curve trajectories, and resistance to multi-epoch overfitting. Across a deduplicated four-genre corpus, we execute a replicated factorial experiment tracked from 1 to 64 epochs, evaluating models via character-normalized cross-entropy (Bits-Per-Character, BPC), perplexity, and the BLiMP-Toki Pona syntactic and compositional benchmarks. Utilizing multi-factor ANOVA, nonlinear mixed-effects models (NLME), mixture design of experiments (Mixture DoE), and data-constrained scaling law regression, we quantify shifts in the critical overfitting threshold ($R_{\text{crit}}$) and establish statistically grounded strategies for maximizing sample efficiency in data-constrained pre-training.

#### 3.3 Výzkumné otázky (VO)
* **VO1 (Tokenová efektivita spektrálního sestupu):** Dosahuje spektrální optimalizátor Muon statisticky významně rychlejší konvergence křivky učení $\text{BPC}(t)$ než AdamW, a jaký je jeho ekvivalentní násobek úspory trénovacích tokenů?
* **VO2 (Potlačení memorování ve více epochách):** Posouvá spektrální ortogonalizace u optimalizátoru Muon kritický bod přefitování ($R_{\text{crit}}$) na validační U-křivce do vyšších počtů epoch a zvyšuje efektivní objem opakovaných dat ($D_{\text{eff}}$)?
* **VO3 (Vliv syntaktického kurikula a žánrové směsi):** Zlepšuje kurikulární řazení trénovacích dat a optimalizace žánrových vah pomocí Mixture DoE kompoziční zobecnění a *Out-of-Domain* BPC oproti náhodnému míchání (*i.i.d.*)?
* **VO4 (Interakce optimalizátoru a tokenizace):** Liší se přínos maticové ortogonalizace (Muon) mezi úlohou s krátkými sekvencemi (`WORD`) a úlohou s dlouhými sekvencemi (`CHAR`)?

---

## 3. Technická a matematická prováděcí specifikace (Pro vývojáře / AI Agenta)

Tato sekce obsahuje exaktní parametry, rovnice a pravidla, podle kterých lze bez dalšího doptávání napsat či upravit kód a spustit výpočty.

### 3.1 Specifikace korpusu a tokenizérů (`src/data/`, `src/tokenization/`)
* **Zdroje korpusu (4 žánry):**
  1. *Tatoeba* (krátké konverzační věty, *In-Domain* základ),
  2. *Wikipesija* (encyklopedické články),
  3. *Lipu Sewi* (dlouhé narativní a náboženské překlady),
  4. *Poki Lapo* (komunitní beletrie a poezie).
* **Předzpracování:** Převod na malá písmena (lowercase), normalizace interpunkce, filtrování vět obsahujících nepovolené znaky mimo abecedu Toki Pona (`a, e, i, j, k, l, m, n, o, p, s, t, u, w` + interpunkce), exaktní deduplikace na úrovni vět pomocí SHA-256.
* **Rozdělení (Split):** Stratifikované podle zdrojů v poměru **80 % Train : 10 % Validation : 10 % Test** (pevný `seed = 42`). Navíc vyčleněný *Out-of-Domain* test split pro měření $\Delta\text{BPC}_{\text{domain}}$.
* **Specifikace 3 tokenizérů:**
  * `CHAR`: Slovník všech unikátních znaků + speciální tokeny `<pad>, <bos>, <eos>, <unk>` ($|V|_{\text{CHAR}} \approx 35\text{--}40$).
  * `BPE`: Byte-Pair Encoding trénovaný výhradně na trénovací sadě se zákazem slučování přes mezery (`pre_tokenizer = Whitespace`), cílová velikost slovníku $|V|_{\text{BPE}} = 100$.
  * `WORD`: Celoslovní tokenizér rozdělující text podle bílých znaků a interpunkce, pokrývající slovník *pu/ku* a frekventovaná vlastní jména ($|V|_{\text{WORD}} \approx 140\text{--}630$ dle ořezu vlastních jmen na `<unk>`).

### 3.2 Tabulka velikostí sítí a Iso-Parameter konfigurace (`src/models/`)

Všechny modely používají `Weight Tying` ($\mathbf{W}_{\text{head}} = \mathbf{W}_{\text{emb}}^\top$), `RMSNorm` ($\epsilon = 10^{-5}$) a `bias = False`. Pro **Variantu 2** jsou rozměry Mamby-3 ($d_{\text{state}} = 16$, $\text{expand} = 2$) spárovány s Transformerem tak, aby počet ne-embeddingových parametrů $N$ souhlasil s přesností $\pm 1\,\%$:

| Preset | $n_{\text{layer}}$ | $d_{\text{model}}$ | Transformer ($n_{\text{head}}, d_{\text{ff}}$ SwiGLU) | Mamba-3 ($\text{expand}, d_{\text{state}}, R_{\text{MIMO}}$) | Počet parametrů $N$ (cca) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`micro`** | 2 | 64 | $n_{\text{head}}=2, d_{\text{ff}}=176$ | $\text{expand}=2, d_{\text{state}}=16, R=2$ | **~115k** |
| **`mini`** | 4 | 128 | $n_{\text{head}}=4, d_{\text{ff}}=352$ | $\text{expand}=2, d_{\text{state}}=16, R=2$ | **~820k** |
| **`small`** | 6 | 256 | $n_{\text{head}}=8, d_{\text{ff}}=688$ | $\text{expand}=2, d_{\text{state}}=16, R=2$ | **~4,8M** |
| **`medium`** | 8 | 384 | $n_{\text{head}}=8, d_{\text{ff}}=1024$ | $\text{expand}=2, d_{\text{state}}=16, R=2$ | **~14,2M** |

*(Pozn.: Přesná hodnota $d_{\text{ff}}$ u SwiGLU MLP vrstvy se před spuštěním automaticky doladí funkcí `match_parameter_count()`, aby platilo $|N_{\text{Trans}} - N_{\text{Mamba}}| / N_{\text{Trans}} < 0{,}01$.)*

### 3.3 Trénovací hyperparametry a rozvrh měření (`src/training/`)
* **Kontextové okno (`max_seq_len`):** `128` tokenů pro `WORD`/`BPE`, `512` tokenů pro `CHAR` (aby jedno okno vždy pokrývalo srovnatelný počet fyzických znaků textu!).
* **Efektivní velikost dávky:** `batch_size = 64` sekvencí.
* **Optimalizátor AdamW (Varianta 1, 2, 3-baseline):** $\text{lr} = 6 \cdot 10^{-4}$, $\beta_1 = 0{,}9, \beta_2 = 0{,}95$, $\text{weight\_decay} = 0{,}1$, gradient clipping $1{,}0$.
* **Optimalizátor Hybrid Muon (Varianta 3):**
  * Na 2D váhové matice skrytých vrstev ($W \in \mathbb{R}^{n \times m}$): Nesterov momentum $\mu = 0{,}95$, 5 kroků Newton-Schulzovy iterace:
    $$X_0 = \frac{M_t}{\|M_t\|_F + 10^{-7}}, \quad X_{k+1} = a X_k + b (X_k X_k^\top) X_k + c (X_k X_k^\top)^2 X_k \quad (a=3{,}4445, b=-4{,}7750, c=2{,}0315)$$
    Aktualizace s RMS škálováním dimenze (dle *Liu et al., 2025*): $W_t = W_{t-1} - \eta_{\text{Muon}} \cdot 0{,}2 \sqrt{\max(n, m)} \cdot X_5$, kde $\eta_{\text{Muon}} = 0{,}02$.
  * Na 1D parametry (RMSNorm) a embeddingy: standardní AdamW ($\text{lr} = 6 \cdot 10^{-4}$).
* **Checkpointy pro více-epochový režim:** Vyhodnocení na validační a testovací sadě probíhá přesně na konci **1., 4., 16. a 64. epochy**.

### 3.4 Matematické definice evaluačních metrik a statistických modelů (`src/evaluation/`)

1. **Bits-Per-Character (BPC) a Slovní Perplexita ($\text{PPL}_{\text{word}}$):**
   $$\text{BPC} = \frac{-\sum_{t=1}^{T_{\text{tokens}}} \ln P(x_t \mid x_{<t})}{N_{\text{chars}} \cdot \ln 2}, \qquad \text{PPL}_{\text{word}} = \exp\left( \frac{-\sum_{t=1}^{T_{\text{tokens}}} \ln P(x_t \mid x_{<t})}{N_{\text{words}}} \right)$$
2. **Muennighoffův škálovací zákon pro opakovaná data ($D_{\text{eff}}$):**
   Efektivní počet unikátních dat při $E$ epochách opakování základního korpusu $U_D$:
   $$D_{\text{eff}}(U_D, E) = U_D + U_D \cdot R^* \left( 1 - \exp\left( -\frac{E - 1}{R^*} \right) \right), \qquad \text{BPC}(N, U_D, E) = E_{\infty} + \frac{A}{N^\alpha} + \frac{B}{D_{\text{eff}}(U_D, E)^\beta}$$
   Kritický bod přefitování $R_{\text{crit}}$ je definován jako epocha $E$, kde validační BPC dosahuje minima na U-křivce: $R_{\text{crit}} = \arg\min_{E \in \{1, 4, 16, 64\}} \text{BPC}_{\text{val}}(E)$.
3. **Hierarchický lineární smíšený model (LMM v R `lme4` / Python `statsmodels`):**
   ```r
   # Varianta 1:
   lmer(BPC ~ Tokenization * log(N) * log(D_eff) + (1 | Seed) + (1 | Domain), data = df)
   # Varianta 2 (s Architekturou):
   lmer(BPC ~ Architecture * Tokenization * log(N) + Architecture * log(D_eff) + (1 | Seed), data = df)
   ```
4. **Zobecněný lineární smíšený model (GLMM) a 2PL IRT model pro BLiMP-Toki Pona (447 párů):**
   Nechť $Y_{ij} \in \{0, 1\}$ značí, zda model $i$ přiřadil vyšší pravděpodobnost gramatické větě $S_j^+$ než negramatické $S_j^-$:
   $$\text{logit}\left(P(Y_{ij} = 1)\right) = a_j (\theta_i - b_j), \qquad \text{resp. v GLMM: } \text{glmer}(Y \sim \text{Tokenizer} + \ln(N) + \text{Rule} + (1 \mid \text{Seed}) + (1 \mid \text{ItemID}), \text{family}=\text{binomial})$$
   kde $\theta_i$ je latentní syntaktická schopnost modelu $i$ (koreluje s $-\text{BPC}_i$), $b_j$ je obtížnost položky/pravidla $j$ a $a_j$ je její diskriminační strmost (rychlost fázového přechodu *emergence*).

---

## 4. Krok-za-krokem exekuční plán pro AI agenta / řešitele

Pokud na projektu začíná pracovat vývojář nebo AI agent, postupuje v těchto 6 krocích:

1. **Krok 1 (Potvrzení varianty):** Zvolit jednu ze tří variant výše (`Varianta 1`, `Varianta 2` nebo `Varianta 3`).
2. **Krok 2 (Příprava dat a testů – `src/data/`, `src/tokenization/`, `src/evaluation/`):**
   * Spustit čištění a stratifikovaný split korpusu (80/10/10).
   * Natrénovat a uložit 3 tokenizéry (`char_tok.json`, `bpe_tok.json`, `word_tok.json`).
   * Vygenerovat deterministickou testovací baterii 447 minimálních párů `BLiMP-Toki Pona` a 20+20 kompozičních párů.
3. **Krok 3 (Implementace modelů a trénovací smyčky – `src/models/`, `src/training/`):**
   * Implementovat/zkontrolovat `Decoder-Only Transformer` s presety `micro, mini, small, medium`.
   * *(Pokud Varianta 2):* Implementovat `Mamba3Model` (SSD/lichoběžníková diskretizace) s automatickým vybalancováním parametrů $N \pm 1\,\%$.
   * *(Pokud Varianta 3):* Implementovat třídu `HybridMuonAdamW` (Newton-Schulz 5 kroků) a kurikulární řazení vět v `Dataset` loaderu.
   * Nastavit ukládání vyhodnocovacích metrik přesně v epochách `1, 4, 16, 64`.
4. **Krok 4 (Spuštění faktorového experimentu – `experiments/`):**
   * Vygenerovat JSON konfiguraci faktorové mřížky a spustit dávkový trénink (`slm-run` / SLURM skript).
5. **Krok 5 (Statistická syntéza a grafy – `experiments/scripts/analyze_results.py` & `analysis_models.R`):**
   * Spočítat ANOVA tabulku ($\eta^2, \omega^2$), LMM, GLMM/IRT, nelineární regresi škálovacích zákonů s bootstrapem (1 000 replikací) a vygenerovat publikační PDF grafy do `results/figures/`.
6. **Krok 6 (Kompletace textu práce – `thesis/kapitoly/`):**
   * Vložit příslušný abstrakt do `thesis/kapitoly/zacatek.tex` a vypsat výsledky do kapitol `01` až `05`.
