# Tři finální varianty diplomové práce (DP_SLM) – Vědecká a technická specifikace

> **Garantující pracoviště:** Katedra statistiky a pravděpodobnosti (KSTP), Fakulta informatiky a statistiky VŠE v Praze  
> **Vedoucí práce:** Ing. Karel Šafr, Ph.D.  
> **Autor:** Bc. František Pavlík  
> **Účel dokumentu:** Kompletní akademické, metodické a technické zadání očištěné od terminologického balastu, slibů obtížně obhajitelných hypotéz a implementačních pastí. Slouží jako přímý podklad pro schválení zadání, psaní textu práce i exekuci experimentů.

---

## 1. Společné metodické ukotvení: Proč toki pona a co přináší pro SLM/LLM

Při trénování malých jazykových modelů na standardních přirozených jazycích vzniká zásadní metodická komplikace zvaná **úzké hrdlo parametrů vnoření (embedding bottleneck)**. Standardní slovníky s 32 000 až 128 000 tokeny vyžadují rozsáhlé matice vnoření, které u modelů s jednotkami milionů parametrů pohltí 50 % až 85 % celkového parametrického rozpočtu. Model pak netestuje sekvenční uvažování skrytých vrstev, ale kapacitu paměti pro memorování slovníku.

Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 základních slov a bezvýjimkovou analytickou syntaxí tento problém odstraňuje:
* **Soustředění kapacity do skrytých vrstev:** Při celoslovní tokenizaci tvoří matice vnoření zanedbatelnou část parametrů. Převážná většina kapacity sítě směřuje přímo do sekvenčního výpočetního jádra (Self-Attention, SSM rekurze, FFN).
* **Měřítková izomorfie vůči velkým modelům:** U velkých modelů (70B+) tvoří matice vnoření rovněž jen 1–2 % celkového počtu parametrů. Zmenšením slovníku úměrně k velikosti sítě se obnovuje strukturální poměr velkých modelů v nízkozdrojovém laboratorním prostředí.
* **Čisté testovací prostředí:** Pravidelná analytická syntax bez tvarosloví umožňuje izolovaně měřit induktivní zkreslení modelů a spolehlivě testovat osvojení gramatických vazeb pomocí syntetických minimálních párů bez rušivého lexikálního šumu.

---

## 2. Přehledové srovnání tří variant

| Kritérium | Varianta 1: Čistá statistická hloubka (Transformer) | Varianta 2: Architektonický duel (Transformer vs. Mamba) | Varianta 3: Optimalizace a dynamika tréninku (Muon vs. AdamW) |
| :--- | :--- | :--- | :--- |
| **Zaměření** | Statistické modelování škálování, tokenizace, křivek učení a osvojení syntaxe | Srovnání globální pozornosti a rekurzivní stavové komprese při shodném počtu parametrů | Vliv spektrální maticové optimalizace na tokenovou efektivitu a oddálení přeučení |
| **Architektura** | 1× Decoder-Only Transformer | 2× Transformer vs. Mamba (vyrovnaný počet parametrů) | 1× Decoder-Only Transformer |
| **Zkoumané osy** | Kapacita $\times$ Tokenizace $\times$ Objem dat $\times$ Opakování epoch | Architektura $\times$ Tokenizace $\times$ Kapacita $\times$ Opakování epoch | Optimalizátor $\times$ Tokenizace $\times$ Kapacita $\times$ Opakování epoch |
| **Statistický aparát** | ANOVA, LMM, GLMM (logistické modely testů), nelineární regrese | Faktorová ANOVA, LMM, GLMM (interakce s architekturou), nelineární regrese | ANOVA, LMM pro křivky učení, detekce minima validační ztráty, nelineární regrese |
| **Implementační náročnost** | Nízká (čistý PyTorch, stabilní kód) | Střední (spolehlivá Mamba-1/2, párování počtu parametrů) | Nízká až střední (implementace Newton-Schulzovy ortogonalizace) |
| **Soulad se schváleným InSIS** | 100 % (plně odpovídá původnímu znění) | 95 % (rozšíření o srovnávací architekturu) | 95 % (rozšíření o srovnávací optimalizátor) |

---

## 3. Detailní specifikace variant (Abstrakty, výzkumné otázky a shrnutí)

---

### VARIANTA 1: Čistá statistická hloubka (Transformer)

#### 3.1 Abstrakt (CZ / EN)
> **Abstrakt (CZ):**  
> Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí slouží v této diplomové práci jako kontrolované prostředí pro výzkum malých jazykových modelů. Malý rozsah slovníku odstraňuje nepoměr v počtu parametrů vnoření a umožňuje soustředit kapacitu sítě do jejích skrytých vrstev, což odpovídá strukturálním proporcím velkých modelů. Cílem práce je statisticky popsat a kvantifikovat, jak objem dat, opakování dat ve více epochách, velikost modelu a úroveň tokenizace ovlivňují ztrátu, datovou efektivitu, dynamiku přeučení a škálování. Nad deduplikovaným korpusem rozděleným podle zdrojů je proveden faktorový experiment s několika velikostmi modelů architektury Transformer, znakovou, podslovní a slovní tokenizací a více náhodnými inicializacemi. Výkon je v průběhu tréninku hodnocen křížovou entropií normalizovanou na znaky, slovní perplexitou, doménovou generalizací a automaticky generovanými testy gramatické přijatelnosti a kompozice. Pomocí analýzy rozptylu, lineárních smíšených modelů, zobecněných lineárních smíšených modelů a nelineární regrese práce odhaduje parametry škálovacích zákonů, určuje kritický bod přeučení při opakování dat a modeluje vztah mezi klesající křížovou entropií a úspěšností v testech gramatiky.
>
> **Abstract (EN):**  
> The minimalist language Toki Pona, characterized by a closed vocabulary of approximately 137 words and regular analytical syntax, serves in this thesis as a controlled environment for researching small language models. The restricted lexicon eliminates parameter disproportion in the embedding layer and concentrates model capacity within the hidden layers, reflecting the structural balance of large language models. The objective of this work is to statistically describe and quantify how data volume, multi-epoch data repetition, model size, and tokenization granularity affect loss, data efficiency, overfitting dynamics, and scaling. Using a deduplicated corpus stratified by source, a factorial experiment is conducted across multiple Transformer model sizes, character, subword, and word tokenization, and multiple random seeds. Model performance is tracked throughout training using character-normalized cross-entropy, word perplexity, domain generalization, and automatically generated tests of grammatical acceptability and composition. Using analysis of variance, linear mixed models, generalized linear mixed models, and non-linear regression, the thesis estimates scaling law parameters, identifies the critical point of overfitting under data repetition, and models the relationship between decreasing cross-entropy and grammar test accuracy.

#### 3.2 Výzkumné otázky
* **VO1 (Interakce tokenizace a měřítka):** Jak závisí odhadnuté parametry mocninných škálovacích zákonů na zvolené granularitě tokenizace (znaková, podslovní, celoslovní) a při jakém objemu parametrů a dat dochází k bodu překřížení, kdy podslovní či znaková reprezentace překoná celoslovní model?
* **VO2 (Opakování dat a bod přeučení):** Jak se projevuje klesající mezní výnos opakovaného tréninku na stejném textu a jak tokenizační strategie ovlivňuje polohu minima validační ztráty (kritického bodu přeučení $R_{\text{crit}}$)?
* **VO3 (Vztah spojité ztráty a diskrétní gramatiky):** Jaká je funkční závislost mezi klesající křížovou entropií (BPC) a pravděpodobností úspěchu v syntaktických minimálních párech, a liší se tato dynamika mezi jednotlivými gramatickými pravidly?
* **VO4 (Rozklad rozptylu a vliv inicializace):** Jaký podíl celkového rozptylu validační ztráty je vysvětlen systematickými faktory (kapacita, data, tokenizace) v porovnání s náhodnou variabilitou danou inicializačními semínky?

#### 3.3 Shrnutí varianty
Varianta s nejnižším technickým rizikem a stoprocentním souladem se schváleným tématem v InSISu. Veškeré úsilí směřuje do rigorózního statistického vyhodnocení (ANOVA s velikostmi účinku, LMM pro rozklad rozptylu, GLMM pro položkové testy gramatiky a nelineární regrese mocninných křivek).

---

### VARIANTA 2: Architektonický duel (Transformer vs. Mamba)

#### 3.1 Abstrakt (CZ / EN)
> **Abstrakt (CZ):**  
> Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí slouží v této diplomové práci jako kontrolované prostředí pro výzkum malých jazykových modelů. Malý rozsah slovníku eliminuje zkreslení dané velikostí matice vnoření a umožňuje zkoumat vlastnosti sekvenčního jádra sítě. Cílem práce je v podmínkách omezeného objemu dat statisticky porovnat dvě odlišná sekvenční paradigmata, kauzální Transformer s globální pozorností a rekurzivní model stavového prostoru Mamba, a vyhodnotit jejich interakci s tokenizací, kapacitou sítě a opakováním dat. Nad deduplikovaným korpusem rozděleným podle zdrojů je proveden faktorový experiment při striktně vyrovnaném počtu parametrů obou architektur, se znakovou, podslovní a slovní tokenizací a více náhodnými inicializacemi. Výkon je hodnocen křížovou entropií normalizovanou na znaky, slovní perplexitou, doménovou generalizací a automatickými testy gramatické přijatelnosti a kompozice. Pomocí analýzy rozptylu, lineárních a zobecněných lineárních smíšených modelů a nelineární regrese práce odhaduje architektonicky specifické škálovací zákony, testuje vliv rekurzivního stavu na odolnost vůči přeučení při opakování dat a formuluje doporučení pro volbu sekvenční architektury při práci s malými daty.
>
> **Abstract (EN):**  
> The minimalist language Toki Pona, featuring a closed vocabulary of approximately 137 words and regular analytical syntax, serves in this thesis as a controlled environment for researching small language models. The minimal lexicon eliminates parameter skew from the embedding matrix and allows direct evaluation of the sequential core. The objective of this work is to statistically compare two sequence modeling paradigms under data constraints, the causal Transformer with global attention and the recurrent Mamba state-space model, and to evaluate their interaction with tokenization, model capacity, and data repetition. Using a deduplicated corpus stratified by source, a factorial experiment is conducted under strict parameter matching between both architectures, using character, subword, and word tokenization and multiple random seeds. Performance is assessed using character-normalized cross-entropy, word perplexity, domain generalization, and automated tests of grammatical acceptability and composition. Using analysis of variance, linear and generalized linear mixed models, and non-linear regression, the thesis estimates architecture-specific scaling laws, tests the effect of the recurrent state on overfitting resistance during data repetition, and provides guidelines for architectural choice in low-resource settings.

#### 3.2 Výzkumné otázky
* **VO1 (Interakce architektury a tokenizace):** Jak ovlivňuje přechod od krátkých celoslovních sekvencí k výrazně delším znakovým sekvencím relativní výkonnost modelu Mamba vůči Transformeru při shodném počtu parametrů?
* **VO2 (Vliv rekurzivního stavu na přeučení):** Vykazuje rekurzivní model stavového prostoru s konečnou pamětí vyšší odolnost vůči memorování dat při opakovaných epochách (posun minima validační ztráty) ve srovnání s Transformerem disponujícím přímou pozorností přes celý kontext?
* **VO3 (Architektonické škálovací exponenty):** Liší se statisticky významně parametry škálovacích zákonů (exponenty pro parametry i data) mezi architekturou Transformer a modelem Mamba v režimu malých dat?
* **VO4 (Syntaktická kompetence na dlouhých vazbách):** Zjišťují logistické smíšené modely (GLMM) systematický rozdíl v úspěšnosti obou architektur u syntaktických jevů vyžadujících oddělení větného kontextu a vnořené modifikátory?

#### 3.3 Shrnutí varianty
Přináší vysoce atraktivní vědecké srovnání dvou fundamentálních architektur současného hlubokého učení (globální pozornost vs. stavová rekurze). Metodicky čisté díky striktní podmínce vyrovnání parametrů ($N \pm 1\,\%$) a jednotnému protokolu evaluace.

---

### VARIANTA 3: Optimalizace a dynamika tréninku (Muon vs. AdamW)

#### 3.1 Abstrakt (CZ / EN)
> **Abstrakt (CZ):**  
> Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí slouží v této diplomové práci jako kontrolované prostředí pro výzkum tréninku jazykových modelů v podmínkách nedostatku dat. Malý slovník soustředí kapacitu modelu do jeho výpočetních vrstev, což umožňuje zkoumat optimalizační dynamiku bez zkreslení rozsáhlými maticemi vnoření. Cílem práce je statisticky popsat a kvantifikovat, jak volba optimalizačního algoritmu, srovnání standardního prvkového algoritmu AdamW s maticovým spektrálním optimalizátorem Muon, úroveň tokenizace a kapacita modelu ovlivňují efektivitu učení, průběh validační ztráty a nástup přeučení při opakování dat. Nad deduplikovaným korpusem rozděleným podle zdrojů je proveden faktorový experiment s architekturou Transformer, sledující vývoj metrik napříč tréninkovými epochami. Výkon je hodnocen křížovou entropií normalizovanou na znaky, perplexitou a automaticky generovanými testy gramatické přijatelnosti a kompozice. Pomocí analýzy rozptylu, lineárních smíšených modelů a nelineární regrese práce kvantifikuje rozdíl v rychlosti konvergence, vyhodnocuje posun bodu přeučení a stanovuje postupy pro efektivní trénink při opakování omezeného korpusu.
>
> **Abstract (EN):**  
> The minimalist language Toki Pona, with a closed vocabulary of approximately 137 words and regular analytical syntax, serves in this thesis as a controlled environment for researching language model training under data constraints. The restricted lexicon focuses model capacity within the hidden layers, allowing the study of optimization dynamics without distortion from large embedding matrices. The objective of this work is to statistically describe and quantify how the choice of optimization algorithm, comparing the standard coordinate-wise AdamW with the spectral matrix optimizer Muon, tokenization granularity, and model capacity affect learning efficiency, validation loss trajectories, and the onset of overfitting under data repetition. Using a deduplicated corpus stratified by source, a factorial experiment is conducted with the Transformer architecture, tracking performance metrics across training epochs. Performance is measured using character-normalized cross-entropy, perplexity, and automated tests of grammatical acceptability and composition. Using analysis of variance, linear mixed models, and non-linear regression, the thesis quantifies differences in convergence rates, evaluates shifts in the overfitting threshold, and determines procedures for efficient training when repeating limited corpora.

#### 3.2 Výzkumné otázky
* **VO1 (Tokenová a výpočetní efektivita):** Dosahuje spektrální optimalizátor Muon v raných fázích tréninku statisticky významně rychlejšího poklesu normalizované ztráty (BPC) na stejný počet zpracovaných tokenů v porovnání s algoritmem AdamW?
* **VO2 (Vliv ortogonalizace na memorování):** Posouvá spektrální vyrovnání singulárních hodnot u optimalizátoru Muon minimum validační ztráty do vyšších epoch tréninku, a oddaluje tak nástup přeučení při opakování dat?
* **VO3 (Interakce optimalizátoru s tokenizací):** Liší se přínos maticového optimalizátoru mezi krátkými sekvencemi celých slov a dlouhými sekvencemi znaků?
* **VO4 (Stabilita napříč kapacitou sítě):** Zůstává relativní výhoda spektrálního optimalizátoru stabilní při škálování kapacity modelu od stovek tisíc po jednotky milionů parametrů?

#### 3.3 Shrnutí varianty
Soustředí se na geometrii optimalizace a chování moderních spektrálních metod při tréninku z malých dat. Text byl očištěn od nesouvisejících témat (kurikulární řazení a návrhy směsí byly vypuštěny) a nezavazuje se k pevnému číslu epoch.

---

## 4. Technická a metodická pravidla pro realizaci kódu a výpočtů

Ať je vybrána kterákoli z uvedených variant, pro zajištění vědecké validity musí kód a experimentální protokol dodržovat následující čtyři zásady:

### 4.1 Vyvážení výpočetního kroku mezi tokenizacemi (Iso-Character Update)
* Znaková tokenizace produkuje v průměru 5× delší sekvence než celoslovní tokenizace.
* Pokud by byl nastaven shodný počet sekvencí v dávce (`batch_size = 32`) a shodný počet kroků optimalizátoru, znakový model by v jednom kroku viděl 5× více fyzického textu než slovní model.
* **Pravidlo:** Efektivní velikost dávky musí být normalizována na počet fyzických znaků textu na krok:
  $$\text{batch\_size}_{\text{CHAR}} \times L_{\text{CHAR}} \approx \text{batch\_size}_{\text{BPE}} \times L_{\text{BPE}} \approx \text{batch\_size}_{\text{WORD}} \times L_{\text{WORD}} \approx \text{konstanta}$$
  Například při $L_{\text{WORD}} = 64$ a $\text{batch} = 64$ ($4\,096$ slov $\approx 18\,000$ znaků) musí mít znakový model $L_{\text{CHAR}} = 256$ a $\text{batch} = 72$ ($18\,432$ znaků).

### 4.2 Korektní výpočet BPC bez zkreslení neznámými tokeny
* U celoslovní tokenizace (`WORD`) nesmí být vlastní jména a nestandardní tvary nahrazovány jediným tokenem `<unk>` způsobem, který by uměle snížil ztrátu (ztráta predikce jednoho `<unk>` by byla započtena za celé vícepísmenné jméno).
* **Pravidlo:** Pokud slovní tokenizér narazí na mimovariabilní slovo, toto slovo se budto zahrne do uzavřeného slovníku jako samostatný token, nebo se při měření BPC penalizuje plnou délkou jeho znaků. Předzpracování textu v korpusu navíc zachovává velká písmena u vlastních jmen jako regulérní součást tokenizace.

### 4.3 Průběžné logování validační ztráty pro detekci $R_{\text{crit}}$
* V abstraktu se slibuje určení „kritického bodu přeučení při opakování dat“.
* **Pravidlo:** Validační ztráta se nesmí měřit pouze ve 4 řídkých bodech (1, 4, 16, 64), ale musí se zaznamenávat **na konci každé jednotlivé epochy**.
* Kritický bod přeučení $R_{\text{crit}}$ je definován jako epocha, ve které validační BPC dosahuje globálního minima na vyhlazené křivce validační ztráty:
  $$R_{\text{crit}} = \arg\min_{e \in \{1, \dots, E_{\max}\}} \text{BPC}_{\text{val}}(e)$$

### 4.4 Korektní zápis statistických modelů
* **Opakovaná měření v čase (LMM):** Každý trénovací běh má svůj unikátní identifikátor `Run_ID`. Měření v čase (napříč epochami) představují vnitřní závislost. Model v R (`lme4`) musí obsahovat náhodný efekt běhu:
  ```r
  lmer(BPC ~ Tokenizer * log(Params) + Epoch + (1 | Run_ID) + (1 | Seed), data = results)
  ```
* **Položková analýza gramatiky (GLMM):** Výsledky testů minimálních párů (úspěch 0/1) se modelují logistickým smíšeným modelem s náhodným efektem konkrétní větné dvojice (`Item_ID`):
  ```r
  glmer(Correct ~ Architecture + Tokenizer + Rule + (1 | Item_ID) + (1 | Seed), 
        family = binomial, data = blimp_results)
  ```
* **Nelineární regrese mocninných zákonů:** Parametry $E, A, B, \alpha, \beta$ se odhadují nelineární metodou nejmenších čtverců (funkce `nls()` v R nebo `scipy.optimize.curve_fit` v Pythonu), nikoli pouhou lineární regresí logaritmů, která deformuje reziduální rozptyl u součtového modelu.
