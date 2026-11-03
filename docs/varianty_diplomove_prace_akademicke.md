# Tři varianty diplomové práce (DP_SLM) – Vědecká a technická specifikace

> **Garantující pracoviště:** Katedra statistiky a pravděpodobnosti (KSTP), Fakulta informatiky a statistiky VŠE v Praze  
> **Vedoucí práce:** Ing. Karel Šafr, Ph.D.  
> **Autor:** Bc. František Pavlík  
> **Účel dokumentu:** Akademická, metodická a technická specifikace variant diplomové práce. Slouží jako podklad pro konzultaci zadání s vedoucím, psaní textu práce a realizaci experimentů.

> [!NOTE]
> Hodnoty označené jako **(předběžné)** jsou výchozí volby, které se ověří nebo upraví v pilotní fázi (viz 4.8). Nejde o výsledky ani o závazné parametry.

---

## 1. Společné metodické ukotvení: Proč toki pona

Při trénování malých jazykových modelů (řádově $10^5$ až $10^7$ parametrů) na přirozených jazycích vzniká **úzké hrdlo parametrů vnoření (embedding bottleneck)**. Podslovní slovníky s 32 000 až 128 000 tokeny vyžadují matice vnoření o velikosti $V \times d_{\text{model}}$ (vstup a případně netied výstupní hlava), které u modelů s jednotkami milionů parametrů tvoří většinu, nebo dokonce překračují celý parametrický rozpočet. Velká část kapacity modelu pak připadá na reprezentaci lexikonu, nikoli na sekvenční výpočet ve skrytých vrstvách.

Toki pona má jádrovou slovní zásobu zhruba 120 slov (*pu*), resp. přibližně 137 slov (*ku suli*), a pravidelnou analytickou syntax bez flexe. To přináší:

* **Řádově nižší podíl parametrů vnoření.** Při celoslovní tokenizaci ($V \approx 140$ včetně speciálních tokenů) se podíl vnoření mění s velikostí modelu:

  | Konfigurace (předběžné) | $d_{\text{model}}$ | Vnoření, $V=140$, netied ($2Vd$) | Podíl při $V=140$ | Vnoření, $V=32\,000$ |
  | :--- | ---: | ---: | ---: | ---: |
  | $N \approx 10^5$ | 64 | ≈ 18 tis. | ≈ 15–18 % | ≈ 4,1 mil. (> celý model) |
  | $N \approx 10^6$ | 128 | ≈ 36 tis. | ≈ 3–4 % | ≈ 8,2 mil. (> celý model) |
  | $N \approx 10^7$ | 384 | ≈ 108 tis. | ≈ 1 % | ≈ 24,6 mil. (> celý model) |

  Teprve u největších konfigurací se podíl blíží proporcím velkých modelů (u modelů 70B+ jde o jednotky procent nebo méně). Pro nejmenší modely je podíl stále nezanedbatelný, a proto se v analýzách uvádí **počet parametrů bez vnoření** ($N_{\text{non-emb}}$) i celkový počet.
* **Srovnatelnost tokenizací v jednom jazyce.** Malý slovník umožňuje v jednom přirozeném jazyce porovnat znakovou, podslovní (BPE) i celoslovní tokenizaci bez obřích slovníků. To není u angličtiny ani u syntetických korpusů typu TinyStories (Eldan & Li, 2023) prakticky možné.
* **Testovatelnost syntaxe.** Pravidelná syntax s gramatickými částicemi (*li*, *e*, *pi*, *la*) umožňuje konstruovat syntaktické minimální páry po vzoru BLiMP (Warstadt et al., 2020). Jazyk má i víceznačnosti (kompozita, vynechání *li* po *mi*/*sina*, variabilita mezi mluvčími), takže gramatičnost párů je nutné ověřit (viz 4.7).
* **Limity a vnější validita.** Toki pona je kontrolované laboratorní prostředí, nikoli reprezentace přirozených jazyků obecně. Závěry platí pro nízkozdrojový režim a malé modely. Jejich přenos na morfologicky bohaté jazyky nebo velké modely je otázkou dalšího výzkumu.

---

## 2. Přehledové srovnání tří variant

| Kritérium | Varianta 1: Statistická hloubka (Transformer) | Varianta 2: Transformer vs. Mamba | Varianta 3: Hybrid Muon+AdamW vs. AdamW |
| :--- | :--- | :--- | :--- |
| **Zaměření** | Škálování při opakování dat, vliv tokenizace, dynamika přeučení, osvojení syntaxe | Srovnání globální pozornosti a rekurentního stavu při vyrovnaném počtu parametrů bez vnoření | Vliv maticové ortogonalizace aktualizací na rychlost konvergence a přeučení |
| **Architektura** | Kauzální Decoder-Only Transformer | Kauzální Transformer vs. selektivní stavový model Mamba | Kauzální Decoder-Only Transformer |
| **Zkoumané osy** | $N$ × $D$ × Tokenizace × Epochy × Semínka | Architektura × $N$ × $D$ × Tokenizace × Epochy × Semínka | Optimalizátor × $N$ × Tokenizace × Epochy × Semínka |
| **Statistický aparát** | LMM s AR(1), GLMM pro minimální páry, nelineární regrese škálovacích zákonů, ANOVA koncových metrik | LMM (Architektura × Epocha), GLMM (Architektura × Pravidlo), nelineární regrese s grupováním podle architektury, ANOVA | LMM trajektorií, GLMM, analýza času do prahu ztráty, ANOVA |
| **Implementační náročnost** | Nízká až střední | Střední až vyšší (CUDA jádra `mamba-ssm`, párování parametrů, měření FLOPs) | Střední (Newtonova–Schulzova iterace, oddělené rychlosti učení pro 2D a ostatní parametry) |
| **Vztah k zadání v InSIS** | Odpovídá schválenému znění | Rozšíření o srovnávací architekturu | Rozšíření o srovnávací optimalizátor |

---

## 3. Detailní specifikace variant

---

### VARIANTA 1: Statistická hloubka (Transformer)

#### 3.1 Abstrakt (Původní vs. aktualizovaná verze)
> **Původní verze:**  
> Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí slouží v této diplomové práci jako kontrolované prostředí pro výzkum malých jazykových modelů. Malý rozsah slovníku odstraňuje nepoměr v počtu parametrů vnoření a umožňuje soustředit kapacitu sítě do jejích skrytých vrstev, což odpovídá strukturálním proporcím velkých modelů. Cílem práce je statisticky popsat a kvantifikovat, jak objem dat, opakování dat ve více epochách, velikost modelu a úroveň tokenizace ovlivňují ztrátu, datovou efektivitu, dynamiku přeučení a škálování. Nad deduplikovaným korpusem rozděleným podle zdrojů je proveden faktorový experiment s několika velikostmi modelů architektury Transformer, znakovou, podslovní a slovní tokenizací a více náhodnými inicializacemi. Výkon je v průběhu tréninku hodnocen křížovou entropií normalizovanou na znaky, slovní perplexitou, doménovou generalizací a automaticky generovanými testy gramatické přijatelnosti a kompozice. Pomocí analýzy rozptylu, lineárních smíšených modelů, zobecněných lineárních smíšených modelů a nelineární regrese práce odhaduje parametry škálovacích zákonů, určuje kritický bod přeučení při opakování dat a modeluje vztah mezi klesající křížovou entropií a úspěšností v testech gramatiky.
>
> **Aktualizovaná verze:**  
> Poznatky o škálovacích zákonech a opakovaném tréninku na stejných datech pocházejí převážně z velkých modelů a rozsáhlých korpusů. U malých modelů je zkoumání komplikováno tím, že velkou část parametrů spotřebují matice vnoření. Práce využívá minimalistický jazyk toki pona s jádrovou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí jako kontrolované prostředí, v němž je podíl parametrů vnoření řádově nižší. Cílem je statisticky kvantifikovat, jak objem dat, počet opakování dat, velikost modelu a granularita tokenizace ovlivňují ztrátu, nástup přeučení a osvojení syntaxe u modelů architektury Transformer. Faktorový experiment zahrnuje modely o velikosti $10^5$ až $10^7$ parametrů, znakovou, podslovní a celoslovní tokenizaci a více náhodných inicializací. Výkon se průběžně měří křížovou entropií v bitech na znak a syntaktickými minimálními páry, a to na odložených datech ze stejných i z nových zdrojů. Smíšené modely pro opakovaná měření a položkové logistické modely doplňuje nelineární regrese škálovacích zákonů pro opakovaná data. Práce odhaduje parametry těchto zákonů, polohu minima validační ztráty a vztah mezi ztrátou a úspěšností v gramatických testech.

#### 3.2 Výzkumné otázky a hypotézy
* **VO1 (Tokenizace a měřítko):** Jak závisí parametry škálovacího zákona na granularitě tokenizace a dochází v testovaném rozsahu $N$ a $D$ k překřížení křivek BPC mezi tokenizacemi?  
  * **H1 (konfirmační, nesměrová):** Škálovací exponent $\alpha$ se mezi tokenizacemi liší. Poloha případného překřížení se analyzuje explorativně a uvádí se jen v rámci testovaného rozsahu (bez extrapolace).
* **VO2 (Opakování dat a přeučení):** Jak tokenizace a velikost modelu ovlivňují počet epoch do minima validační ztráty ($R_{\text{crit}}$) a saturační konstantu opakování $R_D^*$?  
  * **H2 (konfirmační, nesměrová):** $R_{\text{crit}}$ se liší mezi tokenizacemi a závisí na $N$ (interakce Tokenizace × $N$).
* **VO3 (Ztráta a gramatika):** Jaký je vztah mezi BPC a pravděpodobností úspěchu v minimálních párech po oddělení efektu tréninkového času, a liší se mezi gramatickými pravidly?  
  * **H3 (konfirmační):** Pravděpodobnost úspěchu klesá s rostoucí BPC i po kontrole kroku tréninku a mezi běhy. Rozdíly mezi pravidly (lokální *li*/*e* vs. *pi*, *la*) se analyzují explorativně.
* **VO4 (Rozklad rozptylu):** Jaký podíl rozptylu validační ztráty vysvětlují systematické faktory oproti variabilitě mezi semínky?  
  * **Explorativní:** Odhad komponent rozptylu a ICC, včetně toho, zda se variabilita mezi semínky mění s $N$.

#### 3.3 Shrnutí varianty
Varianta s nejnižším implementačním rizikem, odpovídající schválenému zadání. Těžiště práce leží ve statistické analýze: smíšené modely křivek učení, položkové GLMM a nelineární regrese škálovacích zákonů.

---

### VARIANTA 2: Transformer vs. Mamba

#### 3.1 Abstrakt (Původní vs. aktualizovaná verze)
> **Původní verze:**  
> Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí slouží v této diplomové práci jako kontrolované prostředí pro výzkum malých jazykových modelů. Malý rozsah slovníku eliminuje zkreslení dané velikostí matice vnoření a umožňuje zkoumat vlastnosti sekvenčního jádra sítě. Cílem práce je v podmínkách omezeného objemu dat statisticky porovnat dvě odlišná sekvenční paradigmata, kauzální Transformer s globální pozorností a rekurzivní model stavového prostoru Mamba, a vyhodnotit jejich interakci s tokenizací, kapacitou sítě a opakováním dat. Nad deduplikovaným korpusem rozděleným podle zdrojů je proveden faktorový experiment při striktně vyrovnaném počtu parametrů obou architektur, se znakovou, podslovní a slovní tokenizací a více náhodnými inicializacemi. Výkon je hodnocen křížovou entropií normalizovanou na znaky, slovní perplexitou, doménovou generalizací a automatickými testy gramatické přijatelnosti a kompozice. Pomocí analýzy rozptylu, lineárních a zobecněných lineárních smíšených modelů a nelineární regrese práce odhaduje architektonicky specifické škálovací zákony, testuje vliv rekurzivního stavu na odolnost vůči přeučení při opakování dat a formuluje doporučení pro volbu sekvenční architektury při práci s malými daty.
>
> **Aktualizovaná verze:**  
> Transformery s globální pozorností a stavové modely s rekurentním stavem představují dva odlišné přístupy k modelování sekvencí. Jejich srovnání u malých modelů a omezených dat však zkresluje velký podíl parametrů vnoření. Práce využívá minimalistický jazyk toki pona s jádrovou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí jako kontrolované prostředí, v němž je tento podíl řádově nižší. Cílem je statisticky porovnat kauzální Transformer a selektivní stavový model Mamba a vyhodnotit jejich interakci s tokenizací, velikostí modelu, objemem dat a opakováním dat. Faktorový experiment páruje obě architektury podle počtu parametrů bez vnoření a zahrnuje modely o velikosti $10^5$ až $10^7$ parametrů, několik objemů dat, znakovou, podslovní a celoslovní tokenizaci a více náhodných inicializací. Výkon se měří křížovou entropií v bitech na znak, výkonem na textech z nových zdrojů a syntaktickými minimálními páry. Pomocí smíšených modelů a nelineární regrese práce odhaduje škálovací zákony obou architektur, porovnává jejich průběh přeučení a testuje rozdíly v úspěšnosti mezi gramatickými jevy. Závěry se vztahují k testovanému prostředí.

#### 3.2 Výzkumné otázky a hypotézy
* **VO1 (Architektura × délka sekvence):** Jak se mění relativní BPC a propustnost tréninku Mamby vůči Transformeru při přechodu od celoslovních ke znakovým sekvencím?  
  * **H1 (konfirmační, nesměrová):** Existuje interakce Architektura × Tokenizace v BPC. Propustnost (znaky/s) se reportuje deskriptivně.
* **VO2 (Průběh přeučení):** Liší se trajektorie validační ztráty a $R_{\text{crit}}$ mezi architekturami?  
  * **Explorativní:** Teoretický mechanismus, který by spojoval konečný rekurentní stav s odolností vůči memorování trénovacích dat, není ustálený (memorování je vlastnost parametrů, ne délky kontextové paměti). Otázka je proto formulována jako empirické srovnání.
* **VO3 (Škálovací parametry):** Liší se parametry škálovacího zákona ($\alpha$, $\beta$, $E$) mezi architekturami?  
  * **H3 (konfirmační, nesměrová):** Alespoň jeden z exponentů $\alpha$, $\beta$ se mezi architekturami liší. Test probíhá přes model se společnými a architektonicky specifickými parametry (viz 4.4).
* **VO4 (Syntaktické jevy):** Liší se úspěšnost architektur mezi gramatickými jevy podle vzdálenosti závislých prvků?  
  * **H4 (konfirmační, nesměrová):** Interakce Architektura × Pravidlo v GLMM je nenulová. Kontext v experimentu je krátký (≈ 280 znaků, viz 4.1), takže „dlouhé vazby“ znamenají vazby v rámci několika vět, ne dlouhý kontext.

#### 3.3 Shrnutí varianty
Srovnání dvou současných přístupů k modelování sekvencí. Párování podle parametrů bez vnoření (±1 %) nezaručuje shodný výpočet ani shodné vyladění, proto se navíc měří FLOPs a rychlost učení se ladí pro každou architekturu zvlášť (4.6).

---

### VARIANTA 3: Hybrid Muon+AdamW vs. AdamW

#### 3.1 Abstrakt (Původní vs. aktualizovaná verze)
> **Původní verze:**  
> Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí slouží v této diplomové práci jako kontrolované prostředí pro výzkum tréninku jazykových modelů v podmínkách nedostatku dat. Malý slovník soustředí kapacitu modelu do jeho výpočetních vrstev, což umožňuje zkoumat optimalizační dynamiku bez zkreslení rozsáhlými maticemi vnoření. Cílem práce je statisticky popsat a kvantifikovat, jak volba optimalizačního algoritmu, srovnání standardního prvkového algoritmu AdamW s maticovým spektrálním optimalizátorem Muon, úroveň tokenizace a kapacita modelu ovlivňují efektivitu učení, průběh validační ztráty a nástup přeučení při opakování dat. Nad deduplikovaným korpusem rozděleným podle zdrojů je proveden faktorový experiment s architekturou Transformer, sledující vývoj metrik napříč tréninkovými epochami. Výkon je hodnocen křížovou entropií normalizovanou na znaky, perplexitou a automaticky generovanými testy gramatické přijatelnosti a kompozice. Pomocí analýzy rozptylu, lineárních smíšených modelů a nelineární regrese práce kvantifikuje rozdíl v rychlosti konvergence, vyhodnocuje posun bodu přeučení a stanovuje postupy pro efektivní trénink při opakování omezeného korpusu.
>
> **Aktualizovaná verze:**  
> Optimalizátor Muon ortogonalizuje aktualizace dvourozměrných váhových matic Newtonovou–Schulzovou iterací. Vnoření, výstupní hlavu a jednorozměrné parametry ponechává algoritmu AdamW. Jeho chování u malých modelů trénovaných opakovaně na omezených datech zatím není dostatečně popsáno. Práce využívá minimalistický jazyk toki pona s jádrovou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí jako kontrolované prostředí. Díky malému slovníku spadá pod Muon většina parametrů modelu, u větších konfigurací přes 95 %. Cílem je statisticky porovnat hybridní konfiguraci Muon+AdamW s čistým AdamW a kvantifikovat jejich vliv na rychlost poklesu ztráty na zpracovaný znak i na jednotku výpočtu, na průběh validační ztráty a na nástup přeučení. Faktorový experiment s architekturou Transformer zahrnuje modely o velikosti $10^5$ až $10^7$ parametrů, znakovou, podslovní a celoslovní tokenizaci a více náhodných inicializací. Oba optimalizátory se ladí samostatně. Výkon se měří křížovou entropií v bitech na znak a syntaktickými minimálními páry. Pomocí smíšených modelů, položkových logistických modelů a analýzy času do dosažení prahu ztráty práce porovnává efektivitu konvergence a polohu minima validační ztráty v testovaném prostředí.

#### 3.2 Výzkumné otázky a hypotézy
* **VO1 (Efektivita konvergence):** Dosahuje Muon+AdamW zvoleného prahu validační BPC za méně zpracovaných znaků a za méně FLOPs / kratší reálný čas než AdamW?  
  * **H1 (konfirmační, jednostranná):** Muon+AdamW potřebuje k dosažení prahu méně kroků (směr podle dosavadní literatury, Jordan 2024; Liu et al. 2025). Režie Newtonovy–Schulzovy iterace se měří a reportuje. Hypotéza o její výši se nestanovuje.
* **VO2 (Přeučení):** Liší se $R_{\text{crit}}$ a průběh validační ztráty po minimu mezi optimalizátory?  
  * **Explorativní:** Muon ortogonalizuje aktualizace, ne samotné váhy. Mechanismus vlivu na přeučení proto není předem dán.
* **VO3 (Optimalizátor × tokenizace):** Liší se přínos Muonu mezi tokenizacemi při stejném počtu znaků na krok?  
  * **H3 (konfirmační, nesměrová):** Test interakce Optimalizátor × Tokenizace.
* **VO4 (Optimalizátor × velikost modelu):** Mění se relativní přínos Muonu s $N$?  
  * **H4 (konfirmační, nesměrová):** Test interakce Optimalizátor × $\log N$. Vyžaduje ladění rychlosti učení pro každou velikost, nebo µP parametrizaci (4.6).

#### 3.3 Shrnutí varianty
Zaměřuje se na optimalizační dynamiku při tréninku z omezených dat. Toki pona je pro tuto otázku vhodné, protože při malém slovníku spadá pod Muon většina parametrů a srovnání se tak méně ředí parametry optimalizovanými AdamW.

---

## 4. Technická a metodická pravidla

### 4.1 Vyrovnání objemu textu na krok a délky kontextu
V toki pona připadá na slovo v průměru zhruba 4,4 znaku včetně mezery (předběžné, ověří se na korpusu). Při stejném počtu tokenů v kontextu by tedy znakový model viděl asi 4,4× méně textu než celoslovní.

* **Stejný počet znaků na krok:** $\text{batch}_{\text{CHAR}} L_{\text{CHAR}} \approx \text{batch}_{\text{BPE}} L_{\text{BPE}} \approx \text{batch}_{\text{WORD}} L_{\text{WORD}} \approx C_{\text{step}}$.
* **Stejný úsek textu v kontextu:** $L$ se v tokenech volí tak, aby kontext pokrýval stejný počet znaků. Příklad (předběžné):

  | Tokenizace | $L$ (tokeny) | Znaky v kontextu | Batch | Znaky na krok |
  | :--- | ---: | ---: | ---: | ---: |
  | WORD | 64 | ≈ 280 (≈ 64 slov, několik vět) | 64 | ≈ 18 000 |
  | BPE | dle naměřeného poměru znaků/token | ≈ 280 | 64 | ≈ 18 000 |
  | CHAR | 280 | 280 | 64 | 17 920 |

* Když jsou délka kontextu ve znacích i batch u všech tokenizací stejné, má každý krok stejný objem textu i stejný počet sekvencí. Rychlost učení se pak mezi tokenizacemi kvůli velikosti dávky nemusí přepočítávat. Ladí se ale kvůli jiné statistice gradientu (4.6).

### 4.2 Definice BPC a ošetření slov mimo slovník
* **Normalizace textu:** Unicode NFC, zachování velkých písmen, interpunkce a mezer, jednotné konce řádků. Normalizace je společná pro všechny tokenizéry.
* **BPC:** $\text{BPC} = \frac{1}{C_{\text{raw}}}\sum_{t=1}^{T} -\log_2 P(x_t \mid x_{<t})$, kde $C_{\text{raw}}$ je počet znaků normalizovaného textu, stejný pro všechny tokenizace.
* **Slovní perplexita** je přepočet: $\text{PPL}_{\text{word}} = 2^{\text{BPC}\cdot C_{\text{raw}}/W}$. Uvádí se jen pro srovnání s literaturou, nejde o samostatnou metriku.
* **Slova mimo slovník (OOV) u celoslovní tokenizace:** vlastní jména a nestandardní slova se rozepisují po znacích (znakový fallback). Celoslovní slovník je tedy „~137 slov + znaková abeceda“. Model nikdy nedostane `<unk>`, takže BPC není uměle snížené.

### 4.3 Plán učení a detekce $R_{\text{crit}}$
Při kosinovém útlumu nastaveném na $E_{\max}$ má model v dřívějších epochách stále vysokou rychlost učení. Jeho ztráta proto neodpovídá modelu dotrénovanému na daný počet epoch (Hoffmann et al., 2022).

* **Plán učení WSD:** lineární warmup (≈ 2 % kroků, předběžné), poté konstantní rychlost učení, poté krátký útlum (≈ 10 % kroků, předběžné).
* **Útlumové větve:** z kontrolních bodů konstantní fáze v předem vybraných epochách (např. logaritmicky rozložených) se spustí krátké větve s útlumem. Validační ztráta po útlumu je hodnota pro daný počet epoch. Náklady jsou zhruba 10 % původního běhu na každou větev a započítávají se do rozpočtu (4.8). Validační ztráta v konstantní fázi se loguje v každé epoše jako doplňková křivka.
* **$R_{\text{crit}}$:** epocha minima vyhlazené křivky validační BPC z útlumových větví (vyhlazení splajnem). Pokud minimum do $E_{\max}$ nenastane, jde o pravostranně cenzorované pozorování. Ve statistických analýzách se ošetří modelem pro cenzorovaná data (analýza přežití nebo Tobitův model), nikoli dosazením $E_{\max}$.
* **Testovací metriky** se počítají pro kontrolní bod v $R_{\text{crit}}$ na testovacích sadách (4.7), ne na validační sadě, na které se bod vybíral.

### 4.4 Škálovací zákony pro opakovaná data
Tvar Chinchilla $L = E + A N^{-\alpha} + B D^{-\beta}$ předpokládá jednu epochu nad unikátními daty. Pro opakování se použije formulace Muennighoff et al. (2023):

$$D' = U_D + U_D\,R_D^*\left(1 - e^{-R_D/R_D^*}\right), \qquad R_D = \text{počet epoch} - 1$$

$$L(N, U_D, R_D) = E + \frac{A}{N^{\alpha}} + \frac{B}{(D')^{\beta}}$$

kde $U_D$ je počet unikátních znaků a $R_D^*$ je saturační konstanta opakování. Analogickou korekci pro parametry ($N'$) z původní práce zde nepoužíváme, protože $N$ se volí pro každý objem dat zvlášť.

* **Odhad:** minimalizace Huberovy ztráty mezi $\log \hat L$ a $\log L$ (postup Chinchilly) metodou L-BFGS z mřížky počátečních hodnot. Výsledné parametry se kontrolují na stabilitu vůči volbě startu.
* **Identifikovatelnost:** při rozsahu $N$ jen přes dva řády a několika úrovních $D$ jsou parametry silně korelované. Reportuje se korelační matice odhadů. Pokud fit plného modelu není stabilní, fixují se méně důležité parametry (např. $E$ podle největšího modelu).
* **Nejistota:** intervaly spolehlivosti se počítají bootstrapem přes **jednotlivé běhy** (resampling bodů křivky v rámci buněk návrhu), ne jen přes semínka. Při 3–5 semíncích by bootstrap přes semínka nedával věrohodné intervaly.
* **Srovnání skupin (V1 tokenizace, V2 architektura):** model se společnými a skupinově specifickými parametry, porovnání vnořených modelů testem věrohodnostním poměrem nebo podle AIC, doplněné bootstrapovým intervalem pro rozdíl exponentů.

### 4.5 Statistické modely

#### A. Křivky učení (LMM)
```r
library(nlme)
lme(
  fixed  = BPC ~ Tokenizer * log(Params) * ns(log(Epoch), df = 3) + log(DataSize),
  random = ~ 1 + log(Epoch) | Run_ID,
  correlation = corAR1(form = ~ Epoch | Run_ID),
  data = training_logs
)
```
* `Run_ID` označuje jeden trénovací běh (konfigurace + semínko). Semínko určuje inicializaci i pořadí dat. Samostatný náhodný efekt `Seed` se nepoužívá, protože totéž číslo semínka v různých konfiguracích není stejná jednotka.
* Interakce s `ns(log(Epoch))` odpovídá na VO2 (různý tvar křivky podle tokenizace a $N$).
* Ve V2 a V3 se `Tokenizer` doplní o `Architecture`, resp. `Optimizer`, a jejich interakce s epochou.
* **VO4 (rozklad rozptylu):** na koncové metrice (BPC v $R_{\text{crit}}$) se odhadne model `BPC ~ (1 | Config) + (1 | Config:Run_ID)`. Komponenta mezi konfiguracemi představuje systematické faktory, komponenta uvnitř konfigurace semínka. Uvádí se jejich poměr a ICC, případně zvlášť pro každé $N$.

#### B. Minimální páry (GLMM)
Odezva je 1, pokud model přiřadí gramatické větě vyšší skóre než negramatické.
Skóre věty: $\text{Score}(S) = \sum_t \log P(x_t \mid x_{<t})$ přes všechny tokeny věty. Páry jsou konstruovány tak, aby se obě věty lišily co nejméně. U párů s různou délkou (např. vynechané *li*) se jako citlivostní analýza použije i skóre normalizované počtem znaků.

```r
# V1 (VO3): vztah BPC a úspěchu, oddělení vlivu uvnitř a mezi běhy
glmer(Correct ~ Tokenizer * Rule + BPC_within + BPC_between + log(Step) +
        (1 | Rule:Item_ID) + (1 | Run_ID),
      family = binomial, data = grammar_results)

# V2
glmer(Correct ~ Architecture * Rule + Tokenizer + (1 | Rule:Item_ID) + (1 | Run_ID),
      family = binomial, data = grammar_results)

# V3
glmer(Correct ~ Optimizer * Rule + Tokenizer + (1 | Rule:Item_ID) + (1 | Run_ID),
      family = binomial, data = grammar_results)
```
`BPC_within` je odchylka BPC kontrolního bodu od průměru jeho běhu, `BPC_between` je průměr BPC běhu. Bez tohoto rozdělení by vztah BPC a gramatiky byl jen důsledkem toho, že obojí se zlepšuje s časem.

#### C. ANOVA a mnohonásobné testy
* Faktorová ANOVA se používá jen na koncové skalární metriky (BPC v $R_{\text{crit}}$, testovací BPC) s parciálním $\eta_p^2$.
* Konfirmační hypotézy každé varianty tvoří jednu rodinu s Holmovou korekcí. Explorativní analýzy se reportují bez korekce a s výslovným označením.

#### D. V3: čas do prahu
Počet znaků (resp. FLOPs) do dosažení zvoleného prahu validační BPC se modeluje jako čas do události (Coxův model nebo AFT model) s cenzurou pro běhy, které prahu nedosáhnou.

### 4.6 Ladění hyperparametrů a férové srovnání
* **Rychlost učení:** ladí se samostatně pro každou architekturu (V2) a každý optimalizátor (V3) a pro každou tokenizaci. Ladění probíhá na pilotní mřížce **pro nejmenší, střední a největší $N$**, mezi nimi se interpoluje. Alternativou je µP parametrizace, která umožňuje přenos rychlosti učení mezi šířkami. Volba se zdokumentuje.
* **Výchozí mřížky (předběžné):** AdamW $\eta \in \{3\text{e-}4, 1\text{e-}3, 3\text{e-}3\}$. Muon (2D matice) $\eta \in \{0{,}01, 0{,}02, 0{,}05\}$ spolu s AdamW pro ostatní parametry. Mřížka se rozšíří, pokud optimum vyjde na jejím okraji.
* **Regularizace:** weight decay a dropout jsou u všech porovnávaných skupin stejné a fixní, protože obojí přímo ovlivňuje polohu $R_{\text{crit}}$. Výchozí volba je dropout 0, aby se přeučení projevilo v čisté podobě a nebylo maskováno regularizací (předběžné). Weight decay se pro Muon a AdamW parametrizuje tak, aby byl efektivně srovnatelný. Postup se uvede.
* **V2 párování:** počet parametrů bez vnoření ±1 %. Měří se a reportují FLOPs na znak a propustnost. Kvůli výkonu malých `d_model` se ověří, zda se použijí CUDA jádra `mamba-ssm`, nebo referenční implementace scanu.

### 4.7 Data, rozdělení a minimální páry
* **Korpus:** v práci se uvedou zdroje, licence, celkový rozsah (počet znaků a slov), podíl slov mimo *ku suli* a ošetření dialektových variant. Rozsah korpusu přímo omezuje maximální $D$ a tím i rozsah škálování.
* **Deduplikace:** exaktní a přibližná (např. MinHash, práh předběžně 0,8). Zvlášť se ohlídají vícenásobné překlady téhož textu (mohou přenášet obsah mezi sadami).
* **Rozdělení:**
  1. trénovací sada,
  2. validační sada ze stejných zdrojů (výběr $R_{\text{crit}}$, ladění),
  3. **testovací sada ze stejných zdrojů** (hlavní testovací metrika),
  4. **testovací sada z odložených zdrojů** (generalizace na nové zdroje).
  
  Poměry se určí podle velikosti korpusu.
* **Tokenizér BPE** se trénuje jen na trénovací sadě. Velikost slovníku BPE se zvolí pod ~137, aby šlo skutečně o podslovní jednotky. Případná varianta nad úrovní slov se uvede zvlášť.
* **Minimální páry:** seznam jevů (např. *li*, *e*, *pi*, *la*, předložky, pořadí modifikátorů), cílově desítky až stovky párů na jev (předběžné), generování podle šablon z lexikonu. Gramatičnost ověří alespoň jeden zkušený mluvčí, v ideálním případě dva s výpočtem shody. Páry se publikují spolu s prací.
* **Baselines:** n-gramový model s Kneser–Neyho vyhlazením (na znacích i slovech) a unigramový model. Uvádějí se v BPC kvůli kontextu výsledků.

### 4.8 Pilot, rozpočet a reprodukovatelnost
* **Pilotní fáze:** ověření poměru znaků na token, mřížky rychlostí učení, $E_{\max}$ (musí přesáhnout typické $R_{\text{crit}}$) a doby běhu.
* **Rozpočet:** v práci se uvede tabulka úrovní faktorů, počet buněk × semínek = počet běhů, počet útlumových větví a odhad GPU hodin. Pokud rozpočet nestačí, redukuje se plný faktorový návrh (např. menší počet úrovní $D$ u některých $N$). Neúplný návrh se pak výslovně zohlední v modelech.
* **Počet semínek:** minimálně 3, cílově 5 (předběžné). Dostatečnost se posoudí podle variability v pilotu.
* **Reprodukovatelnost:** fixní semínka, verze knihoven, konfigurační soubory každého běhu, zveřejnění kódu, logů a minimálních párů.

---

## 5. Klíčová literatura

1. Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models.* arXiv:2203.15556.
2. Muennighoff, N., et al. (2023). *Scaling Data-Constrained Language Models.* NeurIPS 2023. arXiv:2305.16264.
3. Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361.
4. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
5. Jordan, K., et al. (2024). *Muon: An optimizer for hidden layers in neural networks.* (blog / technická zpráva).
6. Liu, J., et al. (2025). *Muon is Scalable for LLM Training.* arXiv:2502.16982.
7. Warstadt, A., et al. (2020). *BLiMP: The Benchmark of Linguistic Minimal Pairs for English.* TACL 8, 377–392.
8. Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.
9. Yang, G., et al. (2022). *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (µP).* arXiv:2203.03466.
10. Hägele, A., et al. (2024). *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (WSD).* arXiv:2405.18392.
