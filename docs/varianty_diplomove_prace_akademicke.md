# Tři finální varianty diplomové práce (DP_SLM) – Vědecká a technická specifikace

> **Garantující pracoviště:** Katedra statistiky a pravděpodobnosti (KSTP), Fakulta informatiky a statistiky VŠE v Praze  
> **Vedoucí práce:** Ing. Karel Šafr, Ph.D.  
> **Autor:** Bc. František Pavlík  
> **Účel dokumentu:** Formální akademická, metodická a technická specifikace variant diplomové práce. Slouží jako přímý podklad pro schválení zadání v InSIS, zpracování textu práce i rigorózní exekuci a reprodukovatelnost experimentů.

---

## 1. Společné metodické ukotvení: Proč toki pona a co přináší pro SLM/LLM

Při trénování malých jazykových modelů (SLM, řádově $10^5$ až $10^7$ parametrů) na standardních přirozených jazycích vzniká zásadní metodická komplikace zvaná **úzké hrdlo parametrů vnoření (embedding bottleneck)**. Standardní podslovní slovníky s 32 000 až 128 000 tokeny vyžadují rozsáhlé matice vnoření $V \times d_{\text{model}}$ (vstupní vnoření a výstupní klasifikační hlava), které u modelů s jednotkami milionů parametrů pohltí 50 % až 85 % celkového parametrického rozpočtu. Model pak netestuje sekvenční uvažování a komputaci skrytých vrstev, nýbrž paměťovou kapacitu pro memorování obřího lexikonu.

Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 základních slov (*pu* a frekventovaná *ku suli*) a vysoce pravidelnou analytickou syntaxí bez flektivní morfologie tento problém řeší:
* **Redukce úzkého hrdla parametrů vnoření:** Při celoslovní tokenizaci ($V \approx 137$) tvoří matice vnoření méně než 2 % celkového počtu parametrů. Drtivá většina kapacity sítě (98 %+) směřuje přímo do sekvenčního výpočetního jádra (Self-Attention, selektivní stavové vrstvy SSM, dopředné sítě FFN).
* **Obnova strukturální proporcionality vůči velkým modelům:** U velkých modelů (např. 70B+) tvoří matice vnoření rovněž jen 1–2 % celkového parametrického rozpočtu. Snížením velikosti slovníku úměrně k velikosti sítě se v laboratorním prostředí obnovuje strukturální rozložení parametrů typické pro velká LLM.
* **Kontrolované laboratorní prostředí pro měření induktivního zkreslení:** Na rozdíl od syntetických korpusů typu TinyStories (Eldan & Li, 2023) nebo čistě znakových modelů na angličtině umožňuje toki pona zkoumat znakovou, podslovní (BPE) i celoslovní tokenizaci v přirozeném lidském jazyce. Pravidelná analytická syntax bez tvarosloví umožňuje měřit osvojení syntaktických struktur pomocí syntetických minimálních párů (inspirovaných benchmarkem BLiMP; Warstadt et al., 2020) bez rušivého morfologického šumu.
* **Vymezení limitů a vnější validita:** Jazyk toki pona slouží jako kontrolovaný laboratorní model pro studium fundamentálních vlastností architektur, optimalizátorů a tokenizací v podmínkách datové nouze, nikoli jako univerzální reprezentace morfologicky bohatých jazyků. Získané závěry mají platnost pro sekvenční modelování v nízkozdrojovém režimu a formulaci komputačních zákonitostí.

---

## 2. Přehledové srovnání tří variant

| Kritérium | Varianta 1: Čistá statistická hloubka (Transformer) | Varianta 2: Architektonický duel (Transformer vs. Mamba SSM) | Varianta 3: Optimalizace a dynamika tréninku (Hybrid Muon+AdamW vs. AdamW) |
| :--- | :--- | :--- | :--- |
| **Zaměření** | Statistické modelování škálování (data-constrained scaling), vliv tokenizace, dynamika opakování dat a syntaktická kompetence | Srovnání globální pozornosti a rekurentní stavové komprese při striktně vyrovnaném počtu parametrů skrytých vrstev | Vliv maticové spektrální optimalizace (Muon) na tokenovou a výpočetní efektivitu konvergence a posun bodu přeučení při opakování dat |
| **Architektura** | 1× Kauzální Decoder-Only Transformer | 2× Kauzální Transformer vs. selektivní stavový model Mamba (vyrovnaný počet non-embedding parametrů) | 1× Kauzální Decoder-Only Transformer |
| **Zkoumané osy** | Kapacita sítě ($N$) $\times$ Objem dat ($D$) $\times$ Tokenizace $\times$ Opakování epoch ($E$) $\times$ Inicializační semínka ($S$) | Architektura $\times$ Kapacita sítě ($N$) $\times$ Objem dat ($D$) $\times$ Tokenizace $\times$ Opakování epoch ($E$) $\times$ Semínka ($S$) | Optimalizátor $\times$ Kapacita sítě ($N$) $\times$ Tokenizace $\times$ Opakování epoch ($E$) $\times$ Semínka ($S$) |
| **Statistický aparát** | LMM s autoregresní strukturou AR(1) pro křivky učení, GLMM pro položkové minimální páry, nelineární regrese (NLS + bootstrap) pro data-constrained scaling, ANOVA | LMM (interakce architektura $\times$ epocha), GLMM (interakce architektura $\times$ syntaktické pravidlo), nelineární regrese / `nlme` (srovnání škálovacích exponentů), ANOVA | LMM (trajektorie konvergence a posun minima), GLMM (syntaktické testy), nelineární regrese dynamiky ztráty, párové testy rychlosti poklesu BPC na krok a FLOP |
| **Implementační náročnost** | Nízká až střední (čistý PyTorch, stabilní pipeline, maximální soustředění na statistické modelování) | Střední až vyšší (kompilace a správa CUDA kernelů pro Mamba-1/2, párování non-embedding parametrů a FLOPs) | Střední (implementace Newtonovy-Schulzovy ortogonalizace, separátní ladění learning rate pro maticové a 1D váhy) |
| **Soulad se schváleným InSIS** | Plný soulad (přesně odpovídá původnímu znění v InSIS) | Soulad s rozšířením (rozšíření o srovnávací architekturu Mamba v rámci schváleného tématu) | Soulad s rozšířením (rozšíření o srovnávací optimalizátor Muon v rámci schváleného tématu) |

---

## 3. Detailní specifikace variant (Abstrakty, výzkumné otázky a shrnutí)

---

### VARIANTA 1: Čistá statistická hloubka (Transformer)

#### 3.1 Abstrakt (Původní vs. aktualizovaná verze)
> **Původní verze:**  
> Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí slouží v této diplomové práci jako kontrolované prostředí pro výzkum malých jazykových modelů. Malý rozsah slovníku odstraňuje nepoměr v počtu parametrů vnoření a umožňuje soustředit kapacitu sítě do jejích skrytých vrstev, což odpovídá strukturálním proporcím velkých modelů. Cílem práce je statisticky popsat a kvantifikovat, jak objem dat, opakování dat ve více epochách, velikost modelu a úroveň tokenizace ovlivňují ztrátu, datovou efektivitu, dynamiku přeučení a škálování. Nad deduplikovaným korpusem rozděleným podle zdrojů je proveden faktorový experiment s několika velikostmi modelů architektury Transformer, znakovou, podslovní a slovní tokenizací a více náhodnými inicializacemi. Výkon je v průběhu tréninku hodnocen křížovou entropií normalizovanou na znaky, slovní perplexitou, doménovou generalizací a automaticky generovanými testy gramatické přijatelnosti a kompozice. Pomocí analýzy rozptylu, lineárních smíšených modelů, zobecněných lineárních smíšených modelů a nelineární regrese práce odhaduje parametry škálovacích zákonů, určuje kritický bod přeučení při opakování dat a modeluje vztah mezi klesající křížovou entropií a úspěšností v testech gramatiky.
>
> **Aktualizovaná verze:**  
> Většina dosavadních poznatků o škálovacích zákonech a dynamice víceepochového tréninku jazykových modelů vychází z rozsáhlých korpusů přirozených jazyků a modelů s miliardami parametrů. Chování malých jazykových modelů v nízkozdrojovém režimu zůstává nedostatečně prozkoumáno, mimo jiné kvůli úzkému hrdlu parametrů vnoření. V této diplomové práci slouží minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a vysoce pravidelnou analytickou syntaxí jako kontrolované laboratorní prostředí pro výzkum malých modelů architektury Transformer. Malý rozsah slovníku redukuje matice vnoření na méně než 2 % parametrů a obnovuje strukturální rozložení parametrů typické pro velká LLM. Cílem práce je statisticky popsat a kvantifikovat, jak objem tréninkových dat, opakování dat ve více epochách, velikost modelu a úroveň granularity tokenizace ovlivňují ztrátu normalizovanou na znaky, datovou efektivitu, dynamiku přeučení a osvojení syntaktických struktur. Nad deduplikovaným korpusem rozděleným na trénovací, validační a doménově odloženou testovací sadu je realizován faktorový experiment s několika velikostmi modelů Transformer ($10^5$ až $10^7$ parametrů), třemi strategiemi tokenizace (znaková, podslovní BPE a celoslovní) a více náhodnými inicializacemi. Výkonnost je v průběhu tréninku monitorována křížovou entropií normalizovanou na znaky (BPC), slovní perplexitou a automaticky generovanými testy syntaktické přijatelnosti na bázi minimálních párů. Pomocí smíšených lineárních modelů (LMM) s autoregresní strukturou reziduí, logistických zobecněných smíšených modelů (GLMM) a nelineární regrese práce odhaduje parametry data-constrained škálovacích zákonů, určuje polohu kritického bodu přeučení ($R_{\text{crit}}$) a analyzuje funkční závislost mezi spojitou křížovou entropií a diskrétní úspěšností v testech gramatické přijatelnosti.

#### 3.2 Výzkumné otázky a formalizované hypotézy
* **VO1 (Interakce tokenizace a měřítka):** Jak závisí odhadnuté parametry mocninných škálovacích zákonů na zvolené granularitě tokenizace (znaková, podslovní BPE, celoslovní) a dochází v testovaném rozsahu parametrů a dat k bodu překřížení, kdy podslovní či znaková reprezentace překoná celoslovní model?  
  * **H1:** Znaková tokenizace vykazuje pomalejší počáteční pokles BPC v nízkoobjemovém režimu, avšak s rostoucí kapacitou parametrů dosahuje vyššího škálovacího exponentu ($\alpha_{\text{CHAR}} > \alpha_{\text{WORD}}$) a v určitém bodě překoná celoslovní reprezentaci.
* **VO2 (Opakování dat a kritický bod přeučení):** Jak ovlivňuje tokenizační strategie a kapacita sítě polohu minima validační ztráty (kritického bodu přeučení $R_{\text{crit}}$) a dynamiku klesajícího mezního výnosu při opakovaném tréninku podle data-constrained škálovacích zákonů?  
  * **H2:** Modely s jemnější granularitou (znaky, podslova) dosahují minima validační ztráty ($R_{\text{crit}}$) při statisticky významně vyšším počtu opakování epoch než modely celoslovní ($R_{\text{crit, CHAR}} > R_{\text{crit, WORD}}$).
* **VO3 (Vztah spojité ztráty a diskrétní gramatiky):** Jaká je funkční závislost mezi klesající křížovou entropií (BPC) a pravděpodobností úspěchu v syntaktických minimálních párech po zohlednění tréninkového kroku, a liší se tato dynamika mezi jednotlivými gramatickými pravidly?  
  * **H3:** Pravděpodobnost správné volby v minimálních párech roste s klesající BPC nelineárně (logisticky), přičemž práh ztráty nutný pro zvládnutí pravidla je statisticky významně nižší pro lokální částicové vazby (*li*, *e*) než pro hierarchickou větnou modifikaci (*pi*).
* **VO4 (Rozklad rozptylu a vliv inicializace):** Jaký podíl celkového rozptylu validační ztráty je vysvětlen systematickými faktory (kapacita, data, tokenizace) v porovnání s náhodnou variabilitou danou inicializačními semínky?  
  * **H4:** Více než 95 % celkového rozptylu validační BPC je vysvětleno systematickými experimentálními faktory; vnitrotřídní korelační koeficient (ICC) náhodných semínek klesá s rostoucí kapacitou sítě.

#### 3.3 Shrnutí varianty
Varianta s minimálním implementačním a technickým rizikem a stoprocentním souladem se schváleným zadáním v InSISu. Veškeré úsilí je alokováno do precizní statistické analýzy (LMM s časovou korelací, logistické GLMM pro položkové hodnocení minimálních párů, data-constrained nelineární regrese s bootstrapem).

---

### VARIANTA 2: Architektonický duel (Transformer vs. Mamba SSM)

#### 3.1 Abstrakt (Původní vs. aktualizovaná verze)
> **Původní verze:**  
> Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí slouží v této diplomové práci jako kontrolované prostředí pro výzkum malých jazykových modelů. Malý rozsah slovníku eliminuje zkreslení dané velikostí matice vnoření a umožňuje zkoumat vlastnosti sekvenčního jádra sítě. Cílem práce je v podmínkách omezeného objemu dat statisticky porovnat dvě odlišná sekvenční paradigmata, kauzální Transformer s globální pozorností a rekurzivní model stavového prostoru Mamba, a vyhodnotit jejich interakci s tokenizací, kapacitou sítě a opakováním dat. Nad deduplikovaným korpusem rozděleným podle zdrojů je proveden faktorový experiment při striktně vyrovnaném počtu parametrů obou architektur, se znakovou, podslovní a slovní tokenizací a více náhodnými inicializacemi. Výkon je hodnocen křížovou entropií normalizovanou na znaky, slovní perplexitou, doménovou generalizací a automatickými testy gramatické přijatelnosti a kompozice. Pomocí analýzy rozptylu, lineárních a zobecněných lineárních smíšených modelů a nelineární regrese práce odhaduje architektonicky specifické škálovací zákony, testuje vliv rekurzivního stavu na odolnost vůči přeučení při opakování dat a formuluje doporučení pro volbu sekvenční architektury při práci s malými daty.
>
> **Aktualizovaná verze:**  
> Při modelování sekvencí představuje volba mezi globální kauzální pozorností a rekurentním stavem fundamentální architektonický kompromis mezi modelovací flexibilitou a paměťovou efektivitou. V podmínkách malých jazykových modelů a omezených dat však přímé srovnání těchto architektur často naráží na zkreslení způsobené rozsáhlými maticemi vnoření. V této diplomové práci slouží minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a vysoce pravidelnou analytickou syntaxí jako kontrolované prostředí pro rigorózní architektonické srovnání. Malý rozsah slovníku redukuje matice vnoření na zanedbatelnou část kapacity sítě a umožňuje měřit vlastnosti sekvenčního jádra. Cílem práce je v podmínkách omezeného objemu dat statisticky porovnat dvě odlišná paradigmata: kauzální Decoder-Only Transformer s kvadratickou pozorností a selektivní stavový model Mamba s lineární časovou složitostí a fixní dimenzí rekurentního stavu. Faktorový experiment je realizován při striktním párování parametrů bez vnoření (non-embedding parameters $\pm 1\,\%$) napříč kapacitou sítě ($10^5$ až $10^7$ parametrů), objemem dat, třemi úrovněmi tokenizace (znaková, podslovní BPE, celoslovní) a více náhodnými inicializacemi. Výkonnost je hodnocena křížovou entropií normalizovanou na znaky (BPC), slovní perplexitou, doménovou generalizací na odložených textech a automaticky generovanými testy syntaktické přijatelnosti na bázi minimálních párů. Pomocí smíšených lineárních modelů (LMM), logistických zobecněných smíšených modelů (GLMM) a nelineární regrese práce odhaduje architektonicky specifické parametry škálovacích zákonů, testuje rozdíly v trajektoriích přeučení při opakování dat a vyhodnocuje úspěšnost obou architektur na lokálních i dlouhých větných vazbách.

#### 3.2 Výzkumné otázky a formalizované hypotézy
* **VO1 (Interakce architektury a sekvenční délky):** Jak ovlivňuje přechod od krátkých celoslovních sekvencí k výrazně delším znakovým sekvencím relativní predikční výkonnost (BPC) a tréninkovou propustnost modelu Mamba vůči Transformeru při shodném počtu parametrů skrytých vrstev?  
  * **H1:** Relativní výpočetní a predikční efektivita modelu Mamba vůči Transformeru statisticky významně roste s rostoucí délkou sekvence (při přechodu od celoslovní ke znakové tokenizaci).
* **VO2 (Dynamika přeučení napříč architekturami):** Liší se průběh validační ztráty a poloha kritického bodu přeučení ($R_{\text{crit}}$) při opakovaném procházení korpusu mezi modelem s globální pozorností (Transformer) a modelem s fixním rekurentním stavem (Mamba)?  
  * **H2:** Model Mamba vykazuje odlišnou trajektorii přeučení projevující se statisticky významným posunem $R_{\text{crit}}$ do vyšších tréninkových epoch ve srovnání s Transformerem disponujícím přímou pozorností přes celý kontext.
* **VO3 (Architektonické škálovací exponenty):** Liší se statisticky významně odhadnuté parametry data-constrained škálovacích zákonů (výkonnostní exponent kapacity $\alpha$ a objemu dat $\beta$) mezi architekturou Transformer a modelem Mamba v režimu omezených dat?  
  * **H3:** Architektury vykazují srovnatelné škálovací exponenty kapacity $\alpha$, avšak Transformer dosahuje statisticky významně nižší asymptotické neredukovatelné ztráty $E$ při nasycení kapacity parametrů.
* **VO4 (Syntaktická kompetence na dlouhých vazbách):** Zjišťují logistické smíšené modely (GLMM) systematický rozdíl v úspěšnosti obou architektur u syntaktických testů v závislosti na strukturální vzdálenosti závislých elementů (např. oddělení větného kontextu částicí *la* vs. lokální vazby *li*/*e*)?  
  * **H4:** V položkovém modelu GLMM se projevuje statisticky významná interakce `Architecture × Rule`, kdy Transformer vykazuje vyšší úspěšnost u syntaktických konstrukcí s dlouhými hierarchickými vazbami, zatímco na lokálních větných vztazích dosahují obě architektury srovnatelné přesnosti.

#### 3.3 Shrnutí varianty
Přináší vysoce aktuální vědecké srovnání dvou fundamentálních architektur současného deep learningu (globální pozornost vs. stavová rekurze). Metodicky čisté díky striktní podmínce vyrovnání parametrů bez vnoření ($N_{\text{non-emb}} \pm 1\,\%$) a jednotnému protokolu evaluace normalizovanému na fyzické znaky.

---

### VARIANTA 3: Optimalizace a dynamika tréninku (Hybrid Muon+AdamW vs. AdamW)

#### 3.1 Abstrakt (Původní vs. aktualizovaná verze)
> **Původní verze:**  
> Minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a pravidelnou analytickou syntaxí slouží v této diplomové práci jako kontrolované prostředí pro výzkum tréninku jazykových modelů v podmínkách nedostatku dat. Malý slovník soustředí kapacitu modelu do jeho výpočetních vrstev, což umožňuje zkoumat optimalizační dynamiku bez zkreslení rozsáhlými maticemi vnoření. Cílem práce je statisticky popsat a kvantifikovat, jak volba optimalizačního algoritmu, srovnání standardního prvkového algoritmu AdamW s maticovým spektrálním optimalizátorem Muon, úroveň tokenizace a kapacita modelu ovlivňují efektivitu učení, průběh validační ztráty a nástup přeučení při opakování dat. Nad deduplikovaným korpusem rozděleným podle zdrojů je proveden faktorový experiment s architekturou Transformer, sledující vývoj metrik napříč tréninkovými epochami. Výkon je hodnocen křížovou entropií normalizovanou na znaky, perplexitou a automaticky generovanými testy gramatické přijatelnosti a kompozice. Pomocí analýzy rozptylu, lineárních smíšených modelů a nelineární regrese práce kvantifikuje rozdíl v rychlosti konvergence, vyhodnocuje posun bodu přeučení a stanovuje postupy pro efektivní trénink při opakování omezeného korpusu.
>
> **Aktualizovaná verze:**  
> Standardní prvkový optimalizátor AdamW aktualizuje váhy jazykových modelů po jednotlivých souřadnicích bez zohlednění maticové geometrie vah. Nově navržený optimalizátor Muon aplikuje spektrální ortogonalizaci pomocí Newtonovy-Schulzovy iterace na 2D váhové matice, zatímco vnoření a jednorozměrné parametry přenechává algoritmu AdamW. V konvenčních malých modelech na přirozených jazycích je však vliv maticové spektrální optimalizace utlumen tím, že rozsáhlé matice vnoření spotřebují většinu kapacity sítě. V této diplomové práci slouží minimalistický jazyk toki pona s uzavřenou slovní zásobou zhruba 137 slov a vysoce pravidelnou analytickou syntaxí jako kontrolované laboratorní prostředí pro výzkum optimalizační dynamiky. Malý rozsah slovníku redukuje parametry vnoření na méně než 2 %, díky čemuž hybridní konfigurace Muon+AdamW řídí spektrální ortogonalizací přes 98 % všech vah modelu. Cílem práce je statisticky porovnat hybridní optimalizátor Muon+AdamW s referenčním algoritmem AdamW a kvantifikovat jejich vliv na rychlost poklesu normalizované ztráty, hardwarovou a tokenovou efektivitu, průběh validačních křivek a nástup přeučení při opakování dat. Nad deduplikovaným korpusem rozděleným na trénovací, validační a doménově odloženou testovací sadu je proveden faktorový experiment s architekturou Transformer napříč různou kapacitou sítě ($10^5$ až $10^7$ parametrů), třemi úrovněmi tokenizace (znaková, podslovní BPE, celoslovní) a více náhodnými inicializacemi. Výkonnost je hodnocena křížovou entropií normalizovanou na znaky (BPC), slovní perplexitou a automaticky generovanými testy syntaktické přijatelnosti. Pomocí smíšených lineárních modelů (LMM), logistických smíšených modelů (GLMM) a nelineární regrese práce kvantifikuje rozdíly v tokenové i výpočetní efektivitě konvergence, vyhodnocuje posun kritického bodu přeučení ($R_{\text{crit}}$) a testuje robustnost spektrální optimalizace vůči změnám granularity tokenizace.

#### 3.2 Výzkumné otázky a formalizované hypotézy
* **VO1 (Tokenová a výpočetní efektivita konvergence):** Dosahuje hybridní optimalizátor Muon+AdamW v raných a středních fázích tréninku statisticky významně rychlejšího poklesu BPC na stejný počet zpracovaných znaků i na jednotku reálného výpočetního času (zohledňujícího režii Newtonovy-Schulzovy ortogonalizace) ve srovnání s čistým AdamW?  
  * **H1:** Muon+AdamW dosahuje daného prahu validační BPC za statisticky významně nižší počet kroků optimalizátoru i celkových FLOPs než AdamW; časová režie ortogonalizace představuje méně než 5 % celkového času tréninkového kroku.
* **VO2 (Vliv spektrální ortogonalizace na trajektorii přeučení):** Posouvá ortogonalizace maticových aktualizací v algoritmu Muon polohu minima validační ztráty ($R_{\text{crit}}$) do vyšších tréninkových epoch, a oddaluje tak nástup přeučení při opakování dat?  
  * **H2:** Běhy optimalizované algoritmem Muon vykazují statisticky významný posun minima validační BPC ($R_{\text{crit}}$) do vyšších epoch a stabilnější trajektorii validační ztráty po dosažení minima ve srovnání s algoritmem AdamW.
* **VO3 (Interakce optimalizátoru a granularity tokenizace):** Liší se relativní přínos optimalizátoru Muon vůči AdamW mezi krátkými sekvencemi celých slov a dlouhými sekvencemi jednotlivých znaků?  
  * **H3:** Relativní výhoda algoritmu Muon v rychlosti konvergence je invariantní vůči zvolené granularitě tokenizace při zachování konstantního objemu fyzických znaků na optimalizační krok.
* **VO4 (Stabilita a škálování s kapacitou modelu):** Zůstává relativní výhoda spektrálního optimalizátoru stabilní při škálování kapacity modelu od stovek tisíc po jednotky milionů parametrů?  
  * **H4:** Relativní zrychlení konvergence dosažené algoritmem Muon vůči AdamW roste s rostoucí dimenzí skrytých vrstev sítě (kapacitou parametrů $N$).

#### 3.3 Shrnutí varianty
Soustředí se na geometrii optimalizace a chování moderních spektrálních metod při tréninku z omezených dat. V prostředí toki pona má Muon maximální teoretický i praktický dopad, neboť díky zanedbatelné velikosti matic vnoření řídí ortogonálními updaty prakticky celou síť.

---

## 4. Technická a metodická pravidla pro realizaci kódu a výpočtů

Pro zajištění vědecké validity, eliminaci metodických pastí a zajištění plné reprodukovatelnosti musí experimentální protokol a kód dodržovat následujících sedm zásad:

### 4.1 Vyvážení výpočetního kroku a kontextu (Iso-Character a Iso-Context Update)
Znaková tokenizace produkuje podstatně delší sekvence než podslovní nebo celoslovní tokenizace. V toki pona činí průměrná délka slova zhruba 4,4 znaku (včetně mezislovních mezer). Pokud by byl nastaven shodný počet sekvencí v dávce (`batch_size = 32`) a shodná délka kontextu v tokenech, znakový model by v jednom kroku zpracoval zhruba 4,4× méně textu než model celoslovní.

* **Pravidlo pro objem dávky (Iso-Character Step):** Efektivní velikost dávky na jeden optimalizační krok musí být striktně normalizována na konstantní počet fyzických znaků textu:
  $$\text{batch\_size}_{\text{CHAR}} \times L_{\text{CHAR}} \approx \text{batch\_size}_{\text{BPE}} \times L_{\text{BPE}} \approx \text{batch\_size}_{\text{WORD}} \times L_{\text{WORD}} \approx C_{\text{step}}$$
  *Příklad:*
  * `WORD`: $L_{\text{WORD}} = 64$ slov ($\approx 280$ znaků textu), $\text{batch} = 64 \implies 4\,096$ slov $\approx 18\,000$ znaků.
  * `BPE` ($V = 256$, průměr $\approx 2,2$ znaku/token): $L_{\text{BPE}} = 128$ tokenů ($\approx 280$ znaků textu), $\text{batch} = 64 \implies 8\,192$ tokenů $\approx 18\,000$ znaků.
  * `CHAR`: $L_{\text{CHAR}} = 280$ znaků, $\text{batch} = 64 \implies 17\,920$ znaků.
* **Pravidlo pro sémantický kontext (Iso-Context Window):** Délka kontextového okna $L$ v jednotlivých tokenizacích je volena tak, aby modelům poskytovala shodný sémantický úsek textu (zhruba 60–65 celých vět toki pona $\approx 280$ znaků).
* **Škálování rychlosti učení:** Pokud je z paměťových důvodů nutné změnit velikost dávky, rychlost učení se škáluje podle pravidla odmocniny ze součinu velikosti dávky ($\eta \propto \sqrt{\text{batch}}$).

### 4.2 Rigorózní definice BPC a eliminace OOV zkreslení
Srovnání křížové entropie napříč různými tokenizacemi vyžaduje striktní normalizaci na fyzické znaky nezávisle na použitém tokenizéru.

* **Matematická definice BPC:** Pro testovací text o celkové délce $C_{\text{raw}}$ znaků (podle standardu Unicode NFC se zachováním mezer a interpunkce) je hodnota BPC (Bits Per Character) definována jako:
  $$\text{BPC} = \frac{\sum_{t=1}^T -\log_2 P(x_t \mid x_{<t})}{C_{\text{raw}}}$$
  kde $T$ je počet tokenů, do nichž daný tokenizér text rozložil, a $P(x_t \mid x_{<t})$ je modelová pravděpodobnost přiřazená $t$-tému tokenu. Jmenovatel $C_{\text{raw}}$ je pro daný text naprosto identický pro znakový, podslovní i celoslovní model.
* **Převeditelnost na slovní perplexitu:** Slovní perplexita nepředstavuje nezávislou metriku, ale je jednoznačným přepočtem z BPC:
  $$\text{PPL}_{\text{word}} = 2^{\text{BPC} \times \bar{c}_{\text{word}}}$$
  kde $\bar{c}_{\text{word}} = C_{\text{raw}} / W$ je průměrný počet znaků na slovo v testovacím korpusu ($W$ je počet slov).
* **Ošetření mimoslovníkových tokenů (OOV) u celoslovního modelu:**
  Lexikon toki pona obsahuje 137 kanonických slov. Vlastní jména (*nimi ma*, *nimi jan*) a cizí výpůjčky tvoří otevřenou třídu. Nahrazení celého víceznakového jména jediným levným tokenem `<unk>` by uměle a nepřípustně snížilo ztrátu celoslovního modelu.
  * **Pravidlo (Fallback spelling):** Celoslovní tokenizér implementuje znakový fallback (při narazení na mimoslovníkové slovo tokenizuje toto slovo po jednotlivých znacích), čímž je zachována uzavřenost slovníku i férovost výpočtu BPC. Alternativně se při použití `<unk>` přičítá uniformní entropická penalizace délky slova:
    $$-\log_2 P(\text{word}) = -\log_2 P(\langle\text{unk}\rangle) + |\text{word}| \cdot \log_2 |\Sigma_{\text{char}}|$$

### 4.3 Tréninkový protokol, learning rate schedule a detekce $R_{\text{crit}}$
V abstraktu se deklaruje určení kritického bodu přeučení při opakování dat ($R_{\text{crit}}$). Standardní trénink s kosinovým útlumem (cosine annealing) naplánovaným na pevný počet kroků však vede k metodické chybě: v epoše $e < E_{\max}$ má model uměle vysoký learning rate, takže jeho ztráta nereprezentuje optimálně dotrénovaný model pro daný počet epoch (Hoffmann et al., 2022).

* **Plán učení (Learning Rate Schedule):** Pro korektní vyhodnocení dynamiky přeučení se použije **Warmup-Stable-Decay (WSD)** plán:
  1. *Warmup:* Lineární náběh během prvních 2 % kroků.
  2. *Stable:* Konstantní rychlost učení $\eta_{\max}$ po celou dobu tréninku.
  3. *Decay:* V libovolném kontrolním bodě (nebo na konci tréninku) proběhne krátký útlum (cooldown, 5–10 % kroků) na minimální rychlost učení.
* **Detekce a vyhlazení $R_{\text{crit}}$:**
  Validační BPC se měří a loguje **na konci každé jednotlivé epochy** $e \in \{1, \dots, E_{\max}\}$. Aby náhodný šum jednotlivé epochy nezpůsobil falešné minimum, křivka validační BPC se vyhladí pomocí lokální polynomiální regrese (LOESS) nebo kubického splajnu $\widehat{\text{BPC}}_{\text{val}}(e)$:
  $$R_{\text{crit}} = \arg\min_{e \in \{1, \dots, E_{\max}\}} \widehat{\text{BPC}}_{\text{val}}(e)$$
* **Ošetření pravostranného cenzoringu:** Pokud validační křivka klesá až do $E_{\max}$, hodnota $R_{\text{crit}}$ je pravostranně cenzorovaná ($R_{\text{crit}} \ge E_{\max}$). V statistických analýzách se takové běhy ošetřují pomocí modelů analýzy přežití nebo Tobitovy regrese, nikoli pouhým dosazením hodnoty $E_{\max}$.
* **Dvoustupňová evaluace:** Kontrolní bod $R_{\text{crit}}$ se určuje výhradně na validační sadě. Finální generalizační ztráta, slovní perplexita a syntaktické minimální páry se pro tento kontrolní bod vyhodnocují na **striktně odložené testovací sadě**.

### 4.4 Škálovací zákony v režimu opakovaných dat (Data-Constrained Scaling)
Standardní formulace škálovacího zákona Chinchilla (Hoffmann et al., 2022) $L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$ předpokládá trénink v jediné epoše nad unikátními daty. V režimu opakování dat dochází k nasycení efektivní velikosti korpusu.

* **Data-Constrained formulace (Muennighoff et al., 2023):**
  Při $R$ průchodech daty ($R = \text{epochy}$) se efektivní objem dat $D'$ modeluje jako:
  $$D' = D_U \cdot R_D^* \left(1 - \exp\left(-\frac{R}{R_D^*}\right)\right)$$
  kde $D_U$ je objem unikátních dat v tokenech a $R_D^*$ je saturační konstanta opakování. Výsledná ztráta je dána vztahem:
  $$L(N, D_U, R) = E + \frac{A}{N^\alpha} + \frac{B}{(D')^\beta}$$
* **Identifikovatelnost a odhad parametrů:**
  Parametry $E, A, B, \alpha, \beta$ a $R_D^*$ se odhadují nelineární metodou nejmenších čtverců (Levenberg-Marquardtův algoritmus) s logaritmickou Huberovou ztrátou $\rho(\ln \hat{L} - \ln L)$ pro potlačení heteroskedasticity a odlehlých hodnot.
* **Nejistota odhadu:** Intervaly spolehlivosti pro odhadnuté exponenty $\alpha$ a $\beta$ a test hypotézy o rozdílu exponentů napříč architekturami (V2 VO3) se konstruují neparametrickým bootstrapem vzorkováním trénovacích semínek (1 000 replikací).

### 4.5 Rigorózní formulace statistických modelů

#### A. Longitudinální analýza křivek učení (LMM)
Opakovaná měření v čase (napříč epochami uvnitř tréninkového běhu) vykazují silnou autokorelaci. Model v R (`nlme`) nesmí předpokládat lineární vliv epochy, nýbrž nelineární trajektorii s náhodnými sklony a autoregresní chybovou strukturou AR(1):
```r
# R (balík nlme): Model longitudinální trajektorie validační BPC
library(nlme)
lmm_trajectories <- lme(
  fixed = BPC ~ Tokenizer * log(Params) + poly(log(Epoch), 2),
  random = ~ 1 + log(Epoch) | Run_ID,
  correlation = corAR1(form = ~ Epoch | Run_ID),
  data = training_logs
)
```
*Identifikátory:* `Run_ID` unikátně identifikuje konkrétní trénovací běh (kombinaci faktorů $N$, $D$, tokenizace a konkrétního semínka). Rozklad rozptylu (VO4) se provádí pomocí vnitrotřídního korelačního koeficientu (ICC) rozložením komponent rozptylu mezi systematické faktory a náhodná inicializační semínka.

#### B. Položková analýza syntaktické přijatelnosti (GLMM)
Výsledky testů syntaktických minimálních párů představují binární odezvu $Y_{ijk} \in \{0, 1\}$ (1 = model přiřadil vyšší normalizovanou pravděpodobnost gramatické větě než negramatické větě z páru). Model obsahuje náhodný efekt konkrétní větné dvojice (`Item_ID`, vnořený v pravidle) a náhodný efekt modelu (`Run_ID`):

* **Pro Variantu 1:**
  ```r
  glmer(Correct ~ Tokenizer * Rule + scale(BPC_val) + (1 | Item_ID) + (1 | Run_ID), 
        family = binomial(link = "logit"), data = grammar_results)
  ```
* **Pro Variantu 2:**
  ```r
  glmer(Correct ~ Architecture * Rule + Tokenizer + (1 | Item_ID) + (1 | Run_ID), 
        family = binomial(link = "logit"), data = grammar_results)
  ```
* **Pro Variantu 3:**
  ```r
  glmer(Correct ~ Optimizer * Rule + Tokenizer + (1 | Item_ID) + (1 | Run_ID), 
        family = binomial(link = "logit"), data = grammar_results)
  ```

*Vyhodnocení minimálního páru napříč tokenizacemi:* Skóre věty $S$ je definováno jako suma log-pravděpodobností tokenů normalizovaná celkovým počtem znaků věty:
$$\text{Score}(S) = \frac{1}{|S|_{\text{char}}} \sum_{t=1}^{|S|_{\text{tok}}} \log P(x_t \mid x_{<t})$$
Model uspěje v testu minimálního páru, pokud $\text{Score}(S_{\text{grammatical}}) > \text{Score}(S_{\text{ungrammatical}})$.

#### C. Faktorová analýza rozptylu (ANOVA)
Faktorová ANOVA se aplikuje výhradně na agregované skalární metriky v optimálním bodě (např. minimální validační BPC v bodě $R_{\text{crit}}$, hodnota $R_{\text{crit}}$). Reportují se parciální velikosti účinku $\eta_p^2$ a post-hoc párová srovnání s Holm-Bonferroniho korekcí pro mnohonásobné testování hypotéz.

### 4.6 Protokol ladění hyperparametrů a férového srovnání
Srovnání architektur (V2) či optimalizátorů (V3) při fixních hyperparametrech vede k artefaktům, neboť optimální rychlost učení se mezi modely zásadně liší.

* **Separátní ladění learning rate:** Na střední velikosti modelu ($N \approx 1\text{M}$) se provede systematický grid search optimální hodnoty $\eta_{\max}$:
  * Pro AdamW: $\eta \in \{1 \times 10^{-4}, 3 \times 10^{-4}, 6 \times 10^{-4}, 1 \times 10^{-3}, 3 \times 10^{-3}\}$.
  * Pro hybridní Muon+AdamW: pro 2D matice $\eta_{\text{Muon}} \in \{0.02, 0.05, 0.1\}$, pro vnoření a biasy $\eta_{\text{AdamW}} \in \{3 \times 10^{-4}, 1 \times 10^{-3}\}$.
* **Fixní regularizace pro determinismus škálování:** Všechny modely používají identický weight decay ($0.1$) a gradient clipping ($1.0$). Dropout je nastaven na $0$, aby nedocházelo k deformaci škálovacích křivek stochastickým vypínáním neuronů.
* **Párování parametrů u Mamba vs. Transformer (V2):** Počty parametrů obou architektur se vyrovnávají výhradně na úrovni parametrů skrytých vrstev (**non-embedding parameters** v toleranci $\pm 1\,\%$). Zároveň se měří reálné tréninkové FLOPs na krok pro kontrolu hardwarové efektivity.

### 4.7 Hygiena dat, rozdělení korpusu a prevence úniku dat
* **Deduplikace korpusu:** Před rozdělením korpusu se provede exaktní deduplikace i fuzzy deduplikace (MinHash LSH s Jaccardovým prahem 0.8), aby se vyloučilo memorování shodných textů mezi trénovací a testovací sadou.
* **Třísložkové hierarchické dělení:**
  1. *Trénovací sada (80 %)*
  2. *In-domain validační sada (10 %):* Náhodný stratifikovaný výběr z tréninkových zdrojů pro detekci $R_{\text{crit}}$ a ladění hyperparametrů.
  3. *Out-of-domain testovací sada (10 %):* Kompletně odložené specifické zdroje (např. překlady beletrie, ucelené konverzační korpusy), sloužící výhradně pro finální vyhodnocení doménové generalizace a syntaktických testů.
* **Konzistence tokenizéru:** Podslovní tokenizér (BPE) se trénuje **striktně pouze na trénovací sadě korpusu**. Aplikace BPE natrénovaného na celém korpusu představuje nepřípustný únik informací (data leakage).

---

## 5. Klíčová akademická literatura

1. **Hoffmann, J., et al. (2022).** *Training Compute-Optimal Large Language Models.* DeepMind. [Chinchilla Scaling Laws, arXiv:2203.15556]
2. **Muennighoff, N., et al. (2023).** *Scaling Data-Constrained Language Models.* NeurIPS 2023. [Data-Constrained Scaling Laws, arXiv:2305.16264]
3. **Kaplan, J., et al. (2020).** *Scaling Laws for Neural Language Models.* OpenAI. [arXiv:2001.08361]
4. **Gu, A., & Dao, T. (2023).** *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752]
5. **Jordan, K. (2024) / Liu, et al. (2025).** *Muon: An optimizer for hidden layers in neural networks.* [Spectral Orthogonalized Matrix Optimization]
6. **Warstadt, A., et al. (2020).** *BLiMP: The Benchmark of Linguistic Minimal Pairs for English.* TACL, 8, 377–392.
7. **Eldan, R., & Li, Y. (2023).** *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* Microsoft Research. [arXiv:2305.07759]
