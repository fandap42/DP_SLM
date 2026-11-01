# Manažerský průvodce třemi variantami diplomové práce (DP_SLM)

> **Účel dokumentu:** Lidsky, srozumitelně a bez zbytečného akademického žargonu vysvětlit, o čem každá ze tří variant je, jaký reálný problém v praxi řeší, kolik práce vás na ní čeká, kde jsou rizika a jak ji prodáte u obhajoby.

---

## Společný příběh všech 3 variant: „Proč trénujeme malé modely na umělém jazyce?“ (Váš hlavní prodejní argument)

Představte si, že vyrábíte letadla (velké jazykové modely jako GPT-4 nebo Llama-70B). Postavit jedno velké letadlo a poslat ho do vzduchu stojí **2 miliony dolarů**. Nemůžete jich postavit stovku a schválně je nechat havarovat, jen abyste statisticky změřil, jak tvar křídla ovlivňuje spotřebu paliva.

Potřebujete **aerodynamický tunel a zmenšený model letadla**, který stojí pár korun, takže v něm můžete udělat **300 přesných měření** a odvodit z nich fyzikální zákony pro velké letadlo.

* **Proč nejde udělat zmenšený model na angličtině nebo češtině?**  
  Protože angličtina má ve slovníku ~128 000 slov. Když zmenšíte model z 70 miliard parametrů na 5 milionů parametrů, ale necháte mu anglický slovník, **85 % celého modelu zabere jen tabulka slovíček** a na samotný „mozek“ (skryté vrstvy) zbyde jen 15 %. To je jako model letadla, který má místo křídel obří olověný kufr – v tunelu se chová úplně jinak než velké letadlo (kde slovník tvoří jen 2 % a mozek 98 %).
* **Proč je Toki Pona geniální řešení?**  
  Toki Pona má jen **137 základních slov**, ale plnohodnotnou větnou skladbu. Když zmenšíte model na pár milionů parametrů a zároveň zmenšíte slovník na 137 slov, **vrátíte modelu přesný poměr velkého LLM: 2 % slovník, 98 % mozek sítě!**  
  Díky tomu máte v rukou **dokonale proporcionální zmenšeninu velkého LLM**, na které můžete za pár hodin na jedné grafické kartě udělat stovky statistických pokusů a výsledky zobecnit pro svět velkých modelů.

---

## VARIANTA 1: „Čistá statistická hloubka“ (Sázka na jistotu a špičkovou statistiku)

### 1. O co v téhle variantě lidsky jde (Elevator Pitch)
Vezmete jednu jedinou, průmyslově nejrozšířenější architekturu (**Transformer**) a proženete ji v našem „aerodynamickém tunelu“ všemi myslitelnými režimy: od miniaturního modelu po větší, od 10 % dat po 100 % dat, od 1 průchodu daty až po 64 opakování (kdy už se model učí texty nazpaměť) a se třemi způsoby čtení textu (po písmenech, po slabikách, po celých slovech).  
**Neřešíte žádné programátorské novinky – 100 % vaší energie jde do luxusní statistické analýzy výsledků.**

### 2. Jaký reálný problém z praxe to řeší?
Firmy dnes trénují menší specializované modely na vlastních interních datech (právní smlouvy, lékařské zprávy, průmyslový kód), kterých mají málo. Potřebují vědět tři věci:
1. **Jak text rozsekat (tokenizovat):** Když máme málo dat, je lepší dávat modelu celá slova, nebo podslova/znaky? Kde přesně se to láme (kdy hrubý slovník přestane stačit a vyhraje jemnější rozsekání)?
2. **Kolikrát můžeme stejná data zopakovat:** Když nemáme nová data a pustíme trénink 10× dokola přes ta samá data, v které chvíli se model přestane učit pravidla a začne jen tupě papouškovat (přefitovávat)?
3. **Kdy model „pochopí“ gramatiku a logiku:** Klesá chybovost modelu plynule, nebo nastane jeden konkrétní zlomový bod, kdy modelu „sepne“ a najednou začne tvořit bezchybné věty?

### 3. Co přesně budete dělat v praxi (Náročnost a kroky)
* **Programování (20 % práce):** Velmi snadné. Základní trénovací pipeline pro Transformer už v podstatě máte hotovou. Jen pustíte plnou mřížku běhů na GPU/klastru a budete ukládat průběžné výsledky v 1., 4., 16. a 64. epoše.
* **Statistika a grafy (80 % práce):** Tady se vyřádíte jako statistik:
  * Uděláte křivky učení v čase (jak klesá chyba a kde se láme do přefitování).
  * Vezmete psychometrický model (**IRT – Teorie odpovědi na položku**, která se používá třeba u testů Scio nebo PISA, aby se zjistilo, která otázka je těžká a která lehká) a aplikujete ho na 447 gramatických chytáků v Toki Pona. Tím přesně změříte, která gramatická pravidla jsou pro umělou inteligenci lehká a která těžká.
  * Pomocí smíšených modelů (LMM) a bootstrapu spočítáte, jak velkou roli hraje náhoda (jaké náhodné číslo / seed na začátku tréninku padlo).

### 4. Rizika a úzká hrdla
* **Riziko zaseknutí na kódu: NULOVÉ.** Vše běží na standardním PyTorchi na jakékoliv grafické kartě (Windows, Linux, AMD i NVIDIA).
* **Jediné riziko:** Práce by mohla působit „moc obyčejně z pohledu AI“, pokud byste v ní měl jen základní tabulku. Proto ji v této variantě vytáhnete nahoru právě špičkovou statistikou (křivky učení, psychometrické IRT, bootstrap).

### 5. Jak to prodáte u obhajoby (Co na to řekne dr. Šafr a komise)
* **100% soulad se schváleným zadáním v InSISu** – nikdo nemůže říct ani popel, že jste se odchýlil od schváleného textu.
* Komise na Katedře statistiky a pravděpodobnosti bude nadšená, protože uvidí metody, kterým perfektně rozumí a které učí (ANOVA, smíšené modely, nelineární regrese, psychometrie IRT), aplikované na moderní téma jazykových modelů.

---

## VARIANTA 2: „Architektonický duel: Transformer vs. Mamba-3“ (Zlatý střed: AI souboj + statistika)

### 1. O co v téhle variantě lidsky jde (Elevator Pitch)
Postavíte proti sobě do ringu **dva největší rivaly současné umělé inteligence** při naprosto férových podmínkách (stejný počet parametrů „na váhu“, stejná data):
1. **Transformer (obhájce titulu):** Při čtení věty se dívá na všechna předchozí slova najednou. Má fotografickou paměť, ale snadno podvádí tím, že si malá data prostě zapamatuje nazpaměť.
2. **Mamba-3 (vyzývatel z roku 2026):** Čte větu slovo po slovu jako člověk a vše podstatné si musí průběžně stlačovat do jedné malé „krabičky v hlavě“ (rekurzivního stavu).

V našem „aerodynamickém tunelu“ změříte, který z těchto dvou mozků se dokáže z malého množství dat naučit jazyk lépe, rychleji a bez tupého šprtání nazpaměť.

### 2. Jaký reálný problém z praxe to řeší?
Celý AI svět v roce 2026 řeší, jestli má nahradit klasické Transformery novými stavovými modely (Mamba). Všichni je ale porovnávají jen na obřích datech a dlouhých knihách. Skoro nikdo neví, **jak se Mamba chová, když má málo dat** nebo když jí text rozsekáte na jednotlivá písmena (`CHAR`), takže je věta najednou 6× delší a Mamba si musí v té své malé paměťové krabičce pamatovat každé písmenko.
* Vaše práce přinese jasnou odpověď: *„Když máte málo dat, je „úzké hrdlo“ v paměti Mamby výhoda (protože ji donutí pochopit pravidla místo šprtání), nebo nevýhoda (protože zapomene začátek věty)?“*

### 3. Co přesně budete dělat v praxi (Náročnost a kroky)
* **Programování (40 % práce):** K vašemu stávajícímu Transformeru dopíšete druhý model – Mambu (buď přes oficiální knihovnu `mamba-ssm`, nebo jako čistý maticový algoritmus SSD v PyTorchi). Musíte si dát pozor na jednu věc: nastavit šířku vrstev tak, aby měl Transformer i Mamba na chlup stejný počet parametrů ($\pm 1\,\%$).
* **Statistika a grafy (60 % práce):** Všechny vaše grafy a testy budou mít krásný srovnávací příběh **Červená křivka (Transformer) vs. Modrá křivka (Mamba)**:
  * Kdo vyhraje na slovech (`WORD`) a kdo vyhraje na písmenech (`CHAR`)? (Interakce v ANOVA).
  * Kdo se dřív přetrénuje, když mu stejná data pustíme 64× dokola?
  * Kdo dělá víc chyb v dlouhých souvětích s částicemi *pi* a *la*?

### 4. Rizika a úzká hrdla
* **Technické riziko (Střední):** Oficiální Mamba-3 používá speciální nízkoúrovňové GPU skripty (Triton kernely), které občas zlobí při instalaci na Windows nebo na některých školních serverech. *Řešení:* Protože v Toki Pona máte krátké věty, můžete Mambu napsat v ~80 řádcích čistého PyTorche přes maticové násobení (chunked SSD) – poběží bleskově všude a nebudete závislý na cizích kernelech.
* **Výpočetní čas:** Protože trénujete 2 architektury místo 1, poběží trénink 2× déle (což je ale u malých modelů otázka maximálně jednoho víkendu na GPU).

### 5. Jak to prodáte u obhajoby
* Máte **extrémně chytlavý příběh**. Každého v místnosti (i člena komise, který není expert na hluboké učení) okamžitě zaujme souboj „Fotografická paměť (Transformer) vs. Průběžná komprese (Mamba)“.
* Pro statistiky máte učebnicovou ukázku **statistické interakce** (ukážete, že neexistuje jeden absolutní vítěz, ale že výsledek závisí na kombinaci architektury a způsobu tokenizace).

---

## VARIANTA 3: „Překonání datové zdi: Optimalizátor Muon a Kurikulární učení“ (Inovátor tréninku 2026)

### 1. O co v téhle variantě lidsky jde (Elevator Pitch)
Neřešíte tvar mozku (necháte klasický **Transformer**), ale řešíte **metodu výuky – tedy „učitele a učebnici“**, když máte kritický nedostatek studijních materiálů (*Data Wall*):
1. **Nový učitel (Optimalizátor Muon vs. AdamW):** Standardní algoritmus učení (AdamW) upravuje každý neuron v síti zvlášť, kvůli čemuž se síť rychle upne na pár nejčastějších frází a zbytek ignoruje. Nový hit roku 2025/2026 (**Muon**) upravuje celé bloky neuronů najednou tak, aby se učily rovnoměrně ve všech směrech.
2. **Seřazená učebnice (Kurikulární učení & Míchání žánrů):** Místo abyste na model chrlil náhodně přeházené věty, otestujete, co se stane, když mu věty seřadíte jako ve škole – v 1. třetině tréninku krátké jednoduché věty (*Tatoeba*) a postupně přidáváte složitá encyklopedická a literární souvětí (*Wikipesija, Lipu Sewi*).

### 2. Jaký reálný problém z praxe to řeší?
V roce 2026 narazil celý AI průmysl na tzv. **Datovou zeď (Data Wall)** – na světě už nejsou žádné další triliony stránek lidského textu. Všichni musí trénovat na omezených datech vícero průchodů (epoch) a hledají způsob, **jak ze stejného množství dat vymáčknout dvojnásobek znalostí, aniž by se model začal učit texty nazpaměť**.
* Vy na svém mikro-modelu statisticky otestujete, zda maticový optimalizátor **Muon** a **chytré dávkování dat (kurikulum + míchání žánrů)** skutečně dokážou oddálit moment přefitování (posunout bod zlomu z 4. epochy třeba na 16. epochu).

### 3. Co přesně budete dělat v praxi (Náročnost a kroky)
* **Programování (25 % práce):** Překvapivě velmi jednoduché! Nemusíte instalovat žádné složité knihovny. Optimalizátor **Muon** je doslova 25 řádků maticového násobení (tzv. Newton-Schulzova iterace), které vložíte přímo do svého trénovacího skriptu vedle AdamW. A seřazení vět podle délky a počtu gramatických částic (*li, e, pi, la*) v Toki Pona je otázka jednoho krátkého Python skriptu.
* **Statistika a grafy (75 % práce):**
  * Porovnáte rychlost klesání chyby (kolik procent trénovacích dat ušetří Muon oproti AdamW, aby dosáhl stejného BPC).
  * Změříte U-křivku přefitování v 1., 4., 16. a 64. epoše (kdo vydrží déle klesat bez memorování).
  * Použijete statistický **návrh experimentů se směsmi (Mixture DoE)** – spočítáte, v jakém poměru namíchat 4 žánry korpusu, aby model nejlépe rozuměl cizím textům.

### 4. Rizika a úzká hrdla
* **Technické riziko: NÍZKÉ.** Všechno běží na vašem stávajícím Transformeru v čistém PyTorchi.
* **Metodické riziko:** U Muonu je potřeba správně nastavit rychlost učení (learning rate), aby bylo srovnání s AdamW férové. Proto si na začátku uděláte malý kalibrační pokus (statistickou plochu odezvy), což vám mimochodem přinese další skvělou statistickou podkapitolu!

### 5. Jak to prodáte u obhajoby
* Spojíte nejžhavější optimalizační novinku současnosti (**Muon**, o kterém teď mluví celá AI komunita díky modelům Kimi/Moonlight) s klasickou průmyslovou statistikou (**plochy odezvy a míchání směsí – Mixture DoE**, které se v chemii a průmyslu používají desítky let, ale v AI je málokdo umí správně spočítat).

---

## Finální doporučení: Kterou si máte vybrat?

1. **Chcete klidné spaní, minimum programování navíc, 100% shodu s InSISem a čistou statistiku?**  
   👉 Berte **VARIANTU 1 (Čistá statistická hloubka)**.
2. **Chcete mít v práci atraktivní souboj dvou technologií, který zaujme na první pohled, a nevadí vám dopsat jeden model navíc?**  
   👉 Berte **VARIANTU 2 (Architektonický duel: Transformer vs. Mamba-3)**.
3. **Chcete zkoumat, „jak se sítě učí a proč přefitovávají“, vyzkoušet hit roku 2026 (Muon) a přitom se vyhnout kompilaci cizích knihoven?**  
   👉 Berte **VARIANTU 3 (Překonání datové zdi: Muon vs. AdamW & Kurikulum)**.
