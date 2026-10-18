# Šablona prezentací FIS VŠE (16:9)

Tento balíček slouží pro snadné a rychlé vytváření prezentací ve fakultní šabloně **FIS VŠE (poměr stran 16:9)**. 

Umožňuje 3 různé způsoby práce:
1. **Z deklarativního YAML souboru** (stačí napsat texty, odrážky a cesty k obrázkům).
2. **Programaticky v Pythonu** (ideální pro automatické generování grafů z experimentů a jejich přímé vkládání do slidů).
3. **Přímo v PowerPointu** (otevřít připravenou šablonu s předvyplněnými layouty).

Veškeré vygenerované prezentace (`slides/*.pptx`) jsou automaticky v `.gitignore`, takže nezatěžují Git repozitář velkými binárními soubory, zatímco složka se šablonou `slides/template/` zůstává verzovaná.

---

## 📂 Soubory v této složce

| Soubor | Účel |
|---|---|
| `FIS_prezentace_sablona.pptx` | **Hotová šablona k otevření v PowerPointu** – obsahuje 8 vzorových snímků s popiskami a zástupnými poli. |
| `FIS_CZ_16_9.pptx` | Čistý PPTX základ s definovanými předlohami a layouty (používá `builder.py`). |
| `builder.py` | Python knihovna (`FISPresentation`) pro tvorbu a formátování prezentací (včetně automatického exportu do PDF). |
| `generate.py` | CLI skript pro převod YAML konfigurace na výslednou prezentaci `.pptx` i `.pdf`. |
| `export_pdf.ps1` | PowerShell skript pro nativní převod PPTX -> PDF přes PowerPoint COM. |
| `example_presentation.yaml` | Vzorová konfigurace v YAML s ukázkami všech podporovaných typů snímků. |
| `example_script.py` | Ukázkový Python skript demonstrující generování grafu v `matplotlib` a export do PPTX i PDF. |


---

## 🚀 Způsob 1: Tvorba prezentace z YAML (Doporučeno pro běžné prezentace)

Nejrychlejší způsob: stačí vytvořit `.yaml` soubor s obsahem a spustit generátor.

### 1. Zkopírujte šablonu
```bash
cp slides/template/example_presentation.yaml slides/moje_prezentace.yaml
```

### 2. Upravte texty v YAML
```yaml
meta:
  author: "Bc. František Pavlík"
  department: "Katedra statistiky a pravděpodobnosti (FIS VŠE)"
  email: "pavf05@vse.cz"

slides:
  - type: title
    title: "Vývoj Small Language Modelu"
    subtitle: "Obhajoba diplomové práce"

  - type: section
    title: "1. Motivace a cíle"

  - type: content
    title: "Proč lokální jazykové modely?"
    bullets:
      - "**Ochrana dat:** Interní data neopouštějí zařízení."
      - "**Nízká latence:** Rychlá odezva bez nutnosti internetového připojení."
      - text: "**Úspora nákladů:**"
        items:
          - "Nulové poplatky za tokeny v cloudu"
          - "Běh na běžném spotřebním hardware"

  - type: content
    title: "Trénovací proces"
    bullets:
      - "Trénováno na českém korpusu s 5B tokeny"
      - "Využití optimalizovaného BPE tokenizeru"
    image: "results/figures/loss.png"   # volitelný obrázek/graf
    caption: "Obr. 1: Vývoj loss křivky"

  - type: two_column
    title: "Srovnání přístupů"
    left_title: "Cloudové LLM (70B)"
    left_bullets:
      - "Vysoké náklady na infrastrukturu"
      - "Riziko úniku interních dat"
    right_title: "Lokální SLM (1.5B)"
    right_bullets:
      - "**Běží přímo na notebooku**"
      - "**100% soukromí a kontrola**"

  - type: table
    title: "Výsledky na benchmarku"
    headers: ["Model", "Parametry", "CZ-MMLU", "Rychlost"]
    rows:
      - ["Baseline", "1.2B", "44.6 %", "71 tok/s"]
      - ["Náš model", "1.5B", "**59.4 %**", "**58 tok/s**"]

  - type: closing
    title: "Děkuji za pozornost"
```

### 3. Vygenerujte prezentaci (vytvoří PPTX i PDF)
```bash
python slides/template/generate.py slides/moje_prezentace.yaml -o slides/moje_prezentace.pptx
# Automaticky vytvoří slides/moje_prezentace.pptx i slides/moje_prezentace.pdf
# (Pokud PDF nepotřebujete, můžete přidat přepínač --no-pdf)
```


---

## 🐍 Způsob 2: Programatická tvorba v Pythonu (Ideální pro experimenty a grafy)

Pokud v diplomce generujete grafy (`matplotlib`, `seaborn`), můžete je do prezentace vložit přímo z paměti bez ukládání dočasných souborů:

```python
import matplotlib.pyplot as plt
from slides.template import FISPresentation

# 1. Inicializace
prs = FISPresentation(
    author="Bc. František Pavlík",
    department="Katedra statistiky a pravděpodobnosti",
    email="pavf05@vse.cz"
)

# 2. Titulní snímek
prs.add_title_slide("Vývoj SLM pro češtinu", "Diplomová práce")

# 3. Vygenerování grafu v matplotlib
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
ax.plot([1, 2, 3, 4], [2.5, 1.8, 1.2, 0.9], marker="o", color="#002F6C")
ax.set_title("Trénovací ztráta", fontweight="bold")
fig.tight_layout()

# 4. Vložení grafu přímo do snímku (spolu s odrážkami)
prs.add_content_slide(
    title="Průběh trénování",
    items=[
        "**Konvergence:** Model stabilně konverguje.",
        ("Validační ztráta dosáhla optima ve 4. epoše", 1),
    ],
    image=fig,                 # Matplotlib Figure objekt přímo!
    image_side="right",        # "right" nebo "left"
    image_caption="Graf 1: Loss křivka"
)

# 5. Samostatný velký graf
prs.add_image_slide("Detailní architektura", image="docs/figures/architecture.png")

# 6. Tabulka
prs.add_table_slide(
    title="Kvantitativní výsledky",
    headers=["Metoda", "Přesnost", "F1"],
    rows=[
        ["Původní", "52.4 %", "0.51"],
        ["Navržená", "**64.8 %**", "**0.63**"],
    ]
)

# 7. Závěr
prs.add_closing_slide()

# 8. Uložení
prs.save("slides/obhajoba.pptx")
```

Spustitelný vzor najdete v [`example_script.py`](file:///C:/Users/pavli/Documents/git/DP_SLM/slides/template/example_script.py):
```bash
python slides/template/example_script.py
```

---

## 💻 Způsob 3: Ruční tvorba přímo v PowerPointu

Pokud potřebujete prezentaci vytvořit nebo doladit ručně:
1. Otevřete soubor [`FIS_prezentace_sablona.pptx`](file:///C:/Users/pavli/Documents/git/DP_SLM/slides/template/FIS_prezentace_sablona.pptx).
2. Soubor obsahuje 8 připravených snímků pokrývajících všechny běžné potřeby:
   - Titulní snímek
   - Běžný snímek s odrážkami
   - Dvousloupcové srovnání
   - Text s prostorem pro graf / obrázek
   - Velký centrální graf / schéma
   - Tabulku s výsledky
   - Předělový mezititulek sekce
   - Závěrečný snímek s kontaktem
3. Uložte jako novou prezentaci mimo šablonu (např. do `slides/moje_prezentace.pptx`).

---

## 🎨 Podporované formátování textu

V textu odrážek (v YAML i Pythonu) lze používat jednoduchý Markdown:
- `**tučný text**` -> **tučné písmo**
- `*kurzíva*` -> *kurzíva*
- `` `kód` `` -> monospaced písmo (Consolas)

### Úrovně odrážek
V YAML:
```yaml
bullets:
  - "**Úroveň 0 (hlavní bod)**"
  - text: "**Úroveň 0 s podbody**"
    items:
      - "Úroveň 1 (podrobnost)"
      - "Další podbod"
```
Nebo v Pythonu pomocí n-tice `("Text", úroveň)`:
```python
items = [
    "Hlavní bod",
    ("Podbod na úrovni 1", 1),
    ("Ještě hlubší podbod", 2),
]
```

---

## 🖼️ Vkládání obrázků a grafů

Generátor automaticky:
- **Zachovává poměr stran (aspect ratio)** – grafy ani screenshoty nebudou nikdy deformované.
- **Centruje obrázek** v přiděleném prostoru (vedle textu nebo na střed snímku).
- Podporuje:
  - Cesty k souborům (`.png`, `.jpg`, `.svg`, `.webp`, `.pdf`)
  - `matplotlib.figure.Figure`
  - `PIL.Image.Image`
  - `io.BytesIO`
