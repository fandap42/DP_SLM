"""
Ukázkový Python skript pro tvorbu prezentace ve fakultní šabloně FIS VŠE.

Použití:
    python slides/template/example_script.py
"""

from pathlib import Path
import matplotlib.pyplot as plt

# Import FISPresentation z balíčku šablony
from builder import FISPresentation


def main():
    # 1. Vytvoření instance prezentace s výchozími metadaty
    prs = FISPresentation(
        author="Bc. František Pavlík",
        department="Katedra statistiky a pravděpodobnosti (FIS VŠE)",
        email="pavf05@vse.cz",
    )

    # 2. Úvodní snímek
    prs.add_title_slide(
        title="Vývoj Small Language Modelu pro češtinu",
        subtitle="Diplomová práce – architektura, trénování a evaluace",
    )

    # 3. Předělový snímek sekce
    prs.add_section_slide("1. Motivace a architektura")

    # 4. Standardní obsahový snímek s odrážkami (podporuje **bold**, *italic* a vnoření)
    prs.add_content_slide(
        title="Hlavní motivace a výzkumné otázky",
        items=[
            "**Small Language Models (SLM):** Kompaktní modely s 1 až 3 miliardami parametrů.",
            "**Výzva pro český jazyk:** Většina open-weight modelů má minimální česká data.",
            (
                "Nízká efektivita běžných tokenizerů (až 3-4 tokeny na jedno české slovo)",
                1,
            ),
            ("Ztráta kontextového okna a pomalejší generování", 1),
            "**Cíl projektu:**",
            ("Vyvinout specializovaný BPE tokenizer pro češtinu", 1),
            ("Předtrénovat model na vyčištěném českém korpusu", 1),
        ],
    )

    # 5. Vytvoření grafu pomocí Matplotlib a jeho přímé vložení (není nutné ukládat na disk)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    models = ["Llama 3.2 1B", "Qwen 2.5 1.5B", "Gemma 2 2B", "Náš český SLM"]
    scores = [44.6, 48.2, 51.0, 59.4]
    colors = ["#94A3B8", "#94A3B8", "#94A3B8", "#00A9E0"]

    bars = ax.barh(models, scores, color=colors, height=0.55)
    ax.set_xlim(0, 75)
    ax.set_xlabel("Přesnost na českém MMLU benchmarku (%)", fontsize=10)
    ax.set_title("Srovnání modelů na českých úlohách", fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    for bar, score in zip(bars, scores):
        ax.text(score + 1.2, bar.get_y() + bar.get_height() / 2, f"{score:.1f} %",
                va="center", ha="left", fontsize=9, fontweight="bold" if score > 55 else "normal")

    fig.tight_layout()

    # 6. Snímek s textem vlevo a grafem vpravo
    prs.add_content_slide(
        title="Výsledky na českém benchmarku",
        items=[
            "**Evaluace modelu:**",
            ("Měřeno na sadě českých úloh (CZ-MMLU, Hellaswag, GSM8k)", 1),
            ("Testováno zero-shot i few-shot metodou", 1),
            "**Klíčové zjištění:**",
            ("Náš model překonává základní modely podobné velikosti o více než 8-11 p. b.", 1),
            ("Díky českému tokenizeru je inference o 35 % rychlejší", 1),
        ],
        image=fig,
        image_side="right",
        image_caption="Graf: Úspěšnost na benchmarku CZ-MMLU",
    )

    # 7. Dvousloupcový snímek
    prs.add_two_column_slide(
        title="Srovnání architektur",
        left_title="Standardní LLM",
        left_items=[
            "7B+ parametrů",
            "Nároky na GPU: min. 16-24 GB VRAM",
            "Inference v cloudu",
            "Latence: 15-20 tokenů/s",
        ],
        right_title="Náš optimalizovaný SLM",
        right_items=[
            "**1.5B parametrů**",
            "**Nároky na paměť: pouze 3-4 GB RAM/VRAM**",
            "**Plně lokální běh (edge / notebook)**",
            "**Rychlost: 55-65 tokenů/s**",
        ],
    )

    # 8. Tabulka
    prs.add_table_slide(
        title="Detailní přehled výsledků",
        headers=["Architektura", "Počet parametrů", "CZ-MMLU", "Propustnost"],
        rows=[
            ["Baseline", "1.2B", "44.6 %", "71 tok/s"],
            ["Adaptovaný BPE", "1.5B", "52.1 %", "64 tok/s"],
            ["Náš finální model", "1.5B", "**59.4 %**", "**58 tok/s**"],
        ],
    )

    # 9. Závěrečný snímek
    prs.add_closing_slide(
        title="Děkuji za pozornost",
        author="Bc. František Pavlík",
        email="pavf05@vse.cz",
    )

    plt.close(fig)

    output_file = Path("slides/vysledky_slm.pptx")
    prs.save(output_file)
    print(f"Prezentace PPTX: {output_file.resolve()}")
    print(f"Prezentace PDF:  {output_file.with_suffix('.pdf').resolve()}")



if __name__ == "__main__":
    main()
