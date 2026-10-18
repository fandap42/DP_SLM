#!/usr/bin/env python
"""
Skript pro generování prezentací ve fakultní šabloně FIS VŠE z YAML / JSON konfigurace.

Použití:
    python slides/template/generate.py moje_prezentace.yaml -o moje_prezentace.pptx
    python slides/template/generate.py --demo -o slides/demo_prezentace.pptx
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import yaml

# Importujeme lokální builder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from slides.template.builder import FISPresentation


def build_from_dict(config: Dict[str, Any], output_path: str, export_pdf: bool = True):
    """Sestaví prezentaci ze zadaného konfiguračního slovníku."""

    meta = config.get("meta", {})
    author = meta.get("author", "Bc. František Pavlík")
    department = meta.get("department", "Katedra statistiky a pravděpodobnosti")
    email = meta.get("email", "pavf05@vse.cz")
    template_path = meta.get("template_path")

    pres = FISPresentation(
        template_path=template_path,
        author=author,
        department=department,
        email=email,
    )

    slides = config.get("slides", [])
    if not slides:
        print("Varování: Seznam snímků (slides) je prázdný.")

    for s_idx, slide_cfg in enumerate(slides, start=1):
        stype = slide_cfg.get("type", "content").lower()
        title = slide_cfg.get("title", f"Snímek {s_idx}")
        notes = slide_cfg.get("notes")

        if stype == "title":
            pres.add_title_slide(
                title=title,
                subtitle=slide_cfg.get("subtitle"),
                author=slide_cfg.get("author"),
                department=slide_cfg.get("department"),
                notes=notes,
            )

        elif stype == "section":
            pres.add_section_slide(
                title=title,
                subtitle=slide_cfg.get("subtitle"),
                notes=notes,
            )

        elif stype in ("content", "standard"):
            pres.add_content_slide(
                title=title,
                items=slide_cfg.get("bullets") or slide_cfg.get("text"),
                image=slide_cfg.get("image"),
                image_side=slide_cfg.get("image_side", "right"),
                image_caption=slide_cfg.get("caption"),
                notes=notes,
            )

        elif stype in ("two_column", "two_columns", "columns"):
            pres.add_two_column_slide(
                title=title,
                left_items=slide_cfg.get("left_bullets") or slide_cfg.get("left_text"),
                right_items=slide_cfg.get("right_bullets") or slide_cfg.get("right_text"),
                left_image=slide_cfg.get("left_image"),
                right_image=slide_cfg.get("right_image"),
                left_title=slide_cfg.get("left_title"),
                right_title=slide_cfg.get("right_title"),
                notes=notes,
            )

        elif stype in ("image", "figure", "chart", "graf"):
            image_src = slide_cfg.get("image") or slide_cfg.get("file")
            if not image_src:
                raise ValueError(f"Snímek {s_idx} ({title}) typu 'image' musí mít definován 'image'.")
            pres.add_image_slide(
                title=title,
                image=image_src,
                caption=slide_cfg.get("caption"),
                notes=notes,
            )

        elif stype in ("table", "tabulka"):
            headers = slide_cfg.get("headers", [])
            rows = slide_cfg.get("rows", [])
            col_widths = slide_cfg.get("col_widths")
            pres.add_table_slide(
                title=title,
                headers=headers,
                rows=rows,
                col_widths=col_widths,
                notes=notes,
            )

        elif stype == "blank":
            pres.add_blank_slide(title=title, notes=notes)

        elif stype in ("closing", "end", "thanks"):
            pres.add_closing_slide(
                title=title,
                author=slide_cfg.get("author"),
                email=slide_cfg.get("email"),
                notes=notes,
            )

        else:
            print(f"Varování: Neznámý typ snímku '{stype}' u snímku '{title}'. Použije se běžný text.")
            pres.add_content_slide(
                title=title,
                items=slide_cfg.get("bullets") or slide_cfg.get("text"),
                notes=notes,
            )

    # Automatické doplnění závěrečného snímku "Děkuji za pozornost", pokud chybí
    has_closing = False
    if slides:
        last_type = slides[-1].get("type", "").lower()
        last_title = slides[-1].get("title", "").lower()
        if last_type in ("closing", "end", "thanks") or "děkuji" in last_title or "dekuji" in last_title:
            has_closing = True

    auto_closing = meta.get("auto_closing", True)
    if auto_closing and not has_closing:
        pres.add_closing_slide(
            title="Děkuji za pozornost",
            author=author,
            email=email,
        )

    out_file = pres.save(output_path, export_pdf=export_pdf)
    print(f"Prezentace byla vygenerována: {out_file}")
    if export_pdf:
        pdf_file = Path(out_file).with_suffix(".pdf")
        if pdf_file.exists():
            print(f"PDF bylo vygenerováno: {pdf_file}")
    return out_file



def create_demo_presentation(output_path: str, export_pdf: bool = True):
    """Vytvoří ukázkovou prezentaci se všemi typy snímků a demo grafem."""

    import matplotlib.pyplot as plt

    # 1. Vygenerujeme dočasný demo graf pro ukázku
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_loss = [2.45, 1.82, 1.45, 1.21, 1.05, 0.94, 0.85, 0.79, 0.74, 0.70]
    val_loss = [2.50, 1.90, 1.55, 1.35, 1.22, 1.15, 1.10, 1.08, 1.06, 1.05]

    ax.plot(epochs, train_loss, marker="o", color="#002F6C", linewidth=2.5, label="Trénovací ztráta (Train)")
    ax.plot(epochs, val_loss, marker="s", color="#00A9E0", linewidth=2.5, linestyle="--", label="Validační ztráta (Val)")
    ax.set_title("Průběh trénování SLM modelu (Cross-Entropy Loss)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epocha", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="none")
    fig.tight_layout()

    pres = FISPresentation(
        author="Bc. František Pavlík",
        department="Katedra statistiky a pravděpodobnosti",
        email="pavf05@vse.cz",
    )

    # 1. Titulní snímek
    pres.add_title_slide(
        title="Vývoj Small Language Modelu pro češtinu",
        subtitle="Diplomová práce – obhajoba a prezentace výsledků",
        author="Bc. František Pavlík",
        department="Katedra statistiky a pravděpodobnosti (FIS VŠE)",
    )

    # 2. Předěl / Sekce
    pres.add_section_slide("1. Cíle a motivace práce")

    # 3. Standardní obsahový snímek s formátováním a vnořenými odrážkami
    pres.add_content_slide(
        title="Motivace a výzkumné otázky",
        items=[
            "**Problém současných LLM:** Modely jako LLaMA-3 70B mají obrovské nároky na VRAM.",
            "**Potřeba českých SLM:**",
            "  Většina malých modelů (1-3B) má slabou znalost českého jazyka.",
            "  Vysoká fragmentace českých slov při použití běžných anglických tokenizerů.",
            "**Cíl práce:**",
            "  Vytrénovat a jemně doladit kompaktní model optimalizovaný pro češtinu.",
            "  Dosáhnout plynulé inference na běžném spotřebním hardware.",
        ],
    )

    # 4. Text + vložený Matplotlib graf
    pres.add_content_slide(
        title="Průběh trénování a konvergence",
        items=[
            "**Trénovací parametry:**",
            "  - Optimalizátor AdamW s cosine learning rate schedulerem",
            "  - Effective batch size: 128",
            "  - Trénováno na 4x NVIDIA A100 (40GB)",
            "**Pozorování:**",
            "  - Model stabilně konvergoval bez přetrénování.",
            "  - Validační ztráta dosáhla minima v 9. epoše.",
        ],
        image=fig,
        image_side="right",
        image_caption="Obr. 1: Křivka trénovací a validační ztráty v čase",
    )

    # 5. Samostatný velký graf
    pres.add_image_slide(
        title="Detailní analýza ztrátové funkce",
        image=fig,
        caption="Graf 1: Detail vývoje loss křivky během pre-trainingu na českém korpusu",
    )

    # 6. Srovnávací dvousloupcový snímek
    pres.add_two_column_slide(
        title="Srovnání: Velké LLM vs. Náš český SLM",
        left_title="Běžné LLM (7B+)",
        left_items=[
            "Vyžaduje cloudové GPU servery",
            "Inference: ~15-20 tokenů/s na běžném PC",
            "Provozní náklady: stovky USD měsíčně",
            "Data odcházejí mimo lokální síť",
        ],
        right_title="Náš navržený SLM (1.5B)",
        right_items=[
            "**Lokální běh i na běžném notebooku**",
            "**Inference: 60+ tokenů/s** (Apple M2 / RTX)",
            "Nulové licenční a cloudové poplatky",
            "**100% soukromí a bezpečnost dat**",
        ],
    )

    # 7. Tabulka s benchmarkem
    pres.add_table_slide(
        title="Výsledky benchmarku na českých testech",
        headers=["Model", "Parametry", "CZ-MMLU (%)", "CZ-Hellaswag", "Rychlost"],
        rows=[
            ["Qwen 2.5 1.5B (base)", "1.5B", "48.2 %", "54.1 %", "62 tok/s"],
            ["Llama 3.2 1B (base)", "1.2B", "44.6 %", "50.8 %", "71 tok/s"],
            ["Náš vyladěný SLM (Full)", "1.5B", "**59.4 %**", "**63.7 %**", "**58 tok/s**"],
            ["Llama 3.1 8B (referenční)", "8.0B", "63.1 %", "67.5 %", "18 tok/s"],
        ],
    )

    # 8. Předěl
    pres.add_section_slide("2. Závěr a další směřování")

    # 9. Závěrečný snímek
    pres.add_closing_slide(
        title="Děkuji za pozornost",
        author="Bc. František Pavlík",
        email="pavf05@vse.cz",
    )

    plt.close(fig)
    out_file = pres.save(output_path, export_pdf=export_pdf)
    print(f"Ukázková prezentace byla vytvořena: {out_file}")
    if export_pdf:
        pdf_file = Path(out_file).with_suffix(".pdf")
        if pdf_file.exists():
            print(f"PDF bylo vygenerováno: {pdf_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Generátor prezentací FIS VŠE z YAML šablony nebo do demo souboru."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Cesta k YAML konfiguračnímu souboru (např. moje_prezentace.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Výstupní soubor .pptx (výchozí: odvozeno z názvu konfigurace nebo slides/prezentace.pptx)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Vygeneruje ukázkovou prezentaci se všemi funkcemi a grafem.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Zakáže automatické generování doprovodného .pdf souboru.",
    )

    args = parser.parse_args()
    export_pdf = not args.no_pdf

    if args.demo or (not args.config and len(sys.argv) == 1):
        out = args.output or "slides/demo_prezentace.pptx"
        create_demo_presentation(out, export_pdf=export_pdf)
        return

    if not args.config:
        print("Chyba: Zadejte cestu k YAML souboru nebo použijte přepínač --demo.")
        parser.print_help()
        sys.exit(1)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Chyba: Soubor '{cfg_path}' neexistuje.")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    out = args.output
    if not out:
        out = str(cfg_path.with_suffix(".pptx"))

    build_from_dict(data, out, export_pdf=export_pdf)



if __name__ == "__main__":
    main()
