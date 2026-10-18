"""
FISPresentation - Generátor prezentací dle fakultní šablony VŠE FIS.

Podporuje:
- Vkládání textu s odrážkami (včetně více úrovní a formátování **bold**, *italic*)
- Vkládání obrázků a grafů (soubory, PIL Image, matplotlib Figure) s automatickým škálováním a centrováním
- Dvousloupcové rozvržení (text + text, text + obrázek, obrázek + obrázek)
- Srovnávací snímky a tabulky
- Úvodní, sekční (předělové) a závěrečné snímky
"""

import io
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image

try:
    import matplotlib.figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "FIS_CZ_16_9.pptx"

# Výchozí fakultní barvy (FIS VŠE)
FIS_BLUE = RGBColor(0, 47, 108)      # Tmavě modrá FIS
FIS_LIGHT_BLUE = RGBColor(0, 169, 224) # Světle modrá akcent
TEXT_DARK = RGBColor(30, 41, 59)     # Tmavý text


def _parse_markdown_runs(paragraph, text: str):
    """
    Rozparsuje jednoduchý Markdown (**bold**, *italic*, `code`)
    a vloží formátované runy do zadaného odstavce.
    """
    # Regex pro **bold**, *italic*, `code`
    token_pattern = re.compile(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)')
    tokens = token_pattern.split(text)

    for token in tokens:
        if not token:
            continue
        run = paragraph.add_run()
        if token.startswith('**') and token.endswith('**') and len(token) >= 4:
            run.text = token[2:-2]
            run.font.bold = True
        elif token.startswith('*') and token.endswith('*') and len(token) >= 2:
            run.text = token[1:-1]
            run.font.italic = True
        elif token.startswith('`') and token.endswith('`') and len(token) >= 2:
            run.text = token[1:-1]
            run.font.name = "Consolas"
        else:
            run.text = token


def _prepare_image_source(image_input: Any) -> Tuple[Union[str, io.BytesIO], Tuple[int, int]]:
    """
    Převede vstup obrázku (soubor, Path, PIL Image, Matplotlib Figure, BytesIO)
    na formát vhodný pro python-pptx a vrátí (image_stream_or_path, (width_px, height_px)).
    """
    if HAS_MATPLOTLIB and isinstance(image_input, matplotlib.figure.Figure):
        buf = io.BytesIO()
        image_input.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        im = Image.open(buf)
        size = im.size
        buf.seek(0)
        return buf, size

    if isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format="PNG")
        buf.seek(0)
        return buf, image_input.size

    if isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        im = Image.open(image_input)
        size = im.size
        image_input.seek(0)
        return image_input, size

    path = str(image_input)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Obrázek nebyl nalezen: {path}")

    with Image.open(path) as im:
        size = im.size
    return path, size


class FISPresentation:
    """
    Třída pro snadné vytváření prezentací ve fakultní šabloně FIS VŠE 16:9.
    """

    def __init__(
        self,
        template_path: Optional[Union[str, Path]] = None,
        author: str = "",
        department: str = "Katedra informačních technologií",
        email: str = "",
    ):
        path = Path(template_path) if template_path else DEFAULT_TEMPLATE_PATH
        if not path.exists():
            raise FileNotFoundError(f"Šablona nebyla nalezena na cestě: {path}")

        self.prs = pptx.Presentation(str(path))
        self.default_author = author
        self.default_department = department
        self.default_email = email

        # Zmapujeme layouty
        # Master 0: Úvodní snímek
        self.master_title = self.prs.slide_masters[0].slide_layouts[0]
        # Master 1: Předěl / Závěr (tmavší modré pozadí)
        self.master_section = self.prs.slide_masters[1].slide_layouts[0]
        # Master 2: Standardní obsahové layouty
        m2_layouts = self.prs.slide_masters[2].slide_layouts
        self.layout_title_and_content = m2_layouts[0]   # Nadpis a obsah
        self.layout_two_content = m2_layouts[3]         # Dva obsahy
        self.layout_comparison = m2_layouts[4]          # Porovnání
        self.layout_title_only = m2_layouts[5]          # Jenom nadpis
        self.layout_blank = m2_layouts[6]               # Prázdný
        self.layout_picture_caption = m2_layouts[8]     # Obrázek s titulkem

    def add_title_slide(
        self,
        title: str,
        subtitle: Optional[str] = None,
        author: Optional[str] = None,
        department: Optional[str] = None,
    ):
        """Vytvoří úvodní titulní snímek prezentace."""
        slide = self.prs.slides.add_slide(self.master_title)

        for ph in slide.placeholders:
            idx = ph.placeholder_format.idx
            if idx == 0:  # Nadpis
                ph.text = title
            elif idx == 1 and subtitle:  # Podnadpis
                ph.text = subtitle
            elif idx == 13:  # Autor
                ph.text = author if author is not None else self.default_author
            elif idx == 14:  # Katedra / oddělení
                ph.text = department if department is not None else self.default_department

        return slide

    def add_section_slide(self, title: str, subtitle: Optional[str] = None):
        """Vytvoří předělový snímek oddílu (mezititulek na modrém pozadí)."""
        slide = self.prs.slides.add_slide(self.master_section)

        for ph in slide.placeholders:
            idx = ph.placeholder_format.idx
            if idx == 0:
                ph.text = title
            elif idx == 1 and subtitle:
                ph.text = subtitle
            elif idx in (15, 16):
                # Vyprázdnit nepotřebné zástupné texty z masteru
                ph.text = ""

        return slide

    def add_closing_slide(
        self,
        title: str = "Děkuji za pozornost",
        author: Optional[str] = None,
        email: Optional[str] = None,
    ):
        """Vytvoří závěrečný snímek s poděkováním a kontaktem."""
        slide = self.prs.slides.add_slide(self.master_section)

        auth = author if author is not None else self.default_author
        mail = email if email is not None else self.default_email

        for ph in slide.placeholders:
            idx = ph.placeholder_format.idx
            if idx == 0:
                ph.text = title
            elif idx == 1:
                ph.text = ""
            elif idx == 15:
                ph.text = auth
            elif idx == 16:
                ph.text = mail

        return slide

    def _fill_text_frame(self, tf, items: Union[str, Sequence[Any]]):
        """Pomocná funkce pro naplnění textového rámce odrážkami s podporou úrovní a Markdownu."""
        tf.text = ""  # Vyčistit výchozí text

        if isinstance(items, str):
            items = [line for line in items.split("\n") if line.strip()]

        def flatten_items(raw_items, current_level=0):
            flat = []
            for item in raw_items:
                if isinstance(item, str):
                    stripped = item.lstrip()
                    indent = len(item) - len(stripped)
                    lvl = current_level + min(indent // 2, 4)
                    flat.append((stripped.lstrip("-*• "), lvl))
                elif isinstance(item, tuple) and len(item) >= 2:
                    flat.append((str(item[0]), int(item[1])))
                elif isinstance(item, dict):
                    if "text" in item:
                        txt = str(item.get("text", ""))
                        lvl = int(item.get("level", current_level))
                        flat.append((txt, lvl))
                        subs = item.get("items") or item.get("subitems") or item.get("sub") or []
                        if subs:
                            flat.extend(flatten_items(subs, current_level=lvl + 1))
                    else:
                        # Mapování typu {"Hlavní bod": ["Podbod 1", "Podbod 2"]}
                        for k, v in item.items():
                            flat.append((str(k), current_level))
                            if isinstance(v, (list, tuple)):
                                flat.extend(flatten_items(v, current_level=current_level + 1))
                            elif isinstance(v, str):
                                flat.append((str(v), current_level + 1))
                else:
                    flat.append((str(item), current_level))
            return flat

        flattened = flatten_items(items)
        first = True
        for text, level in flattened:
            if not text.strip():
                continue
            if first:
                p = tf.paragraphs[0]
                first = False
            else:
                p = tf.add_paragraph()

            p.level = min(max(level, 0), 4)
            _parse_markdown_runs(p, text)


    def _insert_fitted_image(
        self,
        slide,
        image_input: Any,
        left: Inches,
        top: Inches,
        max_width: Inches,
        max_height: Inches,
    ):
        """Vloží obrázek a proporčně ho vycentruje do zadaného obdélníku."""
        img_src, (img_w, img_h) = _prepare_image_source(image_input)
        aspect = img_w / img_h
        box_aspect = max_width / max_height

        if aspect > box_aspect:
            width = max_width
            height = max_width / aspect
        else:
            height = max_height
            width = max_height * aspect

        offset_x = left + (max_width - width) / 2
        offset_y = top + (max_height - height) / 2

        return slide.shapes.add_picture(img_src, offset_x, offset_y, width=width, height=height)

    def add_content_slide(
        self,
        title: str,
        items: Optional[Union[str, Sequence[Any]]] = None,
        image: Optional[Any] = None,
        image_side: str = "right",
        image_caption: Optional[str] = None,
    ):
        """
        Vytvoří standardní obsahový snímek.
        - Pokud je zadán pouze text/odrážky: celá šířka snímku.
        - Pokud je zadán i obrázek: rozdělí snímek na text a obrázek (vlevo/vpravo).
        - Pokud je zadán pouze obrázek: vycentruje velký obrázek přes obsahovou část.
        """
        # Případ 1: Pouze obrázek bez textu
        if image and not items:
            return self.add_image_slide(title=title, image=image, caption=image_caption)

        # Případ 2: Pouze text/odrážky
        if items and not image:
            slide = self.prs.slides.add_slide(self.layout_title_and_content)
            for ph in slide.placeholders:
                if ph.placeholder_format.idx == 0:
                    ph.text = title
                elif ph.placeholder_format.idx == 1:
                    self._fill_text_frame(ph.text_frame, items)
            return slide

        # Případ 3: Text i obrázek dohromady (dvousloupcový layout)
        slide = self.prs.slides.add_slide(self.layout_two_content)
        for ph in slide.placeholders:
            if ph.placeholder_format.idx == 0:
                ph.text = title

        # Placeholder 1 (levý), Placeholder 2 (pravý)
        ph_left = slide.placeholders[1]
        ph_right = slide.placeholders[2]

        if image_side == "right":
            text_ph = ph_left
            img_box = (ph_right.left, ph_right.top, ph_right.width, ph_right.height)
            # Odebereme pravý placeholder, abychom do něj čistě vložili obrázek
            sp = ph_right._element
            sp.getparent().remove(sp)
        else:
            text_ph = ph_right
            img_box = (ph_left.left, ph_left.top, ph_left.width, ph_left.height)
            sp = ph_left._element
            sp.getparent().remove(sp)

        self._fill_text_frame(text_ph.text_frame, items)
        self._insert_fitted_image(slide, image, *img_box)

        if image_caption:
            self._add_caption(slide, image_caption, img_box[0], img_box[1] + img_box[3], img_box[2])

        return slide

    def add_two_column_slide(
        self,
        title: str,
        left_items: Optional[Union[str, Sequence[Any]]] = None,
        right_items: Optional[Union[str, Sequence[Any]]] = None,
        left_image: Optional[Any] = None,
        right_image: Optional[Any] = None,
        left_title: Optional[str] = None,
        right_title: Optional[str] = None,
    ):
        """Vytvoří dvousloupcový snímek s nezávislým obsahem (text, odrážky nebo obrázky)."""
        if left_title or right_title:
            slide = self.prs.slides.add_slide(self.layout_comparison)
            for ph in slide.placeholders:
                idx = ph.placeholder_format.idx
                if idx == 0:
                    ph.text = title
                elif idx == 1 and left_title:
                    ph.text = left_title
                elif idx == 2 and left_items:
                    self._fill_text_frame(ph.text_frame, left_items)
                elif idx == 3 and right_title:
                    ph.text = right_title
                elif idx == 4 and right_items:
                    self._fill_text_frame(ph.text_frame, right_items)
            return slide

        slide = self.prs.slides.add_slide(self.layout_two_content)
        for ph in slide.placeholders:
            if ph.placeholder_format.idx == 0:
                ph.text = title

        ph_left = slide.placeholders[1]
        ph_right = slide.placeholders[2]

        if left_image:
            img_box = (ph_left.left, ph_left.top, ph_left.width, ph_left.height)
            sp = ph_left._element
            sp.getparent().remove(sp)
            self._insert_fitted_image(slide, left_image, *img_box)
        elif left_items:
            self._fill_text_frame(ph_left.text_frame, left_items)

        if right_image:
            img_box = (ph_right.left, ph_right.top, ph_right.width, ph_right.height)
            sp = ph_right._element
            sp.getparent().remove(sp)
            self._insert_fitted_image(slide, right_image, *img_box)
        elif right_items:
            self._fill_text_frame(ph_right.text_frame, right_items)

        return slide

    def add_image_slide(
        self,
        title: str,
        image: Any,
        caption: Optional[str] = None,
    ):
        """Vloží velký obrázek nebo graf na střed snímku s volitelným popiskem."""
        slide = self.prs.slides.add_slide(self.layout_title_only)
        for ph in slide.placeholders:
            if ph.placeholder_format.idx == 0:
                ph.text = title

        # Využitelná plocha: šířka ~11.5", výška ~4.8"
        left = Inches(0.92)
        top = Inches(2.0)
        max_w = Inches(11.5)
        max_h = Inches(4.5 if caption else 4.9)

        pic = self._insert_fitted_image(slide, image, left, top, max_w, max_h)

        if caption:
            # Umístění popisku pod obrázek
            cap_top = pic.top + pic.height + Inches(0.08)
            self._add_caption(slide, caption, pic.left, cap_top, pic.width)

        return slide

    def _add_caption(self, slide, text: str, left, top, width):
        """Pomocná metoda pro přidání popisku obrázku."""
        txBox = slide.shapes.add_textbox(left, top, width, Inches(0.4))
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = text
        run.font.size = Pt(11)
        run.font.italic = True
        run.font.color.rgb = RGBColor(100, 116, 139)

    def add_table_slide(
        self,
        title: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
        col_widths: Optional[Sequence[float]] = None,
    ):
        """Vloží tabulku s daty a hlavičkou."""
        slide = self.prs.slides.add_slide(self.layout_title_only)
        for ph in slide.placeholders:
            if ph.placeholder_format.idx == 0:
                ph.text = title

        num_rows = len(rows) + 1
        num_cols = len(headers)

        table_left = Inches(0.92)
        table_top = Inches(2.1)
        table_width = Inches(11.5)
        table_height = Inches(0.5 * num_rows)

        shape = slide.shapes.add_table(num_rows, num_cols, table_left, table_top, table_width, table_height)
        table = shape.table

        # Nastavení šířek sloupců
        if col_widths and len(col_widths) == num_cols:
            for c_idx, w in enumerate(col_widths):
                table.columns[c_idx].width = Inches(w)

        # Hlavička
        for c_idx, header_text in enumerate(headers):
            cell = table.cell(0, c_idx)
            cell.text = str(header_text)
            p = cell.text_frame.paragraphs[0]
            p.font.bold = True
            p.font.size = Pt(13)
            p.alignment = PP_ALIGN.CENTER
            p.font.color.rgb = RGBColor(255, 255, 255)
            cell.fill.solid()
            cell.fill.fore_color.rgb = FIS_BLUE

        # Řádky s daty
        for r_idx, row_data in enumerate(rows, start=1):
            for c_idx, val in enumerate(row_data):
                cell = table.cell(r_idx, c_idx)
                cell.text = str(val)
                p = cell.text_frame.paragraphs[0]
                p.font.size = Pt(12)
                p.alignment = PP_ALIGN.CENTER if isinstance(val, (int, float)) else PP_ALIGN.LEFT
                cell.fill.solid()
                if r_idx % 2 == 0:
                    cell.fill.fore_color.rgb = RGBColor(241, 245, 249)
                else:
                    cell.fill.fore_color.rgb = RGBColor(255, 255, 255)

        return slide

    def add_blank_slide(self, title: Optional[str] = None):
        """Vytvoří prázdný snímek (volitelně s nadpisem)."""
        slide = self.prs.slides.add_slide(self.layout_blank)
        if title:
            for ph in slide.placeholders:
                if ph.placeholder_format.idx == 0:
                    ph.text = title
        return slide

    def save(self, output_path: Union[str, Path], export_pdf: bool = True) -> Path:
        """
        Uloží prezentaci do zadaného souboru .pptx a volitelně vyexportuje i .pdf.
        
        Args:
            output_path: Cesta k výstupnímu souboru .pptx.
            export_pdf: Pokud je True (výchozí), automaticky vytvoří i stejnojmenné .pdf.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(out))
        if export_pdf:
            try:
                export_to_pdf(out)
            except Exception as e:
                print(f"Upozornění: Export do PDF selhal: {e}")
        return out


def export_to_pdf(pptx_path: Union[str, Path], pdf_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Převede .pptx prezentaci na .pdf pomocí PowerPoint COM (na Windows)
    nebo LibreOffice (headless).
    """
    pptx_path = Path(pptx_path).resolve()
    if not pptx_path.exists():
        raise FileNotFoundError(f"PPTX soubor nebyl nalezen: {pptx_path}")

    if pdf_path is None:
        pdf_path = pptx_path.with_suffix(".pdf")
    else:
        pdf_path = Path(pdf_path).resolve()

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Windows: Použít PowerPoint přes PowerShell COM skript
    ps1_script = Path(__file__).resolve().parent / "export_pdf.ps1"
    if sys.platform == "win32" and ps1_script.exists():
        cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(ps1_script),
            "-InputPptx", str(pptx_path),
            "-OutputPdf", str(pdf_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and pdf_path.exists():
            return pdf_path

    # 2. Alternativní fallback: LibreOffice (soffice)
    try:
        res = subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf", str(pptx_path), "--outdir", str(pdf_path.parent)],
            capture_output=True, text=True
        )
        if pdf_path.exists():
            return pdf_path
    except FileNotFoundError:
        pass

    raise RuntimeError(
        f"Nepodařilo se vyexportovat PDF z {pptx_path}. "
        "Ujistěte se, že je nainstalován Microsoft PowerPoint."
    )

