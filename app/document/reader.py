"""
Hinglish: Ye file DOCX ko READ karti hai aur ek "logical text view" banati hai.

PROBLEM: Word ek visible paragraph ko internally multiple <w:r> runs mein
tod deta hai (spell-check, formatting changes, revision markers ki wajah se).
Agar hum sirf har run par independently regex chalayenge, to "Rahul Sharma"
jaisa naam "Rah" + "ul " + "Sharma" runs mein split hoke miss ho sakta hai.

SOLUTION: Har paragraph ke sabhi runs ka text jodkar ek "logical string"
banate hain, aur saath hi ek "char_index -> (run_index)" mapping bhi rakhte
hain. Isse detection puri paragraph string par hoti hai (accurate), aur
baad mein detected span ko wapas specific runs par map karke redact kar
sakte hain (writer.py mein).
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import docx
from docx.document import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph


@dataclass
class RunSpan:
    """Hinglish: Ek run ka logical text mein occupy kiya hua range."""
    run_index: int          # paragraph.runs mein is run ka index
    start: int              # logical string mein start offset (inclusive)
    end: int                # logical string mein end offset (exclusive)


@dataclass
class LogicalParagraph:
    """
    Hinglish: Ek paragraph ka reconstructed text + run mapping.
    `paragraph` object ko hi rakhte hain taaki writer.py isi se
    seedha runs modify kar sake (formatting preserve karne ke liye).
    """
    paragraph: Paragraph
    text: str
    run_spans: List[RunSpan]
    location: str            # e.g. "body", "table:2:row1:cell0", "header:3"


def _build_logical_text(paragraph: Paragraph) -> Tuple[str, List[RunSpan]]:
    """
    Hinglish: Paragraph ke runs ko concatenate karke logical text banate hain
    aur har run ka start/end offset record karte hain.

    Empty runs (jinme text hi nahi) ko skip karte hain kyunki unka
    contribution zero-length hota hai aur unnecessary complexity add
    karta hai.
    """
    text_parts = []
    spans: List[RunSpan] = []
    cursor = 0
    for idx, run in enumerate(paragraph.runs):
        run_text = run.text or ""
        if run_text == "":
            continue
        start = cursor
        end = cursor + len(run_text)
        spans.append(RunSpan(run_index=idx, start=start, end=end))
        text_parts.append(run_text)
        cursor = end
    return "".join(text_parts), spans


def iter_logical_paragraphs(paragraphs: List[Paragraph], location_prefix: str) -> List[LogicalParagraph]:
    """Hinglish: Paragraphs ki list se LogicalParagraph objects banate hain."""
    result = []
    for i, p in enumerate(paragraphs):
        text, spans = _build_logical_text(p)
        if text.strip() == "":
            continue
        result.append(LogicalParagraph(
            paragraph=p, text=text, run_spans=spans,
            location=f"{location_prefix}:para{i}",
        ))
    return result


def _iter_table_paragraphs(table: Table, location_prefix: str) -> List[LogicalParagraph]:
    """
    Hinglish: Table ke andar har cell ke paragraphs nikalte hain.
    Nested tables (cell ke andar table) ko bhi recursively handle karte hain
    kyunki prospectus jaisे legal documents mein nested tables common hain.
    """
    result = []
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            loc = f"{location_prefix}:row{r_idx}:cell{c_idx}"
            result.extend(iter_logical_paragraphs(cell.paragraphs, loc))
            # Hinglish: Nested table hone par recursively process karo.
            for nt_idx, nested_table in enumerate(cell.tables):
                result.extend(_iter_table_paragraphs(
                    nested_table, f"{loc}:nested_table{nt_idx}"
                ))
    return result


@dataclass
class DocumentContent:
    """Hinglish: Poore document se nikale gaye sabhi logical paragraphs, grouped by area."""
    document: DocxDocument
    body_paragraphs: List[LogicalParagraph]
    table_paragraphs: List[LogicalParagraph]
    header_paragraphs: List[LogicalParagraph]
    footer_paragraphs: List[LogicalParagraph]

    def all_paragraphs(self) -> List[LogicalParagraph]:
        return (self.body_paragraphs + self.table_paragraphs
                + self.header_paragraphs + self.footer_paragraphs)


def load_document(path: str) -> DocumentContent:
    """
    Hinglish: Entry point - DOCX file load karke sabhi text areas
    (body, tables, headers, footers) se logical paragraphs nikalta hai.

    Input: docx file path
    Output: DocumentContent object jisme document handle + saare paragraphs hain
    """
    try:
        doc = docx.Document(path)
    except Exception as exc:  # Hinglish: corrupted/invalid docx ka case
        raise ValueError(f"Could not open DOCX file '{path}': {exc}") from exc

    body_paragraphs = iter_logical_paragraphs(doc.paragraphs, "body")

    table_paragraphs: List[LogicalParagraph] = []
    for t_idx, table in enumerate(doc.tables):
        table_paragraphs.extend(_iter_table_paragraphs(table, f"table{t_idx}"))

    header_paragraphs: List[LogicalParagraph] = []
    footer_paragraphs: List[LogicalParagraph] = []
    # Hinglish: Har section ka apna header/footer ho sakta hai (is prospectus
    # mein 85 sections hain!). Duplicate header/footer objects se bachne ke
    # liye seen-set use karte hain (multiple sections same header share kar
    # sakte hain "same as previous" ki wajah se).
    seen_header_ids = set()
    seen_footer_ids = set()
    for s_idx, section in enumerate(doc.sections):
        for header_attr, seen_set, bucket, label in [
            (section.header, seen_header_ids, header_paragraphs, "header"),
            (section.footer, seen_footer_ids, footer_paragraphs, "footer"),
        ]:
            part_id = id(header_attr._element)
            if part_id in seen_set:
                continue
            seen_set.add(part_id)
            bucket.extend(iter_logical_paragraphs(
                header_attr.paragraphs, f"{label}{s_idx}"
            ))
            # Hinglish: Header/footer ke andar tables bhi ho sakti hain
            # (masthead layouts mein common hai).
            for t_idx, table in enumerate(header_attr.tables):
                bucket.extend(_iter_table_paragraphs(
                    table, f"{label}{s_idx}:table{t_idx}"
                ))

    return DocumentContent(
        document=doc,
        body_paragraphs=body_paragraphs,
        table_paragraphs=table_paragraphs,
        header_paragraphs=header_paragraphs,
        footer_paragraphs=footer_paragraphs,
    )



