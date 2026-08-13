"""
Hinglish: Ye pipeline ka central orchestrator hai. CLI (run_redaction.py)
aur FastAPI (agar banaya) dono isi `process_document()` function ko call
karenge - logic kahin duplicate nahi hoga (assignment requirement #23).

PHASE STATUS:
  Phase 1: DOCX read/write roundtrip.                             DONE
  Phase 2/3: text regex + NER + context-rule redaction.           DONE
  Phase 4/5: image OCR/face/ID/QR pixel-level redaction.          DONE
  Phase 6: metadata cleanup.                                      DONE (core props only)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from app.config import RedactionPolicy
from app.document.image_extractor import extract_images, write_docx_with_replaced_images
from app.document.metadata import clean_core_metadata
from app.document.reader import load_document
from app.synthetic.generator import ConsistentReplacer
from app.text.redactor import redact_logical_paragraphs
from app.vision.image_redactor import redact_image_bytes, ImageRedactionReport


@dataclass
class PipelineResult:
    input_path: str
    output_path: str
    paragraphs_scanned: int
    tables_found: int
    images_found: int
    text_redactions_applied: int = 0  # Hinglish: counts only, kabhi bhi raw PII value nahi
    images_modified: int = 0
    image_redaction_reports: List[Tuple[str, ImageRedactionReport]] = field(default_factory=list)


def process_document(input_path: str, output_path: str, policy: RedactionPolicy = None) -> PipelineResult:
    """
    Hinglish: Full pipeline (text + image):
      1. Text: document load -> detect+redact PII (consistent replacements)
         -> metadata clean -> temp save
      2. Images: temp docx ke images extract karo -> har image OCR/face/
         ID/QR redact karo -> naya docx banao jisme images replaced hain
         (irreversible pixel-level redaction)
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    policy = policy or RedactionPolicy()
    content = load_document(input_path)

    replacer = ConsistentReplacer()
    total_text_redactions = redact_logical_paragraphs(
        content.all_paragraphs(), replacer, policy
    )

    clean_core_metadata(content.document)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Hinglish: Step 1 output ek temp file mein save karte hain (text-redacted,
    # images abhi original hain) - phir isi file ke images replace karke
    # final output banate hain.
    temp_text_redacted_path = str(output_path_obj.with_suffix(".text_only.tmp.docx"))
    content.document.save(temp_text_redacted_path)

    try:
        images = extract_images(temp_text_redacted_path)
        image_replacements = {}
        reports: List[Tuple[str, ImageRedactionReport]] = []
        images_modified_count = 0

        for zip_name, image_bytes in images:
            # Hinglish: Original image format nikal kar pass karte hain (e.g. .jpeg)
            img_ext = Path(zip_name).suffix.lower()
            new_bytes, report = redact_image_bytes(image_bytes, policy, img_ext)
            reports.append((zip_name, report))
            if report.modified:
                image_replacements[zip_name] = new_bytes
                images_modified_count += 1

        write_docx_with_replaced_images(temp_text_redacted_path, output_path, image_replacements)
    finally:
        # Hinglish: temp file cleanup - permanently store nahi karte (privacy requirement #28)
        # Hinglish: Exception aane par bhi clean-up call guarantee hai
        Path(temp_text_redacted_path).unlink(missing_ok=True)

    return PipelineResult(
        input_path=input_path,
        output_path=output_path,
        paragraphs_scanned=len(content.all_paragraphs()),
        tables_found=len(content.document.tables),
        images_found=len(images),
        text_redactions_applied=total_text_redactions,
        images_modified=images_modified_count,
        image_redaction_reports=reports,
    )
