from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from app.config import RedactionPolicy
from app.document.image_extractor import (
    extract_images,
    write_docx_with_replaced_images,
)
from app.document.metadata import clean_core_metadata
from app.document.reader import load_document
from app.synthetic.generator import ConsistentReplacer
from app.text.redactor import redact_logical_paragraphs
from app.vision.image_redactor import redact_image_bytes, ImageRedactionReport


# Progress callback ek function hota hai jo processing ka current status
# UI ya kisi aur caller ko bata sakta hai.
ProgressCallback = Callable[[dict], None]


@dataclass
class PipelineResult:
    # Pipeline complete hone ke baad important summary yahan store hogi.
    input_path: str
    output_path: str
    paragraphs_scanned: int
    tables_found: int
    images_found: int

    # Text aur image redaction ke counts.
    text_redactions_applied: int = 0
    images_modified: int = 0

    # Har processed image ki detailed report.
    image_redaction_reports: List[
        Tuple[str, ImageRedactionReport]
    ] = field(default_factory=list)


def _emit_progress(
    callback: Optional[ProgressCallback],
    stage: str,
    **data
) -> None:
    # Agar callback nahi diya gaya toh kuch nahi karna.
    if callback is None:
        return

    # Current processing stage aur extra information callback ko bhejte hain.
    callback({"stage": stage, **data})


def process_document(
    input_path: str,
    output_path: str,
    policy: RedactionPolicy = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> PipelineResult:

    """
    Pura document redaction pipeline.

    Text PII aur images dono process hote hain.
    """

    # Sabse pehle check karo ki input file actually exist karti hai ya nahi.
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    # Agar user ne policy nahi di, toh default policy use karo.
    policy = policy or RedactionPolicy()

    # UI ko batao ki document load ho raha hai.
    _emit_progress(progress_callback, "loading_document")

    # DOCX ko read karke document ka structured representation milta hai.
    content = load_document(input_path)

    # Document ke saare paragraphs collect karo.
    paragraphs = content.all_paragraphs()

    # Text analysis start hone ki information UI ko bhejo.
    _emit_progress(
        progress_callback,
        "analyzing_text",
        current=0,
        total=len(paragraphs),
    )

    # Same PII ko document mein consistently replace karne ke liye
    # ek replacer object create karte hain.
    replacer = ConsistentReplacer()

    # Paragraphs mein PII detect karke redact karo.
    total_text_redactions = redact_logical_paragraphs(
        paragraphs,
        replacer,
        policy,
        progress_callback=progress_callback,
    )

    # Text redaction ke baad document metadata clean karo.
    _emit_progress(progress_callback, "cleaning_metadata")
    clean_core_metadata(content.document)

    # Output folder exist nahi karta toh automatically create karo.
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Pehle ek temporary DOCX save karenge.
    # Is stage par text redacted hai, lekin images abhi original hain.
    temp_text_redacted_path = str(
        output_path_obj.with_suffix(".text_only.tmp.docx")
    )

    _emit_progress(progress_callback, "saving_document")

    # Text-redacted document ko temporary file mein save karo.
    content.document.save(temp_text_redacted_path)

    try:
        # Temporary DOCX ke andar se saari images extract karo.
        images = extract_images(temp_text_redacted_path)

        # Sirf modified images ko store karne ke liye dictionary.
        image_replacements = {}

        # Har image ki processing report yahan store hogi.
        reports: List[
            Tuple[str, ImageRedactionReport]
        ] = []

        # Kitni images actually modify hui hain.
        images_modified_count = 0

        # Total images ki count.
        total_images = len(images)

        # Ek-ek image ko process karo.
        for image_index, (zip_name, image_bytes) in enumerate(
            images,
            start=1,
        ):
            # UI ko batao ki kaunsi image process ho rahi hai.
            _emit_progress(
                progress_callback,
                "processing_images",
                current=image_index,
                total=total_images,
            )

            # Image ka original extension nikalo.
            # Example: .jpg, .jpeg, .png
            img_ext = Path(zip_name).suffix.lower()

            # Image ke andar PII detect karke pixel-level redaction karo.
            new_bytes, report = redact_image_bytes(
                image_bytes,
                policy,
                img_ext,
            )

            # Image ki processing report save karo.
            reports.append((zip_name, report))

            # Agar image mein actual redaction hui hai,
            # toh new image bytes ko replacement dictionary mein rakho.
            if report.modified:
                image_replacements[zip_name] = new_bytes
                images_modified_count += 1

        # Text aur processed images ko combine karke final DOCX banao.
        _emit_progress(progress_callback, "finalizing")

        write_docx_with_replaced_images(
            temp_text_redacted_path,
            output_path,
            image_replacements,
        )

    finally:
        # Temporary file ko hamesha delete karo.
        # finally ki wajah se error aane par bhi cleanup hoga.
        Path(temp_text_redacted_path).unlink(
            missing_ok=True
        )

    # Puri processing ka result caller ko return karo.
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