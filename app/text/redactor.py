"""
Hinglish: Ye function poore document ke saare logical paragraphs par
detection + redaction chalata hai. NER ko batch mein chalate hain
(sabhi paragraph texts ek saath spaCy ko dete hain) - isse 1000+
paragraphs wale document par bhi reasonable speed milti hai, kyunki
spaCy ka nlp.pipe() internally efficient batching karta hai.
"""
from typing import Callable, List, Optional

from app.config import RedactionPolicy
from app.document.reader import LogicalParagraph
from app.document.writer import apply_redactions
from app.synthetic.generator import ConsistentReplacer
from app.text.detector import detect_pii_in_text
from app.text.ner import extract_entities

ProgressCallback = Callable[[dict], None]


def redact_logical_paragraphs(
    logical_paragraphs: List[LogicalParagraph],
    replacer: ConsistentReplacer,
    policy: RedactionPolicy,
    progress_callback: Optional[ProgressCallback] = None,
) -> int:
    """
    Hinglish: Input logical paragraphs ki list par redaction karta hai
    (in-place, underlying python-docx runs modify hote hain).

    Returns: total number of redactions applied (count only, no raw values).
    """
    if not logical_paragraphs:
        return 0

    # Hinglish: Batch NER - sabhi paragraph texts ek saath process karte
    # hain, taaki spaCy ki per-call overhead baar baar na lage.
    texts = [lp.text for lp in logical_paragraphs]
    all_entities = extract_entities(texts)

    total_redactions = 0
    total_paragraphs = len(logical_paragraphs)
    # Hinglish: ~20 updates max on large docs — real paragraph index, fake timer nahi.
    report_interval = max(1, total_paragraphs // 20) if total_paragraphs > 20 else 1

    for index, (lp, entities) in enumerate(zip(logical_paragraphs, all_entities), start=1):
        detections = detect_pii_in_text(
            lp.text, replacer, policy, precomputed_entities=entities
        )
        if detections:
            spans = [(d.start, d.end, d.replacement) for d in detections]
            apply_redactions(lp, spans)
            total_redactions += len(spans)

        if progress_callback and (
            index == 1
            or index == total_paragraphs
            or index % report_interval == 0
        ):
            progress_callback(
                {"stage": "analyzing_text", "current": index, "total": total_paragraphs}
            )

    return total_redactions
