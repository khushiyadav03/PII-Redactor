"""
Hinglish: Ye function poore document ke saare logical paragraphs par
detection + redaction chalata hai. NER ko batch mein chalate hain
(sabhi paragraph texts ek saath spaCy ko dete hain) - isse 1000+
paragraphs wale document par bhi reasonable speed milti hai, kyunki
spaCy ka nlp.pipe() internally efficient batching karta hai.
"""
from typing import List

from app.config import RedactionPolicy
from app.document.reader import LogicalParagraph
from app.document.writer import apply_redactions
from app.synthetic.generator import ConsistentReplacer
from app.text.detector import detect_pii_in_text
from app.text.ner import extract_entities


def redact_logical_paragraphs(
    logical_paragraphs: List[LogicalParagraph],
    replacer: ConsistentReplacer,
    policy: RedactionPolicy,
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
    for lp, entities in zip(logical_paragraphs, all_entities):
        detections = detect_pii_in_text(
            lp.text, replacer, policy, precomputed_entities=entities
        )
        if not detections:
            continue
        spans = [(d.start, d.end, d.replacement) for d in detections]
        apply_redactions(lp, spans)
        total_redactions += len(spans)

    return total_redactions
