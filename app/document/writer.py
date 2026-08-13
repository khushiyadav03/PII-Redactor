"""
Hinglish: Ye file detection ke baad actual REDACTION karti hai -
logical text mein mile PII spans ko wapas original Word runs par map
karke, un runs ka text replace karti hai.

IMPORTANT DESIGN DECISION:
Ek PII span (jaise "Rahul Sharma") multiple runs mein spread ho sakta hai
("Rah" + "ul " + "Sharma"). Isliye replacement ka poora naya text sirf
PEHLE overlapping run mein daalte hain, aur baaki overlapping runs ka
text EMPTY kar dete hain. Isse:
  1. Formatting (bold/italic/font) pehle run ki preserve hoti hai.
  2. Original text kahin bhi (kisi bhi run mein) reconstruct nahi ho sakta.
  3. Paragraph ka overall visible text sirf ek baar naya replacement dikhata hai.
"""
from typing import List, Tuple
from app.document.reader import LogicalParagraph


def apply_redactions(logical_para: LogicalParagraph, spans: List[Tuple[int, int, str]]) -> None:
    """
    Hinglish: Ek logical paragraph par redaction spans apply karta hai.

    Input:
        logical_para: reader.py se mila LogicalParagraph (paragraph + run mapping)
        spans: list of (start, end, replacement_text) - end-exclusive,
               logical_para.text ke coordinates mein.

    Output: None (paragraph ke runs in-place modify hote hain)

    Edge case: Overlapping spans ko caller pehle hi resolve kar chuka hona
    chahiye (detector.py merge karta hai). Yahan hum non-overlapping assume
    karte hain aur reverse order (right-to-left) mein apply karte hain taaki
    ek span ka replacement dusre span ke offsets ko invalidate na kare.
    """
    # Hinglish: right-to-left process karna zaroori hai kyunki agar hum
    # left-to-right karte aur ek run ka text length badal jaata, to baad
    # ke spans ke stored offsets galat ho jaate. Run-level text set karte
    # hain (poora paragraph text rebuild nahi karte), isliye offsets sirf
    # ek run ke andar hi matter karte hain - lekin phir bhi safest order
    # right-to-left hi hai.
    spans_sorted = sorted(spans, key=lambda s: s[0], reverse=True)

    for start, end, replacement in spans_sorted:
        overlapping_runs = [
            rs for rs in logical_para.run_spans
            if rs.end > start and rs.start < end
        ]
        if not overlapping_runs:
            continue

        first = True
        for rs in overlapping_runs:
            run = logical_para.paragraph.runs[rs.run_index]
            run_local_start = max(0, start - rs.start)
            run_local_end = min(rs.end - rs.start, end - rs.start)
            original_run_text = run.text
            before = original_run_text[:run_local_start]
            after = original_run_text[run_local_end:]

            if first:
                # Hinglish: Replacement pura text sirf pehle overlapping
                # run mein daalte hain.
                run.text = before + replacement + after
                first = False
            else:
                # Hinglish: Baaki runs se sirf PII wala hissa hata dete hain,
                # taaki original text kahin reconstruct na ho sake.
                run.text = before + after


def redact_paragraph_text(logical_para: LogicalParagraph, detections: List) -> int:
    """
    Hinglish: Ek paragraph ke detections (detector.py se) leke unhe
    apply_redactions ke liye (start,end,replacement) tuples mein convert
    karta hai aur apply karta hai.

    Returns: number of redactions applied (for reporting/logging counts,
    NOT the actual PII values - hum raw PII console par print nahi karte,
    privacy requirement ke mutabik).
    """
    spans = [(d.start, d.end, d.replacement) for d in detections]
    apply_redactions(logical_para, spans)
    return len(spans)
