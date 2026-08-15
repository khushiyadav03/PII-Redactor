from typing import List, Tuple

from app.document.reader import LogicalParagraph


def apply_redactions(
    logical_para: LogicalParagraph,
    spans: List[Tuple[int, int, str]],
) -> None:

    # Spans ko right-to-left sort karte hain.
    # Isse pehle replacement ki wajah se baad wale offsets disturb nahi hote.
    spans_sorted = sorted(
        spans,
        key=lambda s: s[0],
        reverse=True,
    )

    # Har detected PII span ko process karo.
    for start, end, replacement in spans_sorted:

        # Check karo ki PII span kin Word runs ke andar hai.
        #
        # Condition:
        # run.end > span.start
        # run.start < span.end
        #
        # Dono true hone ka matlab hai ki run aur PII span overlap kar rahe hain.
        overlapping_runs = [
            rs
            for rs in logical_para.run_spans
            if rs.end > start and rs.start < end
        ]

        # Agar kisi run se match nahi hua toh is span ko skip karo.
        if not overlapping_runs:
            continue

        # Pehle overlapping run mein complete replacement jayega.
        first = True

        for rs in overlapping_runs:

            # RunSpan ke run_index se original Word run nikalo.
            run = logical_para.paragraph.runs[rs.run_index]

            # Logical paragraph ke coordinates ko
            # current run ke local coordinates mein convert karo.
            run_local_start = max(
                0,
                start - rs.start,
            )

            run_local_end = min(
                rs.end - rs.start,
                end - rs.start,
            )

            # Current run ka original text.
            original_run_text = run.text

            # PII se pehle wala text preserve karo.
            before = original_run_text[:run_local_start]

            # PII ke baad wala text preserve karo.
            after = original_run_text[run_local_end:]

            if first:
                # Complete replacement sirf first overlapping run mein daalo.
                #
                # Example:
                # "Rah" + "ul " + "Sharma"
                #
                # becomes:
                # "[NAME]" + "" + ""
                run.text = before + replacement + after
                first = False

            else:
                # Baaki overlapping runs se PII text remove kar do.
                run.text = before + after


def redact_paragraph_text(
    logical_para: LogicalParagraph,
    detections: List,
) -> int:

    # Detector ke objects ko simple
    # (start, end, replacement) format mein convert karo.
    spans = [
        (d.start, d.end, d.replacement)
        for d in detections
    ]

    # Actual redaction perform karo.
    apply_redactions(
        logical_para,
        spans,
    )

    # Sirf count return karo, actual PII value nahi.
    return len(spans)