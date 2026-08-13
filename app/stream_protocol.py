"""
/redact streaming response ka lightweight protocol.
Progress + metrics response body mein bhejte hain taaki cross-origin
frontend CORS custom-header limitation se bach sake.
"""
import json
from typing import Any, Dict, Optional, Tuple

from app.pipeline import PipelineResult

FILE_MARKER = b"\n---FILE---\n"
PROGRESS_PREFIX = "PROGRESS:"
METRICS_PREFIX = "METRICS:"
ERROR_PREFIX = "ERROR:"


def pipeline_result_to_metrics(result: PipelineResult) -> Dict[str, int]:
    """
    Pipeline ke existing counters — koi alag metrics system nahi.

    text_redactions_applied (PII Redactions):
        Text spans jahan detect+replace/mask actually apply hue.
    images_modified (Images Protected):
        Embedded images jinke pixels actually modify hue (scan-only count nahi).
    paragraphs_scanned / tables_found / images_found:
        Document scan stats — detection counts nahi.
    """
    return {
        "text_redactions_applied": result.text_redactions_applied,
        "images_modified": result.images_modified,
        "paragraphs_scanned": result.paragraphs_scanned,
        "tables_found": result.tables_found,
        "images_found": result.images_found,
    }


def encode_progress(stage: str, **data: Any) -> bytes:
    payload = {"stage": stage, **data}
    return f"{PROGRESS_PREFIX}{json.dumps(payload, separators=(',', ':'))}\n".encode("utf-8")


def encode_metrics(result: PipelineResult) -> bytes:
    # Hinglish: Frontend ko hardcoded 0 dikhane ke bajaye backend ke actual
    # pipeline counters use kar rahe hain, taaki UI real processing result reflect kare.
    return f"{METRICS_PREFIX}{json.dumps(pipeline_result_to_metrics(result), separators=(',', ':'))}\n".encode("utf-8")


def encode_error(detail: str) -> bytes:
    return f"{ERROR_PREFIX}{json.dumps({'detail': detail}, separators=(',', ':'))}\n".encode("utf-8")


def parse_stream_response(raw: bytes) -> Tuple[Optional[Dict[str, int]], bytes, Optional[str]]:
    """
    Test helper — stream bytes se metrics dict, DOCX bytes, aur error parse karta hai.
    Returns: (metrics_or_none, docx_bytes, error_detail_or_none)
    """
    marker_idx = raw.find(FILE_MARKER)
    if marker_idx == -1:
        text_part = raw.decode("utf-8", errors="replace")
        if ERROR_PREFIX in text_part:
            for line in text_part.splitlines():
                if line.startswith(ERROR_PREFIX):
                    err = json.loads(line[len(ERROR_PREFIX):])
                    return None, b"", err.get("detail", "Unknown error")
        return None, b"", "Stream missing file marker"

    text_part = raw[:marker_idx].decode("utf-8", errors="replace")
    docx_bytes = raw[marker_idx + len(FILE_MARKER):]

    metrics: Optional[Dict[str, int]] = None
    error_detail: Optional[str] = None
    for line in text_part.splitlines():
        if line.startswith(METRICS_PREFIX):
            metrics = json.loads(line[len(METRICS_PREFIX):])
        elif line.startswith(ERROR_PREFIX):
            err = json.loads(line[len(ERROR_PREFIX):])
            error_detail = err.get("detail", "Unknown error")

    return metrics, docx_bytes, error_detail
