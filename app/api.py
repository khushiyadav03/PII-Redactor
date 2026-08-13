"""
Hinglish: Minimal FastAPI API for PII Redactor.
Endpoints:
  GET /health - health check
  POST /redact - upload DOCX file, returns streamed progress + metrics + redacted DOCX
  GET / - serves frontend
"""
import os
import queue
import tempfile
import logging
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import process_document
from app.config import RedactionPolicy
from app.stream_protocol import (
    FILE_MARKER,
    encode_error,
    encode_metrics,
    encode_progress,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="PII Redactor API")

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
STREAM_MEDIA_TYPE = "application/x-pii-redactor-stream"
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"


def cleanup_file(path: str):
    """Hinglish: Temporary files ko request completion ke baad delete karta hai (privacy requirement)."""
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.info(f"Cleaned up temp file: {path}")
    except Exception as e:
        logger.error(f"Error cleaning up temp file {path}: {e}")


def _stream_redaction(temp_in_path: str, temp_out_path: str, policy: RedactionPolicy):
    """
    Hinglish: Pipeline ko background thread mein chala kar real progress events
    stream karta hai, phir existing PipelineResult counters + valid DOCX bytes bhejta hai.
    """
    progress_queue: queue.Queue = queue.Queue()
    result_holder: dict = {}
    error_holder: dict = {}

    def progress_callback(payload: dict):
        progress_queue.put({"type": "progress", "payload": payload})

    def run_pipeline():
        try:
            result_holder["result"] = process_document(
                temp_in_path,
                temp_out_path,
                policy,
                progress_callback=progress_callback,
            )
            progress_queue.put({"type": "done"})
        except Exception as exc:
            error_holder["error"] = exc
            progress_queue.put({"type": "error"})

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    while True:
        item = progress_queue.get()
        if item["type"] == "progress":
            payload = item["payload"]
            yield encode_progress(payload["stage"], **{
                k: v for k, v in payload.items() if k != "stage"
            })
        elif item["type"] == "error":
            err = error_holder.get("error")
            if isinstance(err, ValueError):
                detail = str(err)
            else:
                detail = "An error occurred during document redaction processing."
                logger.error(f"Internal processing failure: {err}")
            yield encode_error(detail)
            cleanup_file(temp_in_path)
            cleanup_file(temp_out_path)
            return
        elif item["type"] == "done":
            break

    thread.join()

    result = result_holder.get("result")
    if result is None:
        yield encode_error("Sanitized output could not be generated.")
        cleanup_file(temp_in_path)
        cleanup_file(temp_out_path)
        return

    if not os.path.exists(temp_out_path) or os.path.getsize(temp_out_path) == 0:
        yield encode_error("Sanitized output could not be generated.")
        cleanup_file(temp_in_path)
        cleanup_file(temp_out_path)
        return

    yield encode_metrics(result)
    yield FILE_MARKER
    with open(temp_out_path, "rb") as output_file:
        while chunk := output_file.read(65536):
            yield chunk

    cleanup_file(temp_in_path)
    cleanup_file(temp_out_path)


@app.get("/health")
def health_check():
    """Hinglish: Simple health check status API (bina kisi secret/PII info leaks ke)."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def get_frontend():
    """Hinglish: Root pathway par index.html file serve karta hai."""
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/redact")
async def redact_file(
    file: UploadFile = File(...),
    redact_names: bool = Form(True),
    redact_emails: bool = Form(True),
    redact_phones: bool = Form(True),
    redact_companies: bool = Form(True),
    redact_addresses: bool = Form(True),
    redact_ssn: bool = Form(True),
    redact_credit_card: bool = Form(True),
    redact_dob: bool = Form(True),
    redact_ip: bool = Form(True),
    redact_pan: bool = Form(True),
    redact_aadhaar: bool = Form(True),
    redact_passport: bool = Form(True),
    redact_faces: bool = Form(True),
    redact_id_documents: bool = Form(True),
    redact_qr_on_id: bool = Form(True),
    redact_signatures_on_id: bool = Form(True),
):
    """
    Hinglish: Multi-part upload endpoint (DOCX validation + processing + cleanup).
    Constraints: reject empty/oversized/non-docx files, ensure fail-closed image errors,
    and prevent PII leaks in logs.
    """
    filename = file.filename or "document.docx"
    if not filename.lower().endswith(".docx"):
        logger.warning(f"Rejected upload with invalid extension: {filename}")
        raise HTTPException(status_code=400, detail="Only DOCX files are allowed.")

    # Hinglish: temporary directory cleanup aur safe naming block
    fd_in, temp_in_path = tempfile.mkstemp(suffix=".docx")
    fd_out, temp_out_path = tempfile.mkstemp(suffix=".docx")
    os.close(fd_in)
    os.close(fd_out)

    try:
        size = 0
        with open(temp_in_path, "wb") as f:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    logger.warning(f"File upload rejected: size exceeded 15MB limit.")
                    raise HTTPException(status_code=413, detail="File too large. Maximum size is 15MB.")
                f.write(chunk)

        if size == 0:
            logger.warning("File upload rejected: file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        policy = RedactionPolicy(
            redact_names=redact_names,
            redact_emails=redact_emails,
            redact_phones=redact_phones,
            redact_companies=redact_companies,
            redact_addresses=redact_addresses,
            redact_ssn=redact_ssn,
            redact_credit_card=redact_credit_card,
            redact_dob=redact_dob,
            redact_ip=redact_ip,
            redact_pan=redact_pan,
            redact_aadhaar=redact_aadhaar,
            redact_passport=redact_passport,
            redact_faces=redact_faces,
            redact_id_documents=redact_id_documents,
            redact_qr_on_id=redact_qr_on_id,
            redact_signatures_on_id=redact_signatures_on_id,
        )

        safe_out_name = f"redacted_{Path(filename).name}"
        headers = {
            "Content-Disposition": f'attachment; filename="{safe_out_name}"',
            "Cache-Control": "no-store",
        }
        return StreamingResponse(
            _stream_redaction(temp_in_path, temp_out_path, policy),
            media_type=STREAM_MEDIA_TYPE,
            headers=headers,
        )

    except HTTPException:
        cleanup_file(temp_in_path)
        cleanup_file(temp_out_path)
        raise
    except Exception as exc:
        cleanup_file(temp_in_path)
        cleanup_file(temp_out_path)
        logger.error(f"Unexpected endpoint failure: {exc}")
        raise HTTPException(status_code=500, detail="An unexpected system error occurred.")
