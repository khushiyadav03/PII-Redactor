"""
Hinglish: Minimal FastAPI API for PII Redactor.
Endpoints:
  GET /health - health check
  POST /redact - upload DOCX file, returns redacted DOCX
  GET / - serves frontend
"""
import os
import shutil
import tempfile
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import process_document
from app.config import RedactionPolicy

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
)

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
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
async def redact_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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

        # Process the document
        try:
            process_document(temp_in_path, temp_out_path)
        except ValueError as val_err:
            # Hinglish: Malformed DOCX files ya unprocessable images par clear error
            err_msg = str(val_err)
            logger.warning(f"Processing validation failure: {err_msg}")
            raise HTTPException(status_code=422, detail=err_msg)
        except Exception as exc:
            # Hinglish: System levels error - raw Python traceback client ko nahi bhejte security/privacy ke liye
            logger.error(f"Internal processing failure: {exc}")
            raise HTTPException(status_code=500, detail="An error occurred during document redaction processing.")

        if not os.path.exists(temp_out_path) or os.path.getsize(temp_out_path) == 0:
            logger.error("Redacted output file not generated or empty.")
            raise HTTPException(status_code=500, detail="Sanitized output could not be generated.")

        # Hinglish: response return karne ke baad clean-up background task schedule karte hain
        background_tasks.add_task(cleanup_file, temp_in_path)
        background_tasks.add_task(cleanup_file, temp_out_path)

        safe_out_name = f"redacted_{Path(filename).name}"
        return FileResponse(
            temp_out_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=safe_out_name
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
