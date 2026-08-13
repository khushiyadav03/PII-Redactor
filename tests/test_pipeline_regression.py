import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import docx

from app.pipeline import process_document

REAL_DOC = os.path.join(os.path.dirname(__file__), "..", "samples", "Red_Herring_Prospectus.docx")
OUT_DOC = os.path.join(os.path.dirname(__file__), "..", "outputs", "regression_test_output.docx")


def test_full_pipeline_runs_on_real_prospectus_and_output_opens():
    """
    Hinglish: REGRESSION TEST - poora pipeline (text + image) real
    prospectus par chalta hai bina crash kiye, aur output DOCX successfully
    reopen hoti hai (document corruption nahi hui).
    """
    result = process_document(REAL_DOC, OUT_DOC)
    assert result.paragraphs_scanned > 0
    assert result.tables_found == 76
    assert result.images_found == 8
    assert result.text_redactions_applied > 0
    assert result.images_modified >= 2  # at least the PAN and Aadhaar cards

    # Hinglish: output DOCX corrupt nahi honi chahiye - reopen karke verify karo
    reopened = docx.Document(OUT_DOC)
    assert len(reopened.tables) == 76

    os.remove(OUT_DOC)


def test_no_temp_files_left_behind():
    """Hinglish: PRIVACY GUARD - intermediate temp docx cleanup honi chahiye."""
    process_document(REAL_DOC, OUT_DOC)
    temp_path = OUT_DOC.replace(".docx", ".text_only.tmp.docx")
    assert not os.path.exists(temp_path)
    os.remove(OUT_DOC)


def test_fail_closed_on_corrupt_image():
    """Hinglish: FAILURE-HANDLING GUARD - image corrupt hone par processing fail honi chahiye."""
    from unittest.mock import patch
    import pytest

    with patch("app.pipeline.extract_images") as mock_extract:
        mock_extract.return_value = [("word/media/corrupt_image.png", b"invalid image bytes")]
        with pytest.raises(ValueError) as exc:
            process_document(REAL_DOC, OUT_DOC)
        assert "Image could not be decoded" in str(exc.value)

    # Ensure no output document was saved
    assert not os.path.exists(OUT_DOC)

