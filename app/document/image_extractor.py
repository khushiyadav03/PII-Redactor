"""
Hinglish: DOCX ek ZIP hoti hai, aur embedded images word/media/ folder ke
andar raw image files ki tarah stored hoti hain (python-docx directly
image PIXELS modify karne ka API nahi deta). Isliye images ke saath kaam
karne ke liye hum ZIP level par operate karte hain:

  1. extract_images() - saari images nikalo (OCR/face-detection ke liye)
  2. write_docx_with_replaced_images() - naye (redacted) image bytes se
     purani images replace karke ek nayi DOCX file banao.

Ye approach isliye zaroori hai kyunki "irreversible redaction" ka matlab
hai actual pixels badalna, sirf overlay draw karna nahi.
"""
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple


def extract_images(docx_path: str) -> List[Tuple[str, bytes]]:
    """Hinglish: word/media/ ke andar ki saari images (name, bytes) return karta hai."""
    images = []
    with zipfile.ZipFile(docx_path) as z:
        for name in z.namelist():
            if name.startswith("word/media/") and not name.endswith("/"):
                images.append((name, z.read(name)))
    return images


def write_docx_with_replaced_images(
    source_docx_path: str,
    output_docx_path: str,
    image_replacements: Dict[str, bytes],
) -> None:
    """
    Hinglish: source_docx_path ki copy banata hai jisme
    image_replacements dict mein di gayi images (key = "word/media/imageN.ext")
    ke bytes naye (redacted) bytes se replace ho jaate hain. Baaki sab kuch
    (XML, formatting, styles) as-is copy hota hai.

    Input:
        source_docx_path: original (already text-redacted) docx
        output_docx_path: final output path
        image_replacements: {"word/media/image4.png": <new_png_bytes>, ...}
    Output: None (file likha jaata hai output_docx_path par)
    """
    output_path = Path(output_docx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(source_docx_path, "r") as zin, \
         zipfile.ZipFile(output_docx_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in image_replacements:
                data = image_replacements[item.filename]
            zout.writestr(item, data)
