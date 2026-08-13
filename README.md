# PII Redaction Tool

A CLI tool that reads a `.docx` file and produces a redacted copy where personally identifiable information — in both **text** and **embedded images** — is replaced with consistent, synthetic fake values or masked out entirely.

Built for the Scaler AI Labs / EPAM assignment, tested end-to-end on the actual `Red_Herring_Prospectus.docx` (a 1,006-paragraph, 76-table, 8-image real-world legal document that happens to contain two embedded government ID card scans — a PAN card and an Aadhaar card — which made this a genuinely useful test case for the visual-PII requirements).

## 1. Problem statement

Detect and redact PII from a DOCX document — required types: full names, emails, phone numbers, company names, addresses, SSNs, credit card numbers, DOBs, IP addresses. Extended scope (self-imposed): PAN/Aadhaar/passport numbers, and **visual** PII inside embedded images — faces, scanned ID documents, QR codes, signatures.

## 2. Architecture

```
DOCX
 │
 ├── TEXT (paragraphs, tables, headers, footers)
 │      │
 │      ├── reconstruct logical text across split Word runs
 │      ├── Layer 1: regex (email/phone/IP/PAN/SSN/credit-card/Aadhaar/passport/date)
 │      ├── Layer 2: spaCy NER (PERSON, ORG)
 │      ├── Layer 3: context rules (DOB needs "Date of Birth" nearby, company allowlist)
 │      └── map detected spans back to Word runs → redact in place
 │
 └── IMAGES (word/media/*)
        │
        ├── OCR (Tesseract) → words + bounding boxes
        ├── same text-PII detector run on OCR text → mask matching word-boxes
        ├── ID document classification (PAN / Aadhaar / Passport, keyword+pattern based)
        ├── label→value line heuristic → mask name / father's name / signature regions
        ├── face detection (Haar cascade) → mask face boxes
        ├── QR detection (only masked if inside a recognized ID document)
        └── solid black rectangles drawn directly on pixels → re-encoded, swapped into DOCX zip

                    │
              Final redacted.docx
```

Both the CLI and (optional) FastAPI wrapper call the same `app.pipeline.process_document()` — no duplicated logic.

## 3. Supported PII types

**Text:** name, email, phone, company (policy-gated — see below), address*, SSN, credit card, date of birth (context-gated), IP address, PAN, Aadhaar (context-gated), passport (context-gated).

*Address is **not** implemented as a dedicated extractor — see Limitations in the evaluation report. Everything else above is implemented and tested.

**Visual:** faces, PAN/Aadhaar/passport card classification + field-level redaction (name, DOB, ID number, photo), OCR'd text-PII inside any image, QR codes on recognized ID documents. Signature redaction is a best-effort heuristic, documented as imperfect.

## 4. Text pipeline

The core challenge: Word splits a single visible word across multiple internal `<w:r>` runs (e.g. "Rahul Sharma" might be stored as `"Rah"` + `"ul "` + `"Sharma"`). Running regex/NER on each run independently would miss PII that spans a run boundary. So `app/document/reader.py` first reconstructs a **logical paragraph string** (all runs concatenated) with an offset→run mapping, detection runs on that logical string, and `app/document/writer.py` maps the resulting spans back onto the original runs — putting the full replacement in the first overlapping run and blanking the rest, so no fragment of the original text survives in any run.

Detection is layered (regex → NER → context rules) rather than one big model, specifically so precision failures are debuggable and fixable one rule at a time — which is exactly what happened during development (see the evaluation report's "started at 62% precision" story).

**Company-name policy** (the riskiest category — legal documents are full of legitimate regulator/exchange names): any spaCy ORG entity gets redacted **unless** it matches a small allowlist of regulators/exchanges/government bodies (SEBI, NSE, BSE, RBI, UIDAI, etc. — see `COMPANY_ALLOWLIST_KEYWORDS` in `config.py`). This is a recall-over-precision choice, made explicit rather than hidden.

**Consistent replacements**: every unique PII value is mapped to one fake value via a seeded-Faker cache (`app/synthetic/generator.py`), so "Rahul Sharma" appearing five times becomes the same fake name five times, not five different ones.

## 5. Image pipeline

Tesseract OCR extracts words + pixel bounding boxes from every embedded image. The **same** text-PII detector used on document text is reused on the OCR'd text (regex layer works identically; NER is unreliable on short/fragmented OCR text, documented in `id_detector.py`). A lightweight keyword+regex classifier (`app/vision/id_detector.py`) decides whether an image is a PAN/Aadhaar/passport card; if so, a label→value line heuristic locates the "Name"/"Father's Name"/"Signature" fields by position rather than NER. Faces are found with OpenCV's Haar cascade. All detected regions are masked by drawing **solid black rectangles directly onto the pixel array** (not an overlay) before re-encoding — the original pixels are gone, not hidden.

## 6. ID document handling

| Type | Fields redacted |
|---|---|
| PAN | name, father's name, DOB, PAN number, signature (best-effort), photo |
| Aadhaar | name, DOB, Aadhaar number, photo (address/QR: see limitations) |
| Passport | name, passport number, DOB, photo, signature (best-effort) |

A key design choice: once an image is confidently classified as a specific ID type, the ID-number regex is applied to the **whole** OCR'd text for that image rather than requiring a nearby context keyword — because the keyword and the number are often spatially far apart on the card (e.g. "Unique Identification Authority of India" appears in a banner, the Aadhaar number appears lower down), so a local text-proximity check misses it even though the document-level classification already justifies treating that number as sensitive.

## 7. Redaction strategy

Text: replace in-place inside Word runs (formatting preserved). Images: pixel-level masking (formatting/structure of the DOCX otherwise untouched). Metadata: `author`, `last_modified_by`, `comments`, `title`, `subject`, `keywords` core properties are cleared. Tracked-changes/revision history is **not** guaranteed to be removed — `python-docx` doesn't expose a safe API for this, and that's stated plainly rather than silently claimed.

## 8. Replacement strategy

`ConsistentReplacer` (Faker, seeded by a hash of the original value) guarantees the same input always maps to the same fake output, both within one run and across repeated runs of the tool on the same document (reproducible for testing). Fake names are length-capped relative to the original to avoid blowing up table layouts.

## 9. Tech stack

Python, `python-docx`, regex, spaCy (`en_core_web_sm`), Tesseract + `pytesseract`, OpenCV (Haar cascade + built-in QR detector), Faker, pytest. No LangChain, no vector DB, no LLM calls — deterministic, local, explainable, per the assignment's own constraints.

## 10. Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 11. Tesseract installation

```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

## 12. spaCy model installation

```bash
python -m spacy download en_core_web_sm
```

## 13. Usage

```bash
python run_redaction.py --input "samples/Red_Herring_Prospectus.docx" --output "outputs/redacted.docx"
```

## 14. Example command (actual run used for this deliverable)

```bash
python run_redaction.py --input samples/Red_Herring_Prospectus.docx --output outputs/redacted.docx
```

Output:
```
Paragraphs scanned: 4686
Tables found: 76
Images found: 8
Text redactions applied: 3362
Images modified (pixel-redacted): 3
  - word/media/image4.png: doc_type=pan confidence=high ... modified=True
  - word/media/image5.png: doc_type=aadhaar confidence=low ... modified=True
  - word/media/image2.jpeg: doc_type=unknown confidence=low text_boxes=1 ... modified=True
```

## 15–20. Evaluation methodology, results, precision/recall/accuracy, false positives/negatives

See **`reports/evaluation_report.md`** — full breakdown across a synthetic labeled test set and a real-document spot-check (20 actual paragraphs from the prospectus), including the false-positive fix that was made during development (spaCy misreading acronyms like "PAN"/"SSN" as company names) and the honest recall gaps that remain (Indian-name recall, some phone-number groupings).

Headline numbers:
- Synthetic test set: **Precision 1.00, Recall 1.00, F1 1.00** (10/10, after the acronym-misfire fix)
- Real-document spot-check: **Precision 0.88, Recall 0.80, F1 0.84** (name detection is the weak point — 44% recall on real Indian names)

## 21. Limitations

Full list in `reports/evaluation_report.md` §5. Short version: name recall on Indian names is the weakest link; address extraction isn't implemented; signature/QR detection are best-effort heuristics that were tested and shown to have real gaps; tracked-changes metadata isn't sanitized.

## 22. Future improvements

See `reports/evaluation_report.md` §6 — phone regex coverage, a name gazetteer to lift recall, a dedicated address extractor, and a DNN-based face detector.

## Project structure

```
pii-redactor/
├── app/
│   ├── config.py                 # policy, allowlists, thresholds
│   ├── pipeline.py                # process_document() - single source of truth
│   ├── document/
│   │   ├── reader.py               # split-run reconstruction, tables/headers/footers
│   │   ├── writer.py               # span → run redaction
│   │   ├── image_extractor.py      # ZIP-level image extract/replace
│   │   └── metadata.py             # core-properties cleanup
│   ├── text/
│   │   ├── patterns.py             # regex (Layer 1)
│   │   ├── ner.py                  # spaCy wrapper (Layer 2)
│   │   ├── detector.py             # combined layered detector (Layer 3 rules)
│   │   └── redactor.py             # document-level orchestration + batching
│   ├── vision/
│   │   ├── ocr.py                  # Tesseract + preprocessing
│   │   ├── face_detector.py        # Haar cascade
│   │   ├── id_detector.py          # PAN/Aadhaar/passport classification + field heuristics
│   │   ├── qr_detector.py          # OpenCV QR detection
│   │   └── image_redactor.py       # orchestrates all of the above, does the pixel masking
│   ├── synthetic/
│   │   └── generator.py            # consistent fake-value mapping
│   └── evaluation/
│       ├── metrics.py              # precision/recall/F1
│       └── evaluator.py            # synthetic labeled test set + runner
├── tests/                          # 24 pytest tests (unit + regression)
├── samples/                        # input sample (Red_Herring_Prospectus.docx)
├── outputs/                        # redacted.docx goes here
├── reports/evaluation_report.md
├── requirements.txt
└── run_redaction.py                # CLI entry point
```

## Privacy notes

Raw PII values are never printed to console or written to logs/reports — only counts and categories. No document content is sent to any external/cloud API; all processing (regex, spaCy, OCR, face detection) runs locally. Temporary intermediate files are deleted after each run.
