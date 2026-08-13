#!/usr/bin/env python3
"""
Hinglish: Ye CLI hai - assignment ka primary demonstration.
Usage:
    python run_redaction.py --input "samples/Red_Herring_Prospectus.docx" --output "outputs/redacted.docx"
"""
import argparse
import sys

from app.pipeline import process_document


def main():
    parser = argparse.ArgumentParser(description="PII Redaction Tool for DOCX files")
    parser.add_argument("--input", required=True, help="Path to input DOCX file")
    parser.add_argument("--output", required=True, help="Path to write redacted DOCX file")
    args = parser.parse_args()

    try:
        result = process_document(args.input, args.output)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: Invalid document - {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Input:  {result.input_path}")
    print(f"Output: {result.output_path}")
    print(f"Paragraphs scanned: {result.paragraphs_scanned}")
    print(f"Tables found: {result.tables_found}")
    print(f"Images found: {result.images_found}")
    print(f"Text redactions applied: {result.text_redactions_applied}")
    print(f"Images modified (pixel-redacted): {result.images_modified}")
    for zip_name, report in result.image_redaction_reports:
        print(f"  - {zip_name}: doc_type={report.doc_type} confidence={report.doc_confidence} "
              f"text_boxes={report.text_pii_boxes_masked} faces={report.faces_masked} "
              f"id_fields={report.id_fields_masked} qr={report.qr_codes_masked} modified={report.modified}")


if __name__ == "__main__":
    main()
