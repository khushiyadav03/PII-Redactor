"""
Hinglish: Ye REAL evaluation hai - fabricated numbers NAHI hain.

DATASET: Hum khud se ek labeled test set banate hain jisme:
  1. Har required PII type ke TRUE POSITIVE examples hain (asli PII).
  2. Har type ke liye HARD NEGATIVE examples hain (jo PII jaisa dikhta
     hai lekin nahi hai - jaise CIN number, order number, incorporation
     date, generic 12-digit number, page number, regulatory body ka naam).

Ye sentences prospectus ke actual style se inspired hain (real values
use nahi kiye - taaki test dataset khud PII leak na kare), taaki
evaluation realistic ho. Ground truth spans manually (by hand, in code)
label kiye gaye hain - is dataset ke liye hum khud "ground truth" hain
kyunki hum khud ye sentences likh rahe hain, matlab labels 100% accurate
hain (koi human-annotation ambiguity nahi is synthetic set mein).

LIMITATION (documented honestly): Ye evaluation SYNTHETIC test sentences
par hai, real prospectus document par NAHI (real document ka full manual
PII annotation is assignment ke time-scope mein practical nahi tha).
Isliye ye numbers "detector logic ki correctness" batate hain, lekin
real-document-specific edge cases (jaise OCR noise, unusual Indian name
spellings) poori tarah capture nahi karte - ye README mein bhi likha hai.
"""
from dataclasses import dataclass
from typing import List

from app.config import RedactionPolicy
from app.evaluation.metrics import LabeledSpan, compute_metrics, overall_metrics
from app.synthetic.generator import ConsistentReplacer
from app.text.detector import detect_pii_in_text


@dataclass
class TestCase:
    text: str
    ground_truth: List[LabeledSpan]
    note: str = ""


def _span(text: str, substring: str, category: str, occurrence: int = 0) -> LabeledSpan:
    """Hinglish: helper - substring dhoondh kar LabeledSpan bana deta hai (manual offset counting se bachne ke liye)."""
    idx = -1
    for _ in range(occurrence + 1):
        idx = text.index(substring, idx + 1)
    return LabeledSpan(start=idx, end=idx + len(substring), category=category)


def build_test_dataset() -> List[TestCase]:
    cases = []

    # ---- Email ----
    t = "Please contact the compliance officer at cs.connect@examplecorp.com for grievances."
    cases.append(TestCase(t, [_span(t, "cs.connect@examplecorp.com", "email")]))

    # ---- Phone ----
    t = "You may reach the registrar at +91 9876543210 during business hours."
    cases.append(TestCase(t, [_span(t, "+91 9876543210", "phone")]))

    # Hard negative: order number should NOT be phone
    t = "Please quote Order No 9876543210 while raising a support ticket."
    cases.append(TestCase(t, [], note="order number, not a phone"))

    # ---- IP address ----
    t = "The internal application server is hosted at 10.20.30.40 on the corporate network."
    cases.append(TestCase(t, [_span(t, "10.20.30.40", "ip")]))

    # ---- PAN ----
    t = "The applicant's PAN is ABCDE1234F as verified against income tax records."
    cases.append(TestCase(t, [_span(t, "ABCDE1234F", "pan")]))

    # ---- Aadhaar (needs context) ----
    t = "As per the Aadhaar card, the Aadhaar number is 2345 6789 0123 for KYC verification."
    cases.append(TestCase(t, [_span(t, "2345 6789 0123", "aadhaar")]))

    # Hard negative: 12-digit number without Aadhaar context
    t = "The total invoice amount recorded was 2345 6789 0123 across all line items."
    cases.append(TestCase(t, [], note="12-digit number without Aadhaar context"))

    # ---- DOB (needs context) ----
    t = "Date of Birth: 15/08/1990 as mentioned in the identity proof submitted."
    cases.append(TestCase(t, [_span(t, "15/08/1990", "dob")]))

    # Hard negative: ordinary business date
    t = "The board resolution was passed on 15/08/1990 approving the said transaction."
    cases.append(TestCase(t, [], note="business date, not DOB"))

    # ---- Credit card (Luhn-valid) ----
    t = "Payment was made using card number 4532015112830366 for the transaction."
    cases.append(TestCase(t, [_span(t, "4532015112830366", "credit_card")]))

    # Hard negative: CIN-style registration number (not a card, fails Luhn)
    t = "Corporate Identity Number: U28129PN1979PLC141032 as issued by the Registrar."
    cases.append(TestCase(t, [], note="CIN, not a credit card"))

    # ---- SSN ----
    t = "For US tax purposes, the SSN on file is 123-45-6789."
    cases.append(TestCase(t, [_span(t, "123-45-6789", "ssn")]))

    # ---- Name ----
    t = "Rohan Dey was appointed as the Chief Financial Officer of the company."
    cases.append(TestCase(t, [_span(t, "Rohan Dey", "name")]))

    # ---- Company (non-allowlisted) ----
    t = "The company has a supply agreement with Bhandary Metal Extrusion Private Limited."
    cases.append(TestCase(t, [_span(t, "Bhandary Metal Extrusion Private Limited", "company")]))

    # Hard negative: regulatory body should NOT be flagged as company
    t = "The offer complies with SEBI ICDR Regulations and is listed on NSE and BSE."
    cases.append(TestCase(t, [], note="regulatory bodies are allowlisted"))

    # Hard negative: page reference number
    t = "For more information, please refer to page 243 of this prospectus."
    cases.append(TestCase(t, [], note="page number, not PII"))

    return cases


def run_evaluation():
    """
    Hinglish: Poore test dataset par detector chalata hai aur per-category
    + overall precision/recall/F1 return karta hai.
    """
    dataset = build_test_dataset()
    replacer = ConsistentReplacer()
    policy = RedactionPolicy()

    all_predicted = []
    all_ground_truth = []
    per_case_results = []

    offset = 0
    for case in dataset:
        detections = detect_pii_in_text(case.text, replacer, policy)
        predicted_spans = [
            LabeledSpan(start=d.start + offset, end=d.end + offset, category=d.category)
            for d in detections
        ]
        gt_spans = [
            LabeledSpan(start=g.start + offset, end=g.end + offset, category=g.category)
            for g in case.ground_truth
        ]
        all_predicted.extend(predicted_spans)
        all_ground_truth.extend(gt_spans)
        per_case_results.append((case, detections))
        offset += len(case.text) + 1  # Hinglish: offsets ko unique rakhne ke liye (case-by-case matching zaroori nahi hai yahan, but safe)

    per_category = compute_metrics(all_predicted, all_ground_truth)
    overall = overall_metrics(per_category)
    return per_category, overall, per_case_results
