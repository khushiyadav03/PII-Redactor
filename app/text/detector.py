"""
Hinglish: Ye teeno layers ko combine karke final PII detections deta hai:
  Layer 1: regex (structured PII - email, phone, PAN, etc.)
  Layer 2: spaCy NER (names, companies)
  Layer 3: contextual rules (DOB vs random date, company allowlist policy)

Design: har detection ek Detection object hai jisme sirf (start, end,
category, replacement) hota hai - matched raw text ko hum turant consume
karke replacement bana lete hain aur store nahi karte (privacy requirement:
raw PII ko unnecessarily hold/log nahi karna - config.py #28).
"""
import re
from dataclasses import dataclass
from typing import List

from app.config import (
    RedactionPolicy, COMPANY_ALLOWLIST_KEYWORDS, DOB_CONTEXT_KEYWORDS,
    AADHAAR_CONTEXT_KEYWORDS, PASSPORT_CONTEXT_KEYWORDS, NON_PII_NUMBER_CONTEXT,
    NON_COMPANY_ACRONYMS,
)
from app.synthetic.generator import ConsistentReplacer
from app.text import patterns as pat
from app.text.ner import extract_entities

CONTEXT_WINDOW = 60  # Hinglish: kitne characters aage/peeche dekhna hai context ke liye


@dataclass
class Detection:
    start: int
    end: int
    category: str
    replacement: str


def _has_context(text: str, start: int, end: int, keywords: List[str]) -> bool:
    """Hinglish: span ke aas-paas ke text mein koi context keyword hai ya nahi."""
    window_start = max(0, start - CONTEXT_WINDOW)
    window_end = min(len(text), end + CONTEXT_WINDOW)
    surrounding = text[window_start:window_end].lower()
    return any(kw in surrounding for kw in keywords)


def _is_allowlisted_company(name: str) -> bool:
    """Hinglish: Regulatory/public bodies ko company-PII nahi maante (config.py policy)."""
    lname = name.lower()
    return any(kw in lname for kw in COMPANY_ALLOWLIST_KEYWORDS)


def _is_likely_ner_misfire(ent_text: str) -> bool:
    """
    Hinglish: spaCy ORG-label ke do common misfire patterns filter karte hain:
      1. ent_text ek known non-company acronym hai (PAN, SSN, KYC, etc.)
      2. ent_text mein digits hain (real company names is dataset ke
         context mein digits nahi rakhte; jab ORG span mein number ho
         to aksar ye "Order No 12345" jaisa false span hota hai, real
         company name nahi). Ye ek documented heuristic limitation hai -
         genuine company names jinme digits hote hain (jaise "3M",
         "7-Eleven") is heuristic se miss ho sakte hain.
    """
    stripped = ent_text.strip()
    if stripped.lower() in NON_COMPANY_ACRONYMS:
        return True
    if any(ch.isdigit() for ch in stripped):
        return True
    return False


def _overlaps_any(start: int, end: int, taken_ranges: List[tuple]) -> bool:
    return any(start < t_end and end > t_start for t_start, t_end in taken_ranges)


def detect_pii_in_text(
    text: str,
    replacer: ConsistentReplacer,
    policy: RedactionPolicy,
    precomputed_entities=None,
) -> List[Detection]:
    """
    Hinglish: Ek logical paragraph text par saare layers chalakar final,
    non-overlapping Detection list return karta hai.

    Input:
        text: paragraph ka poora logical text (reader.py se)
        replacer: ConsistentReplacer (same-entity => same-fake-value)
        policy: kaunse categories on/off hain
        precomputed_entities: (optional) NER batch se pehle se nikale
            gaye entities, taaki har paragraph ke liye alag se NER call
            na karni pade (performance ke liye batch processing detector
            ke bahar hoti hai - see redactor.py)

    Output: list of Detection, sorted by start, NON-overlapping
            (agar do detections overlap karte hain to zyada "specific"
            wala jeetta hai - regex-based structured PII ko NER-based
            se priority dete hain).
    """
    detections: List[Detection] = []
    taken_ranges: List[tuple] = []  # Hinglish: already-claimed spans, overlap avoid karne ke liye

    def _claim(start, end, category, replacement):
        if _overlaps_any(start, end, taken_ranges):
            return
        taken_ranges.append((start, end))
        detections.append(Detection(start, end, category, replacement))

    # ---------- LAYER 1: high-precision structured regex ----------
    if policy.redact_emails:
        for m in pat.EMAIL_RE.finditer(text):
            _claim(m.start(), m.end(), "email", replacer.fake_email(m.group()))

    if policy.redact_pan:
        for m in pat.PAN_RE.finditer(text):
            _claim(m.start(), m.end(), "pan", replacer.fake_pan(m.group()))

    if policy.redact_ssn:
        for m in pat.SSN_RE.finditer(text):
            _claim(m.start(), m.end(), "ssn", replacer.fake_ssn(m.group()))

    if policy.redact_ip:
        for m in pat.IP_RE.finditer(text):
            _claim(m.start(), m.end(), "ip", replacer.fake_ip(m.group()))

    if policy.redact_phones:
        for m in pat.PHONE_RE.finditer(text):
            # Hinglish: Order/registration numbers jaise false positives se
            # bachne ke liye negative-context check.
            if _has_context(text, m.start(), m.end(), NON_PII_NUMBER_CONTEXT):
                continue
            _claim(m.start(), m.end(), "phone", replacer.fake_phone(m.group()))

    if policy.redact_credit_card:
        for m in pat.CREDIT_CARD_RE.finditer(text):
            digits = re.sub(r"[ -]", "", m.group())
            if len(digits) < 13:
                continue
            if not pat.luhn_checksum_valid(m.group()):
                continue  # Hinglish: CIN/ISIN/registration numbers Luhn fail karte hain, skip
            _claim(m.start(), m.end(), "credit_card", replacer.fake_credit_card(m.group()))

    if policy.redact_aadhaar:
        for m in pat.AADHAAR_RE.finditer(text):
            # Hinglish: 12-digit pattern bahut generic hai (phone/pin-code
            # bhi match ho sakte) - isliye SIRF context ke saath accept.
            if _has_context(text, m.start(), m.end(), AADHAAR_CONTEXT_KEYWORDS):
                _claim(m.start(), m.end(), "aadhaar", replacer.fake_aadhaar(m.group()))

    if policy.redact_passport:
        for m in pat.PASSPORT_RE.finditer(text):
            if _has_context(text, m.start(), m.end(), PASSPORT_CONTEXT_KEYWORDS):
                _claim(m.start(), m.end(), "passport", replacer.fake_passport(m.group()))

    if policy.redact_dob:
        for m in pat.DATE_RE.finditer(text):
            # Hinglish: Har date DOB nahi hoti - prospectus mein incorporation
            # dates, resolution dates, board-meeting dates bhi hain. Sirf
            # "Date of Birth" / "DOB" jaisa context ho to hi redact karo.
            if _has_context(text, m.start(), m.end(), DOB_CONTEXT_KEYWORDS):
                _claim(m.start(), m.end(), "dob", replacer.fake_dob(m.group()))

    # ---------- LAYER 1.5: Address detection (PIN code & layout patterns) ----------
    if policy.redact_addresses:
        # Indian PIN Codes (6 digits, e.g. 110001 or 411 016)
        for m in re.finditer(r"\b[1-9][0-9]{2}\s?[0-9]{3}\b", text):
            _claim(m.start(), m.end(), "address", replacer.fake_address(m.group()))
        # Common address patterns
        for m in re.finditer(r"\b(?:Plot|Flat|House|Shop|Building|Apartment|Suite)\s+(?:No\.?|Number)?\s*\w+\b", text, re.IGNORECASE):
            _claim(m.start(), m.end(), "address", replacer.fake_address(m.group()))

    # ---------- LAYER 2 + 3: NER-based names, companies & addresses ----------
    entities = precomputed_entities if precomputed_entities is not None else extract_entities([text])[0]
    for start, end, ent_text, label in entities:
        if label == "PERSON" and policy.redact_names:
            _claim(start, end, "name", replacer.fake_name(ent_text))
        elif label == "ORG" and policy.redact_companies:
            if _is_allowlisted_company(ent_text):
                continue  # Hinglish: regulators/exchanges/govt bodies - PII nahi maante
            if _is_likely_ner_misfire(ent_text):
                continue  # Hinglish: acronym/digit-based NER misfire - see docstring
            _claim(start, end, "company", replacer.fake_company(ent_text))
        elif label in ("GPE", "LOC", "FAC") and policy.redact_addresses:
            _claim(start, end, "address", replacer.fake_address(ent_text))

    detections.sort(key=lambda d: d.start)
    return detections
