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
from typing import List, Tuple

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
    Hinglish: spaCy ORG-label ke teen common misfire patterns filter karte hain:
      1. ent_text exactly ek known non-company acronym hai (PAN, SSN, KYC, etc.)
      2. ent_text mein digits hain — real company names is prospectus context
         mein digits nahi rakhte; "Order No 12345" jaisa false span hota hai.
      3. ent_text ka PEHLA WORD ek known non-company acronym hai — spaCy kabhi
         kabhi "ICDR Regulations" ya "PAN Card" jaise multi-word spans ko ORG
         tag kar deta hai. First-word check se ye sab filtered ho jaate hain.
         (Documented heuristic limitation: "3M", "7-Eleven" miss ho sakte hain.)
    """
    stripped = ent_text.strip()
    lower = stripped.lower()
    # Exact match
    if lower in NON_COMPANY_ACRONYMS:
        return True
    # Hinglish: Multi-word span jo ek blocked acronym se start ho —
    # jaise "ICDR Regulations", "PAN Card" etc. ko company nahi mante.
    first_word = lower.split()[0].rstrip(".,;:") if lower.split() else ""
    if first_word in NON_COMPANY_ACRONYMS:
        return True
    if any(ch.isdigit() for ch in stripped):
        return True
    return False


def _overlaps_any(start: int, end: int, taken_ranges: List[tuple]) -> bool:
    return any(start < t_end and end > t_start for t_start, t_end in taken_ranges)


# Hinglish: Address label context anchors
ADDRESS_LABEL_RE = re.compile(
    r"\b(?:Registered\s+Office|Corporate\s+Office|Registered\s+Address|Residential\s+Address|"
    r"Permanent\s+Address|Correspondence\s+Address|Mailing\s+Address|Office\s+Address|"
    r"Head\s+Office|Principal\s+Office|Corporate\s+Address|Address|Office)\b",
    re.IGNORECASE
)
PINCODE_RE = re.compile(r"\b[1-9][0-9]{2}\s?[0-9]{3}\b|\b[1-9][0-9]{5}\b")
STREET_RE = re.compile(
    r"\b(?:Road|Rd\.?|Street|St\.?|Lane|Marg|Nagar|Colony|Enclave|Vihar|Phase|Zone|"
    r"Chowk|Sector|Bypass|Highway|Cross|Extension|Industrial\s+Area|SEZ|Gali|Mohalla)\b",
    re.IGNORECASE
)
BUILDING_RE = re.compile(
    r"\b(?:Plot|Flat|House|Shop|Building|Apartment|Suite|Bunglow|No\.?|Number|Floor|Block|Wing|Tower|Complex|Premises|Society)\b",
    re.IGNORECASE
)
DATE_LIKE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
FINANCIAL_AMOUNT_RE = re.compile(r"(?:Rs\.?|INR|₹)\s*\d+(?:,\d+)*(?:\.\d+)?\b|\b\d+(?:,\d+)+(?:\.\d+)?\b")
PAGE_RE = re.compile(r"\b(?:Page|Section|Clause|Annexure|Chapter)\s+\d+\b", re.IGNORECASE)
ORDER_NO_RE = re.compile(r"\b(?:Order|Invoice|Reference|Ref|Job|Serial|Sr)\.?\s*(?:No\.?|Number)?\s*\d+\b", re.IGNORECASE)


def extract_addresses(text: str, entities: List[tuple], replacer: ConsistentReplacer) -> List[Detection]:
    """
    Hinglish: Dedicated multi-line address detector. Heuristic layers:
    1. Score logically split lines for address structural signals (PIN, street, GPE labels, numbers).
    2. Group consecutive candidate lines.
    3. Validate block signals to prevent false positives (no isolated cities/states/PINs).
    """
    lines = []
    cursor = 0
    for line_text in text.split("\n"):
        start = cursor
        end = cursor + len(line_text)
        lines.append((line_text, start, end))
        cursor = end + 1

    line_candidates = []
    for line_text, start, end in lines:
        clean_line = line_text.strip()
        if not clean_line:
            line_candidates.append(None)
            continue

        has_label = bool(ADDRESS_LABEL_RE.search(clean_line))
        has_pincode = bool(PINCODE_RE.search(clean_line))
        has_street = bool(STREET_RE.search(clean_line))
        has_building = bool(BUILDING_RE.search(clean_line))

        line_ents = []
        if entities:
            for e_start, e_end, ent_text, label in entities:
                if label in ("GPE", "LOC", "FAC") and e_start >= start and e_end <= end:
                    line_ents.append(ent_text)
        has_gpe = len(line_ents) > 0
        has_number_pattern = bool(re.search(r"\b\d+/\d+\b|\b\d+-\d+\b|\b\d+-[A-Z]\b|\b\d+\b", clean_line))

        signals = []
        if has_label: signals.append("label")
        if has_pincode: signals.append("pincode")
        if has_street: signals.append("street")
        if has_building: signals.append("building")
        if has_gpe: signals.append("gpe")
        if has_number_pattern and (has_street or has_building or has_gpe):
            signals.append("number")

        is_date = bool(DATE_LIKE_RE.search(clean_line)) and not has_street and not has_building
        is_amount = bool(FINANCIAL_AMOUNT_RE.search(clean_line)) and not has_street and not has_building
        is_page = bool(PAGE_RE.search(clean_line))
        is_order = bool(ORDER_NO_RE.search(clean_line))

        is_hard_negative = is_date or is_amount or is_page or is_order
        is_candidate = False
        if not is_hard_negative:
            if has_label:
                is_candidate = True
            elif has_pincode and (has_street or has_building or has_gpe or has_number_pattern):
                is_candidate = True
            elif (has_street and has_building) or (has_street and has_gpe) or (has_building and has_gpe):
                is_candidate = True
            elif (has_street or has_building or has_gpe) and has_number_pattern and len(clean_line) > 10:
                is_candidate = True

        if is_candidate:
            line_candidates.append({
                "text": line_text,
                "start": start,
                "end": end,
                "signals": signals,
                "has_pincode": has_pincode,
                "has_label": has_label,
                "has_gpe": has_gpe
            })
        else:
            line_candidates.append(None)

    blocks = []
    current_block = []
    for i, candidate in enumerate(line_candidates):
        if candidate is not None:
            current_block.append((i, candidate))
        else:
            if current_block and i + 1 < len(line_candidates) and line_candidates[i+1] is not None:
                gap_text = lines[i][0].strip()
                if len(gap_text) < 15 or "," in gap_text or "and" in gap_text or gap_text.lower() in ("india", "pune", "maharashtra"):
                    current_block.append((i, {
                        "text": lines[i][0],
                        "start": lines[i][1],
                        "end": lines[i][2],
                        "signals": [],
                        "has_pincode": False,
                        "has_label": False,
                        "has_gpe": False
                    }))
                    continue
            if current_block:
                blocks.append(current_block)
                current_block = []
    if current_block:
        blocks.append(current_block)

    detections = []
    for block in blocks:
        block_text = "\n".join(item[1]["text"] for item in block)
        block_start = block[0][1]["start"]
        block_end = block[-1][1]["end"]

        all_signals = set()
        block_has_pincode = False
        block_has_label = False
        block_has_gpe = False
        block_has_street = False
        block_has_building = False

        for idx, item in block:
            all_signals.update(item["signals"])
            if item["has_pincode"]: block_has_pincode = True
            if item["has_label"]: block_has_label = True
            if item["has_gpe"]: block_has_gpe = True
            if "street" in item["signals"]: block_has_street = True
            if "building" in item["signals"]: block_has_building = True

        is_valid = False
        unique_signals_count = len(all_signals)

        if block_has_label and unique_signals_count >= 2:
            is_valid = True
        elif block_has_pincode and (block_has_gpe or block_has_street or block_has_building):
            is_valid = True
        elif block_has_street and block_has_gpe:
            is_valid = True
        elif block_has_building and block_has_gpe:
            is_valid = True
        elif len(block) > 1 and (block_has_street or block_has_building or block_has_gpe):
            is_valid = True

        if is_valid:
            while block_start < block_end and text[block_start].isspace():
                block_start += 1
            while block_end > block_start and text[block_end - 1].isspace():
                block_end -= 1

            if block_end - block_start > 5:
                replacement_val = replacer.fake_address(text[block_start:block_end])
                detections.append(Detection(block_start, block_end, "address", replacement_val))
    return detections


NAME_KEYWORD_RE = re.compile(
    r"\b(?:Name|Full\s+Name|Contact\s+Person|Director|Directors|Chairman|Managing\s+Director|"
    r"Promoter|Promoters|Applicant|Father|Father's\s+Name|Mother|Authorised\s+Signatory|"
    r"Authorized\s+Signatory|Partner|Beneficiary)\b\s*:?",
    re.IGNORECASE
)


def extract_names_with_context(text: str, replacer: ConsistentReplacer) -> List[Detection]:
    """
    Hinglish: Indian names recall improve karne ke liye contextual extraction.
    Snippet ke andar capitalized name-like patterns (e.g. Rahul Sharma) search karta hai.
    """
    IGNORE_NAME_WORDS = {
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "sebi", "nse", "bse", "rbi",
        "uidai", "ltd", "limited", "govt", "government", "india", "director",
        "chairman", "promoter", "promoters", "signatory", "partner", "secretary",
        "executive", "board", "meeting", "annexure", "section", "chapter", "page",
        "table", "company", "companies", "act", "pan", "ssn", "kyc", "cin", "isin"
    }
    
    detections = []
    
    for m in NAME_KEYWORD_RE.finditer(text):
        keyword_end = m.end()
        snippet = text[keyword_end:keyword_end + 120]
        
        for name_match in re.finditer(r"\b[A-Z][a-zA-Z\.]+(?:\s+[A-Z][a-zA-Z\.]+)*\b", snippet):
            name_str = name_match.group()
            name_words = [w.lower().strip(".:") for w in name_str.split()]
            
            is_valid = True
            if len(name_str) <= 2:
                is_valid = False
            if any(w in IGNORE_NAME_WORDS for w in name_words):
                is_valid = False
            if any(ch.isdigit() for ch in name_str):
                is_valid = False
            if len(name_words) == 1 and len(name_words[0]) <= 2:
                is_valid = False
                
            if is_valid:
                match_start = keyword_end + name_match.start()
                match_end = keyword_end + name_match.end()
                
                val_str = text[match_start:match_end].strip(":")
                clean_start = match_start
                clean_end = match_start + len(val_str)
                
                detections.append(Detection(clean_start, clean_end, "name", replacer.fake_name(val_str)))
            
    return detections


def detect_pii_in_text(
    text: str,
    replacer: ConsistentReplacer,
    policy: RedactionPolicy,
    precomputed_entities=None,
) -> List[Detection]:
    """
    Hinglish: Ek logical paragraph text par saare layers chalakar final,
    non-overlapping Detection list return karta hai.
    """
    detections: List[Detection] = []
    taken_ranges: List[tuple] = []

    def _claim(start, end, category, replacement):
        if _overlaps_any(start, end, taken_ranges):
            return
        taken_ranges.append((start, end))
        detections.append(Detection(start, end, category, replacement))

    # Hinglish: Precompute entities early
    entities = precomputed_entities if precomputed_entities is not None else extract_entities([text])[0]

    # ---------- LAYER 0.5: Phase 1 Context Heuristics ----------
    if policy.redact_addresses:
        address_dets = extract_addresses(text, entities, replacer)
        for d in address_dets:
            _claim(d.start, d.end, d.category, d.replacement)

    if policy.redact_names:
        name_dets = extract_names_with_context(text, replacer)
        for d in name_dets:
            _claim(d.start, d.end, d.category, d.replacement)

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
            if _has_context(text, m.start(), m.end(), NON_PII_NUMBER_CONTEXT):
                continue
            _claim(m.start(), m.end(), "phone", replacer.fake_phone(m.group()))

    if policy.redact_credit_card:
        for m in pat.CREDIT_CARD_RE.finditer(text):
            digits = re.sub(r"[ -]", "", m.group())
            if len(digits) < 13:
                continue
            if not pat.luhn_checksum_valid(m.group()):
                continue
            _claim(m.start(), m.end(), "credit_card", replacer.fake_credit_card(m.group()))

    if policy.redact_aadhaar:
        for m in pat.AADHAAR_RE.finditer(text):
            if _has_context(text, m.start(), m.end(), AADHAAR_CONTEXT_KEYWORDS):
                _claim(m.start(), m.end(), "aadhaar", replacer.fake_aadhaar(m.group()))

    if policy.redact_passport:
        for m in pat.PASSPORT_RE.finditer(text):
            if _has_context(text, m.start(), m.end(), PASSPORT_CONTEXT_KEYWORDS):
                _claim(m.start(), m.end(), "passport", replacer.fake_passport(m.group()))

    if policy.redact_dob:
        for m in pat.DATE_RE.finditer(text):
            if _has_context(text, m.start(), m.end(), DOB_CONTEXT_KEYWORDS):
                _claim(m.start(), m.end(), "dob", replacer.fake_dob(m.group()))

    # ---------- LAYER 2 + 3: NER-based names & companies ----------
    for start, end, ent_text, label in entities:
        if label == "PERSON" and policy.redact_names:
            _claim(start, end, "name", replacer.fake_name(ent_text))
        elif label == "ORG" and policy.redact_companies:
            if _is_allowlisted_company(ent_text):
                continue
            if _is_likely_ner_misfire(ent_text):
                continue
            _claim(start, end, "company", replacer.fake_company(ent_text))

    detections.sort(key=lambda d: d.start)
    return detections
