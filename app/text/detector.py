import re
from dataclasses import dataclass
from typing import List, Tuple

from app.config import (
    RedactionPolicy,
    COMPANY_ALLOWLIST_KEYWORDS,
    DOB_CONTEXT_KEYWORDS,
    AADHAAR_CONTEXT_KEYWORDS,
    PASSPORT_CONTEXT_KEYWORDS,
    NON_PII_NUMBER_CONTEXT,
    NON_COMPANY_ACRONYMS,
)

from app.synthetic.generator import ConsistentReplacer
from app.text import patterns as pat
from app.text.ner import extract_entities


# PII ke aas-paas itne characters tak context check karenge.
CONTEXT_WINDOW = 60


@dataclass
class Detection:
    # Logical paragraph mein PII ka starting position.
    start: int

    # Ending position. End exclusive hai.
    end: int

    # PII ka type, jaise email, phone, name, address.
    category: str

    # Original PII ki jagah jo fake value daalni hai.
    replacement: str


def _has_context(
    text: str,
    start: int,
    end: int,
    keywords: List[str],
) -> bool:

    # PII ke around 60 characters ka context window banao.
    window_start = max(0, start - CONTEXT_WINDOW)
    window_end = min(len(text), end + CONTEXT_WINDOW)

    # Context ko lowercase mein convert karo
    # taaki uppercase/lowercase ka difference matter na kare.
    surrounding = text[window_start:window_end].lower()

    # Agar koi bhi expected keyword mil gaya toh True.
    return any(
        kw in surrounding
        for kw in keywords
    )


def _is_allowlisted_company(name: str) -> bool:

    # Kuch organizations ko company PII nahi maana hai.
    # Example: regulatory/public bodies.
    lname = name.lower()

    return any(
        kw in lname
        for kw in COMPANY_ALLOWLIST_KEYWORDS
    )


def _is_likely_ner_misfire(ent_text: str) -> bool:

    # spaCy kabhi-kabhi non-company text ko ORG identify kar sakta hai.
    # Ye function aise obvious false positives ko filter karta hai.

    stripped = ent_text.strip()
    lower = stripped.lower()

    # Case 1:
    # Entity khud ek known non-company acronym hai.
    if lower in NON_COMPANY_ACRONYMS:
        return True

    # Case 2:
    # Entity ka first word blocked acronym hai.
    #
    # Example:
    # "PAN Card"
    # "ICDR Regulations"
    first_word = (
        lower.split()[0].rstrip(".,;:")
        if lower.split()
        else ""
    )

    if first_word in NON_COMPANY_ACRONYMS:
        return True

    # Case 3:
    # Entity mein digits hain.
    # Is project ke document context mein ye
    # company-name ka likely false positive maana gaya hai.
    if any(ch.isdigit() for ch in stripped):
        return True

    return False


def _overlaps_any(
    start: int,
    end: int,
    taken_ranges: List[tuple],
) -> bool:

    # Check karo ki current span already kisi claimed span
    # ke saath overlap kar raha hai ya nahi.
    return any(
        start < t_end and end > t_start
        for t_start, t_end in taken_ranges
    )


# Address identify karne ke liye useful labels.
ADDRESS_LABEL_RE = re.compile(
    r"\b(?:Registered\s+Office|Corporate\s+Office|Registered\s+Address|"
    r"Residential\s+Address|Permanent\s+Address|Correspondence\s+Address|"
    r"Mailing\s+Address|Office\s+Address|Head\s+Office|Principal\s+Office|"
    r"Corporate\s+Address|Address|Office)\b",
    re.IGNORECASE,
)


# Indian PIN code ke common formats.
PINCODE_RE = re.compile(
    r"\b[1-9][0-9]{2}\s?[0-9]{3}\b"
    r"|\b[1-9][0-9]{5}\b"
)


# Address mein commonly aane wale road/street keywords.
STREET_RE = re.compile(
    r"\b(?:Road|Rd\.?|Street|St\.?|Lane|Marg|Nagar|Colony|Enclave|"
    r"Vihar|Phase|Zone|Chowk|Bypass|Highway|Cross|Extension|"
    r"Industrial\s+Area|SEZ|Gali|Mohalla)\b",
    re.IGNORECASE,
)


# Address mein building/location related keywords.
BUILDING_RE = re.compile(
    r"\b(?:Plot|Flat|House|Shop|Building|Apartment|Suite|Bunglow|"
    r"No\.?|Number|Floor|Block|Wing|Tower|Complex|Premises|Society)\b",
    re.IGNORECASE,
)


# Date jaisa dikhne wala text.
DATE_LIKE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
)


# Financial amount identify karne ke liye.
FINANCIAL_AMOUNT_RE = re.compile(
    r"(?:Rs\.?|INR|₹)\s*\d+(?:,\d+)*(?:\.\d+)?\b"
    r"|\b\d+(?:,\d+)+(?:\.\d+)?\b"
)


# Page/section/clause numbers ko address samajhne se bachane ke liye.
PAGE_RE = re.compile(
    r"\b(?:Page|Section|Clause|Annexure|Chapter)\s+\d+\b",
    re.IGNORECASE,
)


# Order/invoice/reference numbers ko address samajhne se bachane ke liye.
ORDER_NO_RE = re.compile(
    r"\b(?:Order|Invoice|Reference|Ref|Job|Serial|Sr)\.?\s*"
    r"(?:No\.?|Number)?\s*\d+\b",
    re.IGNORECASE,
)


def extract_addresses(
    text: str,
    entities: List[tuple],
    replacer: ConsistentReplacer,
) -> List[Detection]:

    # Address usually multiple lines mein ho sakta hai.
    # Isliye pehle text ko lines mein divide karte hain.

    lines = []
    cursor = 0

    for line_text in text.split("\n"):
        start = cursor
        end = cursor + len(line_text)

        # Line ka text + logical paragraph mein uski position save karo.
        lines.append(
            (line_text, start, end)
        )

        # +1 newline character ke liye.
        cursor = end + 1

    # Har line ko address candidate ke form mein analyse karenge.
    line_candidates = []

    for line_text, start, end in lines:

        clean_line = line_text.strip()

        # Empty line ko candidate nahi maana.
        if not clean_line:
            line_candidates.append(None)
            continue

        # Address ke different signals check karo.
        has_label = bool(
            ADDRESS_LABEL_RE.search(clean_line)
        )

        has_pincode = bool(
            PINCODE_RE.search(clean_line)
        )

        has_street = bool(
            STREET_RE.search(clean_line)
        )

        has_building = bool(
            BUILDING_RE.search(clean_line)
        )

        # Current line ke andar location-related NER entities find karo.
        line_ents = []

        if entities:
            for e_start, e_end, ent_text, label in entities:

                if (
                    label in ("GPE", "LOC", "FAC")
                    and e_start >= start
                    and e_end <= end
                ):
                    line_ents.append(ent_text)

        has_gpe = len(line_ents) > 0

        # Address mein numbers common hote hain:
        # 12/4, 12-14, 12-A, 123 etc.
        has_number_pattern = bool(
            re.search(
                r"\b\d+/\d+\b|\b\d+-\d+\b|\b\d+-[A-Z]\b|\b\d+\b",
                clean_line,
            )
        )

        # Current line mein kaunse address signals mile.
        signals = []

        if has_label:
            signals.append("label")

        if has_pincode:
            signals.append("pincode")

        if has_street:
            signals.append("street")

        if has_building:
            signals.append("building")

        if has_gpe:
            signals.append("gpe")

        # Number tabhi useful signal hai jab
        # street/building/location bhi present ho.
        if has_number_pattern and (
            has_street
            or has_building
            or has_gpe
        ):
            signals.append("number")

        # Kuch lines address jaisi dikh sakti hain
        # but actually date/amount/page/order number ho sakti hain.
        is_date = (
            bool(DATE_LIKE_RE.search(clean_line))
            and not has_street
            and not has_building
        )

        is_amount = (
            bool(FINANCIAL_AMOUNT_RE.search(clean_line))
            and not has_street
            and not has_building
        )

        is_page = bool(
            PAGE_RE.search(clean_line)
        )

        is_order = bool(
            ORDER_NO_RE.search(clean_line)
        )

        # Ye strong negative signals hain.
        is_hard_negative = (
            is_date
            or is_amount
            or is_page
            or is_order
        )

        is_candidate = False

        if not is_hard_negative:

            # Explicit "Address" label mil gaya.
            if has_label:
                is_candidate = True

            # PIN + address-related signal.
            elif has_pincode and (
                has_street
                or has_building
                or has_gpe
                or has_number_pattern
            ):
                is_candidate = True

            # Multiple structural signals.
            elif (
                has_street and has_building
            ) or (
                has_street and has_gpe
            ) or (
                has_building and has_gpe
            ):
                is_candidate = True

            # Address signal + number + sufficient text.
            elif (
                has_street
                or has_building
                or has_gpe
            ) and has_number_pattern and len(clean_line) > 10:
                is_candidate = True

        if is_candidate:
            line_candidates.append(
                {
                    "text": line_text,
                    "start": start,
                    "end": end,
                    "signals": signals,
                    "has_pincode": has_pincode,
                    "has_label": has_label,
                    "has_gpe": has_gpe,
                }
            )
        else:
            line_candidates.append(None)

    # Consecutive address-like lines ko ek block mein group karenge.
    blocks = []
    current_block = []

    for i, candidate in enumerate(line_candidates):

        if candidate is not None:
            current_block.append(
                (i, candidate)
            )

        else:
            # Agar beech mein ek short connector line hai,
            # toh usse bhi address block ka part bana sakte hain.
            if (
                current_block
                and i + 1 < len(line_candidates)
                and line_candidates[i + 1] is not None
            ):
                gap_text = lines[i][0].strip()

                if (
                    len(gap_text) < 15
                    or "," in gap_text
                    or "and" in gap_text
                    or gap_text.lower()
                    in ("india", "pune", "maharashtra")
                ):
                    current_block.append(
                        (
                            i,
                            {
                                "text": lines[i][0],
                                "start": lines[i][1],
                                "end": lines[i][2],
                                "signals": [],
                                "has_pincode": False,
                                "has_label": False,
                                "has_gpe": False,
                            },
                        )
                    )
                    continue

            # Current block complete ho gaya.
            if current_block:
                blocks.append(current_block)
                current_block = []

    # Agar last block loop ke end tak pending hai.
    if current_block:
        blocks.append(current_block)

    detections = []

    # Har candidate block ko final validation do.
    for block in blocks:

        block_text = "\n".join(
            item[1]["text"]
            for item in block
        )

        block_start = block[0][1]["start"]
        block_end = block[-1][1]["end"]

        # Block ke saare unique signals collect karo.
        all_signals = set()

        block_has_pincode = False
        block_has_label = False
        block_has_gpe = False
        block_has_street = False
        block_has_building = False

        for idx, item in block:

            all_signals.update(
                item["signals"]
            )

            if item["has_pincode"]:
                block_has_pincode = True

            if item["has_label"]:
                block_has_label = True

            if item["has_gpe"]:
                block_has_gpe = True

            if "street" in item["signals"]:
                block_has_street = True

            if "building" in item["signals"]:
                block_has_building = True

        is_valid = False
        unique_signals_count = len(all_signals)

        # Explicit label + at least one more signal.
        if (
            block_has_label
            and unique_signals_count >= 2
        ):
            is_valid = True

        # PIN + location/address structure.
        elif block_has_pincode and (
            block_has_gpe
            or block_has_street
            or block_has_building
        ):
            is_valid = True

        # Street + location.
        elif block_has_street and block_has_gpe:
            is_valid = True

        # Building + location.
        elif block_has_building and block_has_gpe:
            is_valid = True

        # Multiple lines + at least one address signal.
        elif (
            len(block) > 1
            and (
                block_has_street
                or block_has_building
                or block_has_gpe
            )
        ):
            is_valid = True

        if is_valid:

            # Start/end ke around extra whitespace hatao.
            while (
                block_start < block_end
                and text[block_start].isspace()
            ):
                block_start += 1

            while (
                block_end > block_start
                and text[block_end - 1].isspace()
            ):
                block_end -= 1

            # Bahut chhota block address nahi maana.
            if block_end - block_start > 5:

                # Address ke liye consistent fake replacement banao.
                replacement_val = replacer.fake_address(
                    text[block_start:block_end]
                )

                detections.append(
                    Detection(
                        block_start,
                        block_end,
                        "address",
                        replacement_val,
                    )
                )

    return detections


# Name identify karne ke liye useful context labels.
NAME_KEYWORD_RE = re.compile(
    r"\b(?:Name|Full\s+Name|Contact\s+Person|Director|Directors|"
    r"Chairman|Managing\s+Director|Promoter|Promoters|Applicant|"
    r"Father|Father's\s+Name|Mother|Authorised\s+Signatory|"
    r"Authorized\s+Signatory|Partner|Beneficiary)\b\s*:?",
    re.IGNORECASE,
)


def extract_names_with_context(
    text: str,
    replacer: ConsistentReplacer,
) -> List[Detection]:

    # Context keyword ke baad capitalized words ko
    # possible person name ke roop mein check karte hain.

    IGNORE_NAME_WORDS = {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "sebi",
        "nse",
        "bse",
        "rbi",
        "uidai",
        "ltd",
        "limited",
        "govt",
        "government",
        "india",
        "director",
        "chairman",
        "promoter",
        "promoters",
        "signatory",
        "partner",
        "secretary",
        "executive",
        "board",
        "meeting",
        "annexure",
        "section",
        "chapter",
        "page",
        "table",
        "company",
        "companies",
        "act",
        "pan",
        "ssn",
        "kyc",
        "cin",
        "isin",
    }

    detections = []

    # Text mein name-related keywords search karo.
    for m in NAME_KEYWORD_RE.finditer(text):

        keyword_end = m.end()

        # Keyword ke baad next 120 characters ko inspect karo.
        snippet = text[
            keyword_end:keyword_end + 120
        ]

        # Capitalized words ka possible name pattern.
        for name_match in re.finditer(
            r"\b[A-Z][a-zA-Z\.]+(?:\s+[A-Z][a-zA-Z\.]+)*\b",
            snippet,
        ):

            name_str = name_match.group()

            name_words = [
                w.lower().strip(".:")
                for w in name_str.split()
            ]

            is_valid = True

            # Bahut short candidate ignore karo.
            if len(name_str) <= 2:
                is_valid = False

            # Known non-name words ignore karo.
            if any(
                w in IGNORE_NAME_WORDS
                for w in name_words
            ):
                is_valid = False

            # Digits wala candidate name nahi maana.
            if any(
                ch.isdigit()
                for ch in name_str
            ):
                is_valid = False

            # Single very-short word ignore karo.
            if (
                len(name_words) == 1
                and len(name_words[0]) <= 2
            ):
                is_valid = False

            if is_valid:

                # Snippet ke relative position ko
                # original text ke absolute position mein convert karo.
                match_start = (
                    keyword_end
                    + name_match.start()
                )

                match_end = (
                    keyword_end
                    + name_match.end()
                )

                val_str = text[
                    match_start:match_end
                ].strip(":")

                clean_start = match_start
                clean_end = (
                    match_start
                    + len(val_str)
                )

                detections.append(
                    Detection(
                        clean_start,
                        clean_end,
                        "name",
                        replacer.fake_name(val_str),
                    )
                )

    return detections


def detect_pii_in_text(
    text: str,
    replacer: ConsistentReplacer,
    policy: RedactionPolicy,
    precomputed_entities=None,
) -> List[Detection]:

    # Ye main detection function hai.
    # Multiple detection layers ko combine karta hai.

    detections: List[Detection] = []

    # Already-used text ranges yahan store honge.
    taken_ranges: List[tuple] = []

    def _claim(
        start,
        end,
        category,
        replacement,
    ):
        # Agar current detection kisi existing detection
        # se overlap karti hai toh ise skip karo.
        if _overlaps_any(
            start,
            end,
            taken_ranges,
        ):
            return

        # Range ko reserve karo.
        taken_ranges.append(
            (start, end)
        )

        # Final detection save karo.
        detections.append(
            Detection(
                start,
                end,
                category,
                replacement,
            )
        )

    # NER entities pehle calculate kar lo.
    # Agar caller ne already calculate kiya hai,
    # toh wahi use karo.
    entities = (
        precomputed_entities
        if precomputed_entities is not None
        else extract_entities([text])[0]
    )

    # ============================================================
    # LAYER 0.5: Context-based heuristics
    # ============================================================

    # Address detection.
    if policy.redact_addresses:

        address_dets = extract_addresses(
            text,
            entities,
            replacer,
        )

        for d in address_dets:
            _claim(
                d.start,
                d.end,
                d.category,
                d.replacement,
            )

    # Context-based name detection.
    if policy.redact_names:

        name_dets = extract_names_with_context(
            text,
            replacer,
        )

        for d in name_dets:
            _claim(
                d.start,
                d.end,
                d.category,
                d.replacement,
            )

    # ============================================================
    # LAYER 1: Structured regex-based PII
    # ============================================================

    if policy.redact_emails:

        for m in pat.EMAIL_RE.finditer(text):
            _claim(
                m.start(),
                m.end(),
                "email",
                replacer.fake_email(m.group()),
            )

    if policy.redact_pan:

        for m in pat.PAN_RE.finditer(text):
            _claim(
                m.start(),
                m.end(),
                "pan",
                replacer.fake_pan(m.group()),
            )

    if policy.redact_ssn:

        for m in pat.SSN_RE.finditer(text):
            _claim(
                m.start(),
                m.end(),
                "ssn",
                replacer.fake_ssn(m.group()),
            )

    if policy.redact_ip:

        for m in pat.IP_RE.finditer(text):
            _claim(
                m.start(),
                m.end(),
                "ip",
                replacer.fake_ip(m.group()),
            )

    if policy.redact_phones:

        for m in pat.PHONE_RE.finditer(text):

            # Agar number ke context mein non-PII keyword hai,
            # toh phone number nahi maana.
            if _has_context(
                text,
                m.start(),
                m.end(),
                NON_PII_NUMBER_CONTEXT,
            ):
                continue

            _claim(
                m.start(),
                m.end(),
                "phone",
                replacer.fake_phone(m.group()),
            )

    if policy.redact_credit_card:

        for m in pat.CREDIT_CARD_RE.finditer(text):

            # Spaces/hyphens remove karke actual digit count check karo.
            digits = re.sub(
                r"[ -]",
                "",
                m.group(),
            )

            if len(digits) < 13:
                continue

            # Regex match ke baad Luhn validation.
            if not pat.luhn_checksum_valid(
                m.group()
            ):
                continue

            _claim(
                m.start(),
                m.end(),
                "credit_card",
                replacer.fake_credit_card(
                    m.group()
                ),
            )

    if policy.redact_aadhaar:

        for m in pat.AADHAAR_RE.finditer(text):

            # Aadhaar pattern generic hai,
            # isliye context mandatory hai.
            if _has_context(
                text,
                m.start(),
                m.end(),
                AADHAAR_CONTEXT_KEYWORDS,
            ):
                _claim(
                    m.start(),
                    m.end(),
                    "aadhaar",
                    replacer.fake_aadhaar(
                        m.group()
                    ),
                )

    if policy.redact_passport:

        for m in pat.PASSPORT_RE.finditer(text):

            # Passport pattern ke saath context check.
            if _has_context(
                text,
                m.start(),
                m.end(),
                PASSPORT_CONTEXT_KEYWORDS,
            ):
                _claim(
                    m.start(),
                    m.end(),
                    "passport",
                    replacer.fake_passport(
                        m.group()
                    ),
                )

    if policy.redact_dob:

        for m in pat.DATE_RE.finditer(text):

            # Har date DOB nahi hoti.
            # Isliye DOB-related context check karo.
            if _has_context(
                text,
                m.start(),
                m.end(),
                DOB_CONTEXT_KEYWORDS,
            ):
                _claim(
                    m.start(),
                    m.end(),
                    "dob",
                    replacer.fake_dob(
                        m.group()
                    ),
                )

    # ============================================================
    # LAYER 2 + 3: NER-based names and companies
    # ============================================================

    for start, end, ent_text, label in entities:

        # PERSON entity ko name PII maana.
        if (
            label == "PERSON"
            and policy.redact_names
        ):
            _claim(
                start,
                end,
                "name",
                replacer.fake_name(ent_text),
            )

        # ORG entity ko company candidate maana.
        elif (
            label == "ORG"
            and policy.redact_companies
        ):

            # Allowlisted organizations ko skip karo.
            if _is_allowlisted_company(ent_text):
                continue

            # Obvious NER false positives ko skip karo.
            if _is_likely_ner_misfire(ent_text):
                continue

            _claim(
                start,
                end,
                "company",
                replacer.fake_company(ent_text),
            )

    # Final detections ko text ke order mein return karo.
    detections.sort(
        key=lambda d: d.start
    )

    return detections