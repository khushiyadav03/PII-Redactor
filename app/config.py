"""
Hinglish: Ye file poore project ki configuration rakhti hai - PII types on/off,
company-name policy, confidence thresholds, file paths, etc.
Ek jagah se sab settings control karne ke liye taaki har module mein
hard-coded values scatter na ho.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RedactionPolicy:
    """
    Hinglish: Ye class decide karti hai ki kaun se PII category ko redact
    karna hai. Assignment ke minimum required types by default True hain.
    Extra (PAN/Aadhaar/passport) bhi True hain kyunki user ne explicitly
    manga hai extended requirement mein.
    """
    redact_names: bool = True
    redact_emails: bool = True
    redact_phones: bool = True
    redact_companies: bool = True          # policy-gated, see COMPANY_ALLOWLIST
    redact_addresses: bool = True
    redact_ssn: bool = True
    redact_credit_card: bool = True
    redact_dob: bool = True
    redact_ip: bool = True
    redact_pan: bool = True
    redact_aadhaar: bool = True
    redact_passport: bool = True

    # Visual PII
    redact_faces: bool = True
    redact_id_documents: bool = True
    redact_qr_on_id: bool = True
    redact_signatures_on_id: bool = True


# Hinglish: Company-name redaction sabse risky hai kyunki prospectus mein
# bahut saari legitimate/regulatory organizations ka naam aata hai
# (SEBI, NSE, BSE, RBI, banks, law firms, auditors, stock exchanges).
# Agar hum har ORG entity ko blindly redact karenge to precision bahut gir
# jayegi (false positives). Isliye hum ek "known public/regulatory org"
# allowlist rakhte hain - ye names redact NAHI honge, chahe spaCy unhe
# ORG bataye. Baaki ORG entities (jo allowlist mein nahi hain) ko potential
# PII (private company / vendor / individual's employer) maan kar redact
# karte hain. Ye policy README mein explicitly document ki gayi hai.
COMPANY_ALLOWLIST_KEYWORDS = [
    "sebi", "securities and exchange board", "nse", "national stock exchange",
    "bse", "bombay stock exchange", "rbi", "reserve bank of india",
    "registrar of companies", "roc", "ministry of corporate affairs",
    "income tax department", "uidai", "unique identification authority",
    "government of india", "govt. of india", "companies act",
    "sebi icdr regulations", "depositories act", "nsdl", "cdsl",
    "irdai", "competition commission of india", "cci",
]

# Hinglish: Ye words dates/numbers ke aas paas mile to confidence badhti hai
# ki wo genuinely DOB hai, random date nahi (jaise incorporation date,
# resolution date, agreement date jo prospectus mein bahut zyada hain).
DOB_CONTEXT_KEYWORDS = ["date of birth", "dob", "born on", "d.o.b", "janm"]

SSN_CONTEXT_KEYWORDS = ["social security", "ssn"]

PAN_CONTEXT_KEYWORDS = ["permanent account number", "pan card", "income tax department"]

AADHAAR_CONTEXT_KEYWORDS = ["aadhaar", "aadhar", "unique identification authority", "uidai"]

PASSPORT_CONTEXT_KEYWORDS = ["passport", "republic of india", "type p"]

# Hinglish: Non-PII numeric patterns jo galti se PII detect ho sakte hain -
# in false positives se bachne ke liye negative-context words.
NON_PII_NUMBER_CONTEXT = [
    "invoice", "order no", "order number", "cin", "corporate identity number",
    "isin", "folio", "page", "clause", "section", "regulation",
]

# Hinglish: LIMITATION FIX - spaCy ka en_core_web_sm model chhote/isolated
# sentences mein common PII-related ACRONYMS (jaise "PAN", "SSN", "KYC")
# ko galti se ORG (company) tag kar deta hai, kyunki ye all-caps tokens
# hain aur model ko surrounding paragraph context nahi milta jitna training
# data mein tha. Ye ek known/documented NER limitation hai (README mein
# explain kiya gaya hai) jise hum is chhoti stoplist se practically fix
# karte hain - agar ORG entity exactly in acronyms se match kare to use
# company nahi maante.
NON_COMPANY_ACRONYMS = {
    "pan", "ssn", "kyc", "dob", "cin", "isin", "gst", "tan", "ifsc",
    "nsdl", "cdsl", "roc", "sebi", "rbi", "irdai", "upi", "ipo", "rhp",
    "ecs", "neft", "rtgs", "kpi", "faq", "otp", "pin", "cvv",
    "aadhaar", "aadhar",  # Hinglish: proper noun, spaCy sm often mistags as ORG
}

# Hinglish: Project ke andar ke default paths.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_REPORTS_DIR = BASE_DIR / "reports"

# Hinglish: OCR aur face-detection confidence thresholds - tuning knobs.
OCR_MIN_CONFIDENCE = 40          # 0-100, Tesseract confidence scale
# Hinglish: scaleFactor=1.05 choti/dabi hui ID-photo par recall improve
# karta hai (tested on sample PAN/Aadhaar images) - tradeoff ye hai ki
# kabhi-kabhi textured background par ek extra false-positive box aa
# sakta hai. PRIVACY TOOL ke liye ye acceptable tradeoff hai: face miss
# karne se better hai ki thoda extra background bhi mask ho jaye
# (over-redaction safe-fail direction hai, under-redaction nahi).
FACE_DETECTION_SCALE_FACTOR = 1.05
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_MIN_SIZE_PX = 40

# Hinglish: Kis extent tak fake replacement text allow hai (bahut lamba
# fake naam original table layout todh sakta hai).
MAX_SYNTHETIC_NAME_LENGTH_RATIO = 1.6  # fake name original se itna zyada lamba na ho
