"""
Hinglish: Layer 1 detection - structured PII jo regex se reliably pakdi ja
sakti hai (email, phone, credit card, IP, PAN, Aadhaar, passport, SSN).
Ye "deterministic" hai matlab same input par hamesha same output -
isliye evaluation reproducible rehta hai.

Har PII type ke liye alag compiled regex + ek helper function jo
match validate karta hai (kuch cases mein sirf regex kaafi nahi,
jaise credit card ke liye Luhn check).
"""
import re

# Hinglish: Email - standard pattern, kaafi permissive (real-world emails
# handle karne ke liye) lekin trailing punctuation avoid karta hai.
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# Hinglish: Indian phone numbers - +91 ke saath ya bina, 10-digit mobile,
# ya landline with STD code jaise "+ 91 20 4505 3237". Spaces/hyphens allow
# karte hain kyunki prospectus mein "+ 91 20 4505 3237" style common hai.
PHONE_RE = re.compile(
    # 1. Indian Mobile: starts with +91, 91, or 0 (optional), then digits 6-9, then 9 more digits (with optional spaces/hyphens)
    r"(?<!\d)(?:\+?\s?91[\s-]?)?[6-9]\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)"
    r"|(?<!\d)(?:\+?\s?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"  # 5+5 format e.g. 98765 43210 or 81081 14949
    # 2. Indian Landline: starts with 0 or +91, followed by area code (2-3 digits), then landline number (6-8 digits)
    r"|(?<!\d)(?:0\d{2,3}|(?:\+?\s?91[\s-]?)?\b\d{2,3})[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)"
    r"|(?<!\d)0\d{2,3}[\s-]?\d{6,8}(?!\d)"
    # 3. Generic international with + prefix
    r"|(?<!\d)\+\s?\d{1,3}[\s-]?\d{1,4}[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)"
)

# Hinglish: IPv4 address. IPv6 assignment scope se bahar rakha hai
# (prospectus jaise business documents mein IPv6 milna extremely rare hai;
# scope simple rakhne ke liye IPv4 tak limited kiya, README mein documented).
IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)

# Hinglish: Credit card - 13-19 digits, spaces/hyphens ke saath ya bina.
# Sirf regex kaafi nahi hai (bahut saare 16-digit numbers non-card hote
# hain jaise CIN, ISIN) - isliye Luhn checksum se validate bhi karte hain
# (detector.py mein).
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")

# Hinglish: US SSN format (XXX-XX-XXXX). Assignment US-style PII bhi
# manga hai isliye include kiya, chahe prospectus Indian document ho.
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Hinglish: PAN (Permanent Account Number) - India ka format bahut strict
# hai: 5 letters + 4 digits + 1 letter. Ye ek high-precision pattern hai
# (false positive rate bahut kam) isliye context ki zaroorat nahi.
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

# Hinglish: Aadhaar - 12 digit number, aksar 4-4-4 groups mein likha jaata
# hai. Ye pattern generic hai (koi bhi 12-digit number match karega) isliye
# ISE HAMESHA context keyword ke saath use karna chahiye (detector.py mein),
# warna false positives bahut badh jayenge (phone+pin code jaise cheezein).
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")

# Hinglish: Indian passport number format - 1 letter + 7 digits.
PASSPORT_RE = re.compile(r"\b[A-PR-WYa-pr-wy][0-9]{7}\b")

# Hinglish: Date patterns - kai formats support karte hain kyunki
# prospectus mein "December 10, 2025", "12/04/2025", "06/05/2000" sab
# style milte hain. Ye sirf "date-shaped text" detect karta hai - DOB
# hai ya nahi ye decide karne ke liye detector.py context dekhta hai.
DATE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"
)


def luhn_checksum_valid(digits: str) -> bool:
    """
    Hinglish: Credit card number validate karne ke liye standard Luhn
    algorithm. Ye zyadatar random 13-19 digit numbers (jaise CIN, ISIN,
    registration numbers) ko credit-card samajhne se rokta hai -
    precision improve karta hai.
    """
    digits = re.sub(r"[ -]", "", digits)
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0
