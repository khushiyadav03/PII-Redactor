import re


# Email address detect karne ke liye regex.
EMAIL_RE = re.compile(
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
)


# Indian phone numbers detect karta hai.
#
# Multiple formats support karta hai:
# +91 9876543210
# 98765 43210
# 987-654-3210
# +91 20 4505 3237
# etc.
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\s?91[\s-]?)?[6-9]\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)"
    r"|(?<!\d)(?:\+?\s?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"
    r"|(?<!\d)(?:0\d{2,3}|(?:\+?\s?91[\s-]?)?\b\d{2,3})[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)"
    r"|(?<!\d)0\d{2,3}[\s-]?\d{6,8}(?!\d)"
    r"|(?<!\d)\+\s?\d{1,3}[\s-]?\d{1,4}[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)"
)


# IPv4 address detect karta hai.
# Har part 0-255 ke range mein hona chahiye.
IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)


# Credit card ke possible numbers detect karta hai.
#
# 13-19 digits support karta hai.
# Spaces aur hyphens bhi allowed hain.
#
# Important:
# Regex sirf format check karta hai.
# Actual credit card validation Luhn algorithm se hoti hai.
CREDIT_CARD_RE = re.compile(
    r"\b(?:\d[ -]?){13,19}\b"
)


# US Social Security Number ka standard format.
# Example: 123-45-6789
SSN_RE = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)


# Indian PAN ka strict format:
# 5 uppercase letters + 4 digits + 1 uppercase letter
#
# Example:
# ABCDE1234F
PAN_RE = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
)


# Aadhaar ka 12-digit format.
#
# Spaces allowed hain:
# 1234 5678 9012
# 123456789012
#
# Ye generic pattern hai, isliye detector.py mein
# context check karna zaroori hai.
AADHAAR_RE = re.compile(
    r"\b\d{4}\s?\d{4}\s?\d{4}\b"
)


# Indian passport number:
# 1 letter + 7 digits
#
# Example:
# A1234567
PASSPORT_RE = re.compile(
    r"\b[A-PR-WYa-pr-wy][0-9]{7}\b"
)


# Date ke common formats detect karta hai.
#
# Examples:
# 12/04/2025
# 06-05-2000
# December 10, 2025
DATE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|"
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
)


def luhn_checksum_valid(digits: str) -> bool:
    # Spaces aur hyphens remove karo.
    digits = re.sub(r"[ -]", "", digits)

    # Number sirf digits ka hona chahiye
    # aur length 13-19 ke beech honi chahiye.
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False

    total = 0

    # Right-to-left calculation ke liye digits reverse karo.
    reverse_digits = digits[::-1]

    for i, ch in enumerate(reverse_digits):
        n = int(ch)

        # Right se alternate digit ko double karte hain.
        if i % 2 == 1:
            n *= 2

            # Agar result 9 se bada hai,
            # toh 9 subtract karte hain.
            if n > 9:
                n -= 9

        total += n

    # Valid Luhn number ka total 10 se divisible hota hai.
    return total % 10 == 0