import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import RedactionPolicy
from app.synthetic.generator import ConsistentReplacer
from app.text.detector import detect_pii_in_text


def categories(text, policy=None):
    replacer = ConsistentReplacer()
    dets = detect_pii_in_text(text, replacer, policy or RedactionPolicy())
    return [(d.category, text[d.start:d.end]) for d in dets]


def test_email_detected():
    cats = categories("Contact us at cs.connect@kshinternational.com for details.")
    assert ("email", "cs.connect@kshinternational.com") in cats


def test_phone_detected():
    cats = categories("Call +91 9876543210 for support.")
    assert any(c == "phone" for c, _ in cats)


def test_ip_address_detected():
    cats = categories("The server IP is 192.168.1.100 for internal use.")
    assert ("ip", "192.168.1.100") in cats


def test_pan_detected():
    cats = categories("His PAN is NBWPS1951N as per records.")
    assert ("pan", "NBWPS1951N") in cats


def test_dob_requires_context_true_positive():
    cats = categories("Date of Birth: 12/04/2025 as recorded.")
    assert any(c == "dob" for c, _ in cats)


def test_random_date_without_dob_context_not_flagged():
    """Hinglish: FALSE POSITIVE GUARD - incorporation/resolution date DOB nahi honi chahiye."""
    cats = categories("The resolution was passed on 12/04/2025 by the Board.")
    assert not any(c == "dob" for c, _ in cats)


def test_aadhaar_requires_context():
    cats = categories("Aadhaar Number: 2943 6593 3461 as per UIDAI records.")
    assert any(c == "aadhaar" for c, _ in cats)


def test_random_12digit_number_without_context_not_flagged_as_aadhaar():
    """Hinglish: FALSE POSITIVE GUARD - random 12-digit number Aadhaar nahi hai."""
    cats = categories("The invoice total was 1234 5678 9012 rupees only.")
    assert not any(c == "aadhaar" for c, _ in cats)


def test_cin_number_not_flagged_as_credit_card():
    """Hinglish: FALSE POSITIVE GUARD - CIN jaisa 21-char alphanumeric bhi
    hai, aur pure-digit registration numbers Luhn check fail karne chahiye."""
    cats = categories("Corporate Identity Number: U28129PN1979PLC141032")
    assert not any(c == "credit_card" for c, _ in cats)


def test_regulatory_body_not_redacted_as_company():
    """Hinglish: FALSE POSITIVE GUARD - SEBI/NSE/BSE allowlisted hain."""
    cats = categories("This offer is made in accordance with SEBI ICDR Regulations and listed on NSE and BSE.")
    assert not any(c == "company" and "SEBI" in val for c, val in cats)


def test_person_name_detected_and_consistent_replacement():
    replacer = ConsistentReplacer()
    text1 = "Rahul Sharma is the Managing Director."
    text2 = "Mr. Rahul Sharma signed the agreement."
    d1 = detect_pii_in_text(text1, replacer, RedactionPolicy())
    d2 = detect_pii_in_text(text2, replacer, RedactionPolicy())
    name_repl_1 = [d.replacement for d in d1 if d.category == "name"]
    name_repl_2 = [d.replacement for d in d2 if d.category == "name"]
    assert name_repl_1, "name should be detected in text1"
    assert name_repl_2, "name should be detected in text2"
    assert name_repl_1[0] == name_repl_2[0], "same person should get same fake replacement"


def test_page_number_not_flagged():
    """Hinglish: FALSE POSITIVE GUARD - 'see page 243' jaisa number PII nahi hai."""
    cats = categories("For further details, see page 243 of this document.")
    assert not any(c in ("phone", "aadhaar", "credit_card") for c, _ in cats)


def test_address_detected():
    cats = categories("Our registered office is at Plot No. 12, Sector 15, Pune 411016, Maharashtra.")
    assert any(c == "address" for c, _ in cats)


def test_pin_code_detected_as_address():
    cats = categories("Send mail to PIN Code 110001 today.")
    assert any(c == "address" and "110001" in val for c, val in cats)
