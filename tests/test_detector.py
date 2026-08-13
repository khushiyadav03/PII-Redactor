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
    # Hinglish: PIN code with address label or location context should be matched
    cats = categories("Send mail to registered office at PIN Code 110001 today.")
    assert any(c == "address" and "110001" in val for c, val in cats)


# --- Phase 1: Robust Address Detection Tests ---

def test_address_single_line():
    cats = categories("Registered Office at Plot 45, Residency Road, Pune 411001, India.")
    assert any(c == "address" for c, _ in cats)


def test_address_multi_line():
    cats = categories("Registered Office:\n11/3 Mueller Road,\nPune - 411045,\nMaharashtra, India")
    # Grouped as one single address block
    assert len([val for c, val in cats if c == "address"]) == 1
    assert "Mueller Road" in [val for c, val in cats if c == "address"][0]


def test_address_with_apartment():
    cats = categories("Mailing Address: Flat No 302, Wing B, Green Glen Layout, Bangalore - 560103")
    assert any(c == "address" for c, _ in cats)


def test_address_without_pin():
    # Hinglish: Address without PIN should still be detected if it has strong location + street + label signals
    cats = categories("Registered Office: Mueller Road, Pune, Maharashtra, India")
    assert any(c == "address" for c, _ in cats)


def test_address_containing_company_and_person_name():
    # Hinglish: Names/Companies inside the address block should be absorbed
    cats = categories("Corporate Office:\nScaler AI Labs Private Limited,\nc/o Rahul Sharma, Plot No 12, Sector 14,\nGurgaon - 122001")
    address_vals = [val for c, val in cats if c == "address"]
    assert len(address_vals) == 1
    assert "Scaler AI" in address_vals[0]
    assert "Rahul Sharma" in address_vals[0]


def test_address_false_positives_standalone():
    # Hinglish: Standalone geographic names or numeric IDs must NOT be redacted as address
    assert not any(c == "address" for c, _ in categories("He lives in Pune."))
    assert not any(c == "address" for c, _ in categories("Maharashtra is a state."))
    assert not any(c == "address" for c, _ in categories("The code is 411045."))
    assert not any(c == "address" for c, _ in categories("Refer to Page 41 of the documentation."))
    assert not any(c == "address" for c, _ in categories("The invoice amount was ₹4,11,045 only."))
    assert not any(c == "address" for c, _ in categories("Order No. 411045 has been processed."))
    assert not any(c == "address" for c, _ in categories("The date is 10/12/2025."))


# --- Phase 1: Robust Name Detection Tests ---

def test_name_slashed_promoters_list():
    cats = categories("Promoters:\nRahul Sharma / Priya Gupta / Amit Kumar")
    names = [val for c, val in cats if c == "name"]
    assert "Rahul Sharma" in names
    assert "Priya Gupta" in names
    assert "Amit Kumar" in names


def test_name_next_line_layout():
    cats = categories("Managing Director:\nRahul Sharma")
    assert ("name", "Rahul Sharma") in cats


def test_name_same_line_layout():
    cats = categories("Father's Name: Sugriv Singh")
    assert ("name", "Sugriv Singh") in cats


def test_name_allowlist_protection():
    # Hinglish: Regulatory names or month names should not trigger name redactions
    cats = categories("In January 2026, the board of SEBI and NSE reviewed the RHP of the company.")
    assert not any(c == "name" for c, _ in cats)


# --- Phase 1: Robust Phone Number Tests ---

def test_phone_formats():
    phone_examples = [
        "9876543210",
        "98765 43210",
        "+91 9876543210",
        "+91-9876543210",
        "+91 98765 43210",
        "81081 14949",
        "022-68052182",
        "+91 22 6805 2182"
    ]
    for ph in phone_examples:
        cats = categories(f"You can contact us at {ph} for help.")
        assert any(c == "phone" for c, _ in cats), f"Phone format failed: {ph}"


def test_phone_false_positives():
    # Hinglish: Generic big numbers or share numbers shouldn't be phone PII
    assert not any(c == "phone" for c, _ in categories("We issued 123456789012 shares."))
    assert not any(c == "phone" for c, _ in categories("The page number is 243."))
    assert not any(c == "phone" for c, _ in categories("The total cost is INR 98,76,54,321."))

