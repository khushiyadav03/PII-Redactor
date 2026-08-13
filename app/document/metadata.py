"""
Hinglish: DOCX file mein visible text ke alawa "core properties" metadata
bhi hota hai - author, last-modified-by, title, comments, etc. Ye bhi
privacy leak ho sakta hai (jaise document banane wale ka naam).

python-docx se hum core_properties tak access mil jaata hai aur
inko clear kar sakte hain. LIMITATION: tracked-changes revision history
(agar document mein hai) ko python-docx se safely remove karna reliable
nahi hai - isliye hum ise explicitly documented limitation rakhte hain,
silently "fully sanitized" claim nahi karte (assignment requirement #17).
"""
import docx.document


def clean_core_metadata(document: "docx.document.Document") -> None:
    """
    Hinglish: Sensitive core-properties fields ko clear karta hai.
    Input: python-docx Document object (in-place modify hota hai)
    Output: None
    """
    props = document.core_properties
    # Hinglish: Sab properties clean karte hain. Agar koi property writeable nahi hai
    # to exception catch kar lete hain (different python-docx versions compliance).
    for attr in ["author", "last_modified_by", "comments", "category", "description", "title", "subject", "keywords"]:
        try:
            setattr(props, attr, "")
        except Exception:
            pass
