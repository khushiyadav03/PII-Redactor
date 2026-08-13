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
    props.author = ""
    props.last_modified_by = ""
    props.comments = ""
    props.category = ""
    props.description = ""
    try:
        props.title = ""
        props.subject = ""
        props.keywords = ""
    except Exception:
        # Hinglish: kuch properties set na ho paayein to bhi crash na ho -
        # ye non-critical hai, best-effort cleanup hai.
        pass
