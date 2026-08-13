"""
Hinglish: Layer 2 detection - spaCy ka pretrained NER model use karke
PERSON (names) aur ORG (company names) entities nikalte hain.

Hum en_core_web_sm (chhota, fast model) use kar rahe hain - production
mein bada model (en_core_web_trf) zyada accurate hota lekin bahut slow
hai is document ke size (1000+ paragraphs) ke liye. Ye tradeoff README
mein documented hai.
"""
import spacy

_nlp = None


def get_nlp():
    """Hinglish: Model ek hi baar load karte hain (lazy singleton) - reload
    expensive hai aur baar baar load karne ki zaroorat nahi."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. Install with: "
                "python -m spacy download en_core_web_sm"
            ) from exc
    return _nlp


def extract_entities(texts):
    """
    Hinglish: Multiple paragraph texts ek saath batch-process karte hain
    (nlp.pipe) - ye ek-ek karke process karne se kaafi fast hai bade
    documents ke liye.

    Input: list of strings
    Output: list of list-of-(start, end, text, label) - har input text ke
            corresponding entities, same order mein.
    """
    nlp = get_nlp()
    results = []
    for doc in nlp.pipe(texts, batch_size=64):
        ents = [
            (ent.start_char, ent.end_char, ent.text, ent.label_)
            for ent in doc.ents
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC")
        ]
        results.append(ents)
    return results
