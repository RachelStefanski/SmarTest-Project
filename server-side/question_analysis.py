import spacy
from word2number import w2n
import re

nlp = spacy.load("en_core_web_trf")

class Requirement:
    def __init__(self, category, quantity=1, verb=None):
        self.category = category.strip()
        self.quantity = quantity or 1
        self.verb = verb

    def __repr__(self):
        return f"Requirement(category='{self.category}', quantity={self.quantity}, verb='{self.verb}')"

    def to_dict(self):
        return {
            'category': self.category,
            'quantity': self.quantity,
            'verb': self.verb
        }

    @staticmethod
    def from_dict(d):
        return Requirement(d['category'], d.get('quantity'), d.get('verb'))

# פיצול ע"י פסיקים או "and"
def split_conjoined_phrases(text):
    return [s.strip() for s in re.split(r',|\band\b', text) if s.strip()]

# פונקציה לחילוץ כמות מתוך טקסט
def extract_quantity(text):
    try:
        return int(text)
    except:
        try:
            return w2n.word_to_num(text)
        except:
            return 1

# הסרת פועל מתוך טקסט קטגוריה
def clean_category(text, verb_token):
    doc = nlp(text)
    return " ".join(tok.text for tok in doc if tok != verb_token).strip()

# פיצול טקסט לקטגוריות ודרישות
def extract_phrases_and_requirements(span_text, verb_token):
    reqs = []
    for phrase in split_conjoined_phrases(span_text):
        quantity = None
        for t in nlp(phrase):
            if t.like_num:
                quantity = extract_quantity(t.text)
                phrase = phrase.replace(t.text, "").strip()
                phrase = phrase.replace("  ", " ").strip()
                break
        category = clean_category(phrase, verb_token)
        if category:
            reqs.append(Requirement(category=category, quantity=quantity, verb=verb_token.text))
    return reqs

def extract_requirements(text):
    doc = nlp(text)
    all_reqs = []

    for sent in doc.sents:
        sent_reqs = []
        # חילוץ פעלים מהמשפט
        verbs = [tok for tok in sent if tok.pos_ in {"VERB", "AUX"}]
        for verb in verbs:
            verbs_to_process = [verb] + [tok for tok in sent if tok.head == verb and tok.dep_ == "conj"]
            # חילוץ דרישות מהקטגוריות של הפעלים
            for v in verbs_to_process:
                for tok in sent:
                    if tok.head == v and tok.dep_ in {
                        "dobj", "pobj", "attr", "xcomp", "ccomp", "obl", "nsubj", "conj"
                    }:
                        subtree = list(tok.subtree)
                        if len(subtree) >= 2 and any(t.pos_ in {"NOUN", "PROPN"} for t in subtree):
                            subtree = sorted(subtree, key=lambda t: t.i)
                            span_text = text[subtree[0].idx : subtree[-1].idx + len(subtree[-1])]
                            # חילוץ דרישות מהטקסט של התת־עץ
                            sent_reqs.extend(extract_phrases_and_requirements(span_text, v))

        all_reqs.extend(sent_reqs)

    # סינון דרישות כפולות וגנריות
    filtered = []
    seen = set()

    for req in all_reqs:
        cat = req.category.lower()
        verb = (req.verb or "").lower()
        key = (cat, verb)

        # בדיקת אם הדרישה כבר נראתה, אם הפועל נמצא בקטגוריה או אם היא מיותרת
        # כלומר, קטגוריה דומה כבר קיימת עם כמות גדולה יותר
        already_seen = key in seen
        verb_in_cat = verb and verb in cat
        redundant = any(
            cat in other.category.lower() and
            cat != other.category.lower() and
            len(cat.split()) < len(other.category.lower().split())
            for other in all_reqs
        )

        if not (already_seen or verb_in_cat or redundant):
            seen.add(key)
            filtered.append(req)

    # מיזוג לפי כמות מקסימלית לקטגוריה
    merged = {}
    for req in filtered:
        cat = req.category
        if cat not in merged or req.quantity > merged[cat].quantity:
            merged[cat] = req

    return list(merged.values())


# דוגמה לשימוש
if __name__ == "__main__":
    question = "Describe two political reasons and two economic reasons why World War II started"
    for r in extract_requirements(question):
        print(r)
