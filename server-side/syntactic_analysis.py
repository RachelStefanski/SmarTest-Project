import spacy

nlp = spacy.load("en_core_web_md")

# פונקציה למציאת הנושא של פועל, גם בתתי משפטים
def find_subject(verb):
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass", "agent"):
            return child
    # חיפוש במעלה העץ התחבירי במקרה ואין נושא ישיר
    for ancestor in verb.ancestors:
        if ancestor.dep_ in ("relcl", "advcl", "ccomp", "xcomp"):
            return find_subject(ancestor)
    return None

# פונקציה למציאת המושא של פועל, גם בתתי משפטים
def find_object(verb):
    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "pobj", "nsubjpass"):
            return child
    for ancestor in verb.ancestors:
        if ancestor.dep_ in ("relcl", "advcl", "ccomp", "xcomp"):
            return find_object(ancestor)
    return None

# חילוץ כל זוגות (פועל, נושא, מושא) מכל המשפט – כולל תתי־משפטים
def extract_all_roles(doc):
    triples = []
    for token in doc:
        if token.pos_ == "VERB":
            subj = find_subject(token)
            obj = find_object(token)
            if subj and obj:
                triples.append((token, subj, obj))
    return triples

# חישוב דמיון בטוח בין שני טוקנים
def safe_similarity(tok1, tok2):
    if tok1 and tok2 and tok1.has_vector and tok2.has_vector:
        return tok1.similarity(tok2)
    return 0.0

# מציאת זוג התפקידים הדומה ביותר בין תשובת מורה לתלמיד
def best_match(triples1, triples2):
    best = None
    best_score = -1
    for _, subj1, obj1 in triples1:
        for _, subj2, obj2 in triples2:
            if subj1 and subj2 and obj1 and obj2:
                same = (safe_similarity(subj1, subj2) + safe_similarity(obj1, obj2)) / 2
                flipped = (safe_similarity(subj1, obj2) + safe_similarity(obj1, subj2)) / 2
                score = same - flipped  # גבוה => פחות סיכון של היפוך
                if score > best_score:
                    best = (subj1, obj1, subj2, obj2, same, flipped)
                    best_score = score
    return best

# פונקציה להשוואה כללית
def compare_roles(teacher_answer, student_answer, threshold=0.8):
    teacher_doc = nlp(teacher_answer)
    student_doc = nlp(student_answer)

    teacher_triples = extract_all_roles(teacher_doc)
    student_triples = extract_all_roles(student_doc)

    match = best_match(teacher_triples, student_triples)
    if not match:
        return True  # אם אין התאמה – לא לפסול

    subj1, obj1, subj2, obj2, same, flipped = match

    # אם ההיפוך חזק יותר מההתאמה התקינה – לפסול
    if flipped > same and flipped > threshold:
        return False
    return True

# דוגמאות בדיקה
if __name__ == "__main__":
    teacher = "Germany attacked Poland."
    student1 = "Poland attacked Germany."
    print("\nStudent 1:")
    print("Result:", compare_roles(teacher, student1))
    
