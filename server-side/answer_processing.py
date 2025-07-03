from sentence_transformers import SentenceTransformer
import units_classification
import semantic_similarity
import heapq
import nli_deberta
import syntactic_analysis
from sentence_transformers import CrossEncoder
import ans_splitter
import question_analysis
from ans_techaer_processing import answer_unit

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
feedbacks = []

class cat_match_units:
    def __init__(self, category, quantity):
        self.cat = category
        self.quantity = quantity  # כמות נדרשת
        self.count = 0 #ספירת התאמות שנמצאו
        self.scores = []
    # הוספת ציון התאמה
    def add_score(self, score, i, j, text):
        self.scores.append((score, i, j, text))
        self.count += 1 
    # בדיקת האם כמות ההתאמות הגיעה למכסה הנדרשת
    def is_full(self):
        return self.count >= self.quantity
    def __repr__(self):
        return f"AnswerUnit(text='{self.text[:30]}...', category='{self.category}')"

# הדפסת מטריצת הדמיון
def print_matrix(matrix):
    print("\n--- Similarity Matrix ---")
    for row in matrix:
        print(["{:.3f}".format(v) for v in row])

def normalize(text):
    return str(text).strip().lower()

def greedy_maximum_matching(similarity_matrix, teacher_answer_units, student_answer_units, Question_requirements):
    optimal_matches = {}

    for req in Question_requirements:
        cat = req.get("category", "unknown")
        qty = req.get("quantity", 0)
        optimal_matches[cat] = cat_match_units(cat, qty)
    max_heap = []
    visited_teacher = [False] * len(teacher_answer_units)
    visited_student = [False] * len(student_answer_units)

    # בניית ערימת מקסימום מהנקודות במטריצת הדמיון
    for i, row in enumerate(similarity_matrix):
        for j, score in enumerate(row):
            if score != 0:
                heapq.heappush(max_heap, (-score, i, j))

    # בחירת ההתאמות הטובות ביותר מבלי לחזור על יחידות כבר משוייכות
    while max_heap:
        neg_score, i, j = heapq.heappop(max_heap)
        score = -neg_score
        if not visited_teacher[i] and not visited_student[j]:
            student_text = student_answer_units[j]['text']
            student_cat = student_answer_units[j]['category']
            teacher_unit = teacher_answer_units[i]
            teacher_text = teacher_unit['text']
            teacher_cat = teacher_unit['category']
            if teacher_cat in optimal_matches:
                match_obj = optimal_matches[teacher_cat]
                match_obj.add_score(score, i, j, student_text)
                visited_teacher[i] = True
                visited_student[j] = True
    return optimal_matches


def calculating_score(teacher_answer_units, student_answer, Question_requirements, question_score, cross_model, embed_model, question_type):
    
    if question_type != "open":
        student_answer_normalized = normalize(student_answer)
        normalized_model_answers = [normalize(teacher_answer_units[0])]
        if student_answer_normalized in normalized_model_answers:
            return question_score, ["Perfect"]
        else:
            return 0, ["wrong answer"]
    else:
        student_answer_units = ans_splitter.split_text(student_answer)
        student_answer_units = units_classification.match_units_to_categories(student_answer_units, [question_analysis.Requirement.from_dict(r) for r in Question_requirements], cross_model)
        student_answer_units = [answer_unit(text, category, embed_model) for text, category in student_answer_units]
        student_answer_units = [au.to_dict() for au in student_answer_units]
        similarity_matrix = [[0 for _ in range(len(student_answer_units))] for _ in range(len(teacher_answer_units))]

        for i, teacher_unit in enumerate(teacher_answer_units):
            teacher_emb = teacher_unit['embedding']
            teacher_cat = teacher_unit['category']
            teacher_text = teacher_unit['text']
            for j, student_unit in enumerate(student_answer_units):
                    student_text = student_unit['text']
                    student_cat = student_unit['category']
                    contradiction, probs = nli_deberta.detecting_contradiction(teacher_text, student_text) 
                    syntax_ok = syntactic_analysis.compare_roles(teacher_text, student_text) 
                    sem_similarity = semantic_similarity.cal_similarity(teacher_emb, student_text, embed_model)
                    if contradiction == "entailment" and probs ==1:
                        similarity = round(sem_similarity, 1) + 0.2
                    elif contradiction == "contradiction" or not syntax_ok:
                        similarity = -1
                    elif sem_similarity >= 0.5 and sem_similarity < 0.8:
                        similarity = round(sem_similarity, 1) + 0.1
                    elif sem_similarity >= 0.8:
                        similarity = 1
                    else:
                        similarity = 0
                    similarity_matrix[i][j] = similarity

        print_matrix(similarity_matrix)

        optimal_matches = greedy_maximum_matching(
            similarity_matrix,
            teacher_answer_units,
            student_answer_units,
            Question_requirements
        )

        total_requirements = sum(req.get('quantity', 0) for req in Question_requirements if req.get('quantity') is not None)
        unit_score = question_score / total_requirements if total_requirements else 0
        total_score = 0
        feedbacks = []

        # חישוב הציון הכולל בהתבסס על ההתאמות
        for req in Question_requirements:
            cat = req.get('category')
            required = req.get('quantity', 0)
            match_obj = optimal_matches.get(cat)
            matched_scores = sorted(match_obj.scores, reverse=True) if match_obj else []
            matched_count = min(len(matched_scores), required)
            score_contrib = sum((score if score > 0 else 0) * unit_score for score, _, _, _ in matched_scores[:matched_count])
            total_score += score_contrib

            # בדיקה האם הגענו למס' הדרישות הנדרשות
            if matched_count < required:
                missing = required - matched_count
                feedbacks.append(f"Missing {missing} unit{'s' if missing > 1 else ''} from category '{cat}'")

            # הוספת משוב על ההתאמות
            for score, i, j, student_text in matched_scores[:matched_count]:
                if score == -1:
                    feedbacks.append(f"Contradictory answer in category '{cat}': '{student_text[:40]}...'")
                elif score >= 0.5 and score < 1:
                    feedbacks.append(f"Imprecise answer in category '{cat}': '{student_text[:40]}...'")
        if feedbacks == []:
                feedbacks.append(f"Perfect!")
        return total_score, feedbacks
