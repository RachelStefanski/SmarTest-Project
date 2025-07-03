import os
import json
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, CrossEncoder
import answer_processing
import question_analysis
import units_classification

# --- הגדרות קבועות ---
TESTS_DIR = './tests'
cross_model = CrossEncoder('sentence-transformers/all-MiniLM-L6-v2')
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# --- המרה להטמעה אם צריך ---
def embed_if_needed(model_answer_list):
    embedded = []
    for ans in model_answer_list:
        if isinstance(ans, dict) and "embedding" in ans:
            embedded.append(ans)
        elif isinstance(ans, str):
            embedding = embed_model.encode(ans).tolist()
            embedded.append({
                "text": ans,
                "embedding": embedding
            })
    return embedded

# --- עיבוד שאלה יחידה ---
def process_question(question_data, answer_models):
    question_id = question_data.get('question_id')
    question_type = question_data.get('type', '')
    question_text = question_data.get('question', '')
    score = question_data.get('score', 0)

    # חיפוש תשובת מורה מתאימה
    relevant_answer = next(
        (a for a in answer_models if a.get('question_id') == question_id),
        {}
    )

    # פתוחות: לוקחים את היחידות המפוצלות אם יש
    if question_type == "open":
        requirements = question_data.get('requirements')
        if not requirements:
            requirements = question_analysis.extract_requirements(question_text)
            question_data['requirements'] = requirements

        # עדיפות ל-answer_units מהמורה, אחרת טקסט גולמי
        model_answer_raw = relevant_answer.get('answer_units') or relevant_answer.get('answer') or []
        model_answer = embed_if_needed(model_answer_raw)

        # שיוך קטגוריה ליחידה לפי הדרישות
        for unit in model_answer:
            unit_text = unit["text"]
            pairs = [(unit_text, req['category']) for req in requirements]
            scores = cross_model.predict(pairs)
            best_idx = int(np.argmax(scores))
            unit["category"] = requirements[best_idx]['category']

        question_data['answer_units'] = model_answer
        final_model_answer = model_answer

    else:
        # סגורות: נשתמש פשוט בטקסט או תשובה יחידה
        requirements = []
        question_data['requirements'] = requirements
        model_answer = relevant_answer.get('answer') or question_data.get('model_answer', '')
        question_data['answer_units'] = [model_answer] if isinstance(model_answer, str) else model_answer
        final_model_answer = model_answer

    return {
        'question_id': question_id,
        'type': question_type,
        'question_text': question_text,
        'model_answer': final_model_answer,
        'score': score,
        'requirements': requirements
    }

# --- חישוב ציון לשאלה ---
def grade_answer(model_answers, student_answer, requirements, max_score, question_type):
    return answer_processing.calculating_score(
        model_answers,
        student_answer,
        requirements,
        max_score,
        cross_model,
        embed_model,
        question_type
    )

# --- עיבוד מבחן מלא ---
def process_exam(exam_data):
    questions_map = {}

    for work in exam_data.get('works', []):
        for question in work.get('questions', []):
            processed = process_question(question, work.get('answer_models', []))
            question.update(processed)
            questions_map[question['question_id']] = question

    for student in exam_data.get('students', []):
        print("תשובות תלמיד:", [a['question_id'] for a in student.get('answers', [])])
        total_score = 0
        total_max_score = 0

        for answer in student.get('answers', []):
            qid = answer.get('question_id')
            student_text = answer.get('answer', '')
            question = questions_map.get(qid)

            if question:
                max_score = question.get('weight', 0)
                model_answers = question.get('answer_units', []) if question.get('type') == 'open' else (
                    [question.get('model_answer')] if isinstance(question.get('model_answer'), str) else question.get('model_answer')
                )
                requirements = question.get('requirements', [])
                question_type = question.get('type', '')
                print(f"{qid} - תשובת מורה כוללת {len(model_answers)} יחידות")

                grade, remark = grade_answer(
                    model_answers,
                    student_text,
                    requirements,
                    max_score,
                    question_type
                )
                answer['score'] = grade
                answer['remark'] = remark

                total_score += grade
                total_max_score += max_score
            else:
                answer['score'] = 0
                answer['remark'] = "שאלה לא נמצאה"

        student['final_score'] = round((total_score / total_max_score) * 100, 2) if total_max_score else 0
        general_feedback = (
            "excellent" if student['final_score'] >= 80
            else "good" if student['final_score'] >= 65
            else "needs improvement"
        )
        student['general_feedback'] = general_feedback

# --- בדיקת מבחנים אם הגיע זמנם ---
def check_tests():
    now = datetime.now()
    for test_file in os.listdir(TESTS_DIR):
        if not test_file.endswith('.json'):
            continue

        test_path = os.path.join(TESTS_DIR, test_file)
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        start_time_str = test_data.get('start_time')
        duration_minutes = int(test_data.get('duration_minutes', 0))

        if not start_time_str:
            print(f'⚠️ No start_time found in {test_file}, skipping.')
            continue

        start_dt = datetime.fromisoformat(start_time_str)
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        if now >= end_dt and not test_data.get('checked'):
            print(f'📘 Checking test: {test_file}')
            process_exam(test_data)
            test_data['checked'] = True
            with open(test_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)

# --- עזר: קריאת/שמירת JSON ---
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- הפעלה ידנית לבדיקה ---
if __name__ == "__main__":
    file_path = r"C:\Users\User.DESKTOP-HT62HRR\Desktop\SmarTest-Project\server-side\tests\aa.json"
    exam_data = load_json_file(file_path)
    process_exam(exam_data)
    save_json_file(file_path, exam_data)
    print("✓ מבחן עודכן עם ציונים ודרישות.")
