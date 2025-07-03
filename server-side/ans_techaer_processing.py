import ans_splitter
import units_classification
from sentence_transformers import SentenceTransformer, CrossEncoder


class answer_unit:
    def __init__(self, text, category, model):
        self.text = text
        self.emb = model.encode(text, convert_to_tensor=True) 
        self.category = category
    def to_dict(self):
        return {
            "text": self.text,
            "category": self.category,
            "embedding": self.emb.tolist() if hasattr(self.emb, "tolist") else self.embedding
        }

def analyze_teacher_answer(teacher_answer, question_requirements):
    cross_model = CrossEncoder('sentence-transformers/all-MiniLM-L6-v2') # למטרת match_units_to_categories()
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # למטרת encode()

    units = ans_splitter.split_text(teacher_answer)
    teacher_answer_units = []

    for i, unit in enumerate(units):
        category = units_classification.match_units_to_categories([unit], question_requirements, cross_model)[0][1]
        teacher_answer_units.append(answer_unit(unit, category, embed_model))

    return teacher_answer_units

if __name__ == "__main__":
    teacher_answer = "The Blitz was a sustained bombing campaign by Nazi Germany against Britain, especially London, from 1940 to 1941. Its effect on civilians included thousands of deaths, destruction of homes, fear, and the evacuation of children to the countryside."
    question_requirements = {
              "category": "its effect on British civilians",
              "quantity": 1,
              "verb": "Explain"
            }
    print(analyze_teacher_answer(teacher_answer, question_requirements.to_dict()))