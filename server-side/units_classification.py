from sentence_transformers import CrossEncoder
import numpy as np
import question_analysis
import ans_splitter

def match_units_to_categories(units, requirements, model):
    categorized_units = []
    print(requirements)
    for text in units:
        pairs = [(text, req.category) for req in requirements]
        scores = model.predict(pairs)

        # softmax על הסקור
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()

        best_idx = np.argmax(probabilities)
        best_cat = requirements[best_idx].category 
        best_prob = probabilities[best_idx]
        
        categorized_units.append((text, best_cat))
        print(f"\nSentence: '{text}'\n→ Predicted category: '{best_cat}' (probability: {best_prob:.4f})")

    return categorized_units


if __name__ == "__main__":
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    question = "Explain what the Blitz was and its effect on British civilians."
    units = ans_splitter.split_text("The Blitz was a bombing campaign by Germany on Britain. Civilians were affected badly—many died, homes were destroyed, and kids had to be sent to safer areas.	")
    Question_requirements = question_analysis.extract_requirements(question)
    categorized = match_units_to_categories(units, Question_requirements, model)

    print("\nSummary:")
    for unit, cat in categorized:
        print(f"'{unit}' → {cat}")
