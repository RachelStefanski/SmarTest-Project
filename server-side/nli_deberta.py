import numpy as np
from sentence_transformers import CrossEncoder
import torch.nn.functional as F
import torch

def detecting_contradiction(teacher_units, student_unit):
    model = CrossEncoder('cross-encoder/nli-deberta-v3-base', num_labels=3)
    logits = model.predict([(teacher_units, student_unit)], convert_to_numpy=True)
    probs = F.softmax(torch.tensor(logits[0]), dim=0).numpy()
    labels = ['contradiction', 'neutral', 'entailment']
    predicted_label = labels[np.argmax(probs)]
    return predicted_label, probs.max()

if __name__ == "__main__":
    premise = "The Holocaust was systematically carried out by the Nazis through ghettos, mass shootings, deportations, extermination camps like Auschwitz, gas chambers, and forced labor, resulting in the murder of millions of Jews and other minorities."
    hypothesis = "The Holocaust was a peaceful relocation of people to different parts of Europe to protect them from war."
    detecting_contradiction(premise, hypothesis)
