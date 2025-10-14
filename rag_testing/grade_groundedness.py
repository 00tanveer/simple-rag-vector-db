'''
Groundedness - Response vs retrieved docs
    - Goal: Measure "to what extent does the generated response 
    agree with the retrieved context"
'''
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve
from generation import generate_response_string
import ollama
import json
from pydantic import BaseModel

class GroundednessGrade(BaseModel):
    grounded: bool
    explanation: str

def ollama_grade_groundedness(question, student_answer, retrieved_docs, LANGUAGE_MODEL):
    prompt = (
        f"{grounded_instructions}\n"
        f"FACTS: {retrieved_docs}\n"
        f"STUDENT ANSWER: {student_answer}\n"
        "Grade:\n Respond in JSON with keys 'explanation' and 'grounded' (True or False)."
    )
    response = ollama.chat(
        model=LANGUAGE_MODEL, 
        messages=[{"role": "user", "content": prompt,
        }],
        options={
            "temperature": 0, # Deterministic output
            "top_p": 1, # no nucleus sampling
            "top_k": 1, # only pick most likely token
            "seed": 42  # fixed seed
        },
        format=GroundednessGrade.model_json_schema()
    )
    content = response['message']['content'].strip()

    return json.loads(content)

# Grade prompt
grounded_instructions = """You are a teacher grading a quiz. You will be 
given FACTS and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information 
outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the 
criteria.
A grounded value of False means that the student's answer does not meet 
all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning 
and conclusion are correct. Avoid simply stating the correct answer at 
the outset."""

def groundedness(example_dataset, embedding_model, language_model) -> dict:
    """An evaluator for RAG answer groundedness"""
    groundedness_results = []
    for i in example_dataset:
        question = i["inputs"]["question"]
        # 1. Get the LLM/RAG response for the input question
        retrieved_knowledge = retrieve(question, 5, embedding_model)
        model_answer = generate_response_string(question, retrieved_knowledge, language_model)
        # 2. Evaluate groundedness
        result = ollama_grade_groundedness(
            question,
            model_answer,  # This is the LLM output
            '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge]),
            language_model
        )
        groundedness_results.append({
            'question': question,
            'model_answer': model_answer,
            'grounded': result['grounded'],
            'explanation': result['explanation'] 
        })
    return groundedness_results