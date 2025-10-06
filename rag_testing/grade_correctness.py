import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve
from generation import generate_response_string
import ollama
import json

def ollama_grade_correctness(question, student_answer, reference_answer, LANGUAGE_MODEL):
    prompt = (
        f"{correctness_instructions}\n"
        f"QUESTION: {question}\n"
        f"GROUND TRUTH ANSWER: {reference_answer}\n"
        f"STUDENT ANSWER: {student_answer}\n"
        "Grade:\n Respond in JSON with keys 'explanation' and 'correct' (True or False)."
    )
    response = ollama.chat(model=LANGUAGE_MODEL, messages=[{"role": "user", "content": prompt}])
    content = response['message']['content'].strip()

     # Try to extract JSON if it's wrapped in markdown code blocks
    if '```json' in content:
        # Extract JSON from markdown code block
        start = content.find('```json') + 7
        end = content.find('```', start)
        content = content[start:end].strip()
    elif '```' in content:
        # Extract from generic code block
        start = content.find('```') + 3
        end = content.find('```', start)
        content = content[start:end].strip()
    
    # Try to parse the JSON from the LLM's response
    try:
        result = json.loads(content)
        return result
    except Exception as e:
        print(e)
        print("Failed to parse LLM response:", response['message']['content'])
        return None

correctness_instructions = """You are a teacher grading a quiz. You will be given 
a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. Here is the 
grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative 
to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the 
ground truth answer, as long as it is factually accurate relative to the 
ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the 
criteria.
A correctness value of False means that the student's answer does not meet all 
of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion 
are correct. Avoid simply stating the correct answer at the outset."""

def correctness(example_dataset, embedding_model, language_model) -> dict:
    """An evaluator for RAG answer accuracy"""
    correctness_results = dict()
    for i in example_dataset:
        question = i["inputs"]["question"]
        # 1. Get the LLM/RAG response for the input question
        retrieved_knowledge = retrieve(question, 3, embedding_model)
        model_answer = generate_response_string(question, retrieved_knowledge, language_model)
        # 2. Evaluate correctness
        result = ollama_grade_correctness(
            question,
            model_answer,  # This is the LLM output
            i["reference_outputs"]["answer"],
            language_model
        )
        print(f"Q: {i['inputs']['question']}")
        print(f"Result: {result}\n")
        print(f"Result: {result['correct']}\n")
        # Add to results dictionary
        correctness_results[question] = {
            'model_answer': model_answer,
            'correct': result['correct'] if result else None,
            'explanation': result['explanation'] if result else None
        }
    return correctness_results