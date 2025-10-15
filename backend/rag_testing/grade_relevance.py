'''Relevance - Response vs Input 
    - Goal: Measure "how well does the RAG chain answer address the input question"
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

class RelevanceGrade(BaseModel):
    relevant: bool
    explanation: str

def ollama_grade_relevance(question, student_answer, LANGUAGE_MODEL):
    prompt = (
        f"{relevance_instructions}\n"
        f"QUESTION: {question}\n"
        f"STUDENT ANSWER: {student_answer}\n"
        "Grade:\n Respond in JSON with keys 'explanation' and 'relevant' (True or False)."
    )
    response = ollama.chat(
        model=LANGUAGE_MODEL, 
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0, # Deterministic output
            "top_p": 1, # no nucleus sampling
            "top_k": 1, # only pick most likely token
            "seed": 42  # fixed seed
        },
        format=RelevanceGrade.model_json_schema()
    )
    content = response['message']['content'].strip()    

    return json.loads(content)

relevance_instructions = """You are a teacher grading a quiz. You will 
be given a QUESTION and a STUDENT ANSWER. Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of 
the criteria.
A relevance value of False means that the student's answer does not meet 
all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning 
and conclusion are correct. Avoid simply stating the correct answer at the outset."""

def relevance(example_dataset, embedding_model, language_model) -> dict:
    """An evaluator for RAG answer relevance"""
    relevance_results = []
    for i in example_dataset:
        question = i["inputs"]["question"]
        # 1. Get the LLM/RAG response for the input question
        retrieved_knowledge = retrieve(question, 5, embedding_model)
        model_answer = generate_response_string(question, retrieved_knowledge, language_model)
        # 2. Evaluate relevance
        result = ollama_grade_relevance(
            question,
            model_answer,  # This is the student LLM output for the teacher LLM to judge
            language_model
        )
        # print(f"Q: {i['inputs']['question']}")
        # print(f"Result: {result}\n")
        #Add to results dictionary
        relevance_results.append({
            'question': question,
            'model_answer': model_answer,
            'relevant': result['relevant'],
            'explanation': result['explanation']
        })
    return relevance_results