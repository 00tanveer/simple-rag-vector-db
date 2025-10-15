import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve
from generation import generate_response_string
import ollama
import json
from pydantic import BaseModel

class CorrectnessGrade(BaseModel):
    correct: bool
    explanation: str

def ollama_grade_correctness(question, student_answer, reference_answer, LANGUAGE_MODEL):
    prompt = (
        f"{correctness_instructions}\n"
        f"QUESTION: {question}\n"
        f"GROUND TRUTH ANSWER: {reference_answer}\n"
        f"STUDENT ANSWER: {student_answer}\n"
        "Grade:\n Respond in JSON with keys 'correct' (true or false) and 'explanation'"
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
        format=CorrectnessGrade.model_json_schema()
    )
    content = response['message']['content'].strip()
    return json.loads(content)


correctness_instructions = """You are an expert evaluator grading a student's answer on a quiz question.

Your task: Compare the STUDENT ANSWER against the GROUND TRUTH ANSWER and determine if it is correct.

GRADING CRITERIA:
1. Factual Accuracy: The student answer must be factually accurate relative to the ground truth. All claims must align with or not contradict the reference answer.
2. No Contradictions: The student answer must not contain internal contradictions or statements that conflict with the ground truth.
3. Extra Information is Acceptable: The student may provide additional correct information beyond the ground truth answer, as long as it does not introduce inaccuracies or contradictions.

STEP-BY-STEP PROCESS:
- Examine each factual claim in the student answer
- Check if it aligns with, supports, or contradicts the ground truth answer
- Identify any unsupported claims or hallucinations
- Consider partial correctness: some correct information + some incorrect information = False

OUTPUT: Respond in JSON with this EXACT structure:
{
    "correct": <true> or <false>
    "explanation":  "<string>"
}
"""

def correctness(example_dataset, embedding_model, language_model) -> dict:
    """An evaluator for RAG answer accuracy"""
    correctness_results = []
    for i in example_dataset:
        question = i["inputs"]["question"]
        # 1. Get the LLM/RAG response for the input question
        retrieved_knowledge = retrieve(question, 5, embedding_model)
        model_answer = generate_response_string(question, retrieved_knowledge, language_model)
        # 2. Evaluate correctness
        result = ollama_grade_correctness(
            question,
            model_answer,  # This is the LLM output
            i["reference_outputs"]["answer"],
            language_model
        )
        # print(f"Q: {i['inputs']['question']}")
        # print(f"Result: {result}\n")
        correctness_results.append({
            'question': question,
            'model_answer': model_answer,
            'correct': result['correct'],
            'explanation': result['explanation']
        })
    return correctness_results