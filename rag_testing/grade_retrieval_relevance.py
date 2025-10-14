'''
    Retrieval relevance - Retrieved docs vs input
    - Goal: Measure "how relevant my retrieved knowledge is for the query"
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

class RetrievalRelevanceGrade(BaseModel):
    retrieval_relevance: bool
    explanation: str

def ollama_grade_retrieval_relevance(question, student_facts, language_model):
    prompt = (
        f"{retrieval_relevance_instructions}\n"
        f"QUESTION: {question}\n"
        f"STUDENT ANSWER: {student_facts}\n"
        "Grade:\n Respond in JSON with keys 'explanation' and 'retrieval_relevance' (True or False).'"
    )
    response = ollama.chat(
        model=language_model, 
        messages=[{'role': 'user', 'content': prompt}],
        options={
            "temperature": 0, # Deterministic output
            "top_p": 1, # no nucleus sampling
            "top_k": 1, # only pick most likely token
            "seed": 42  # fixed seed
        },
        format=RetrievalRelevanceGrade.model_json_schema()
    )
    content = response['message']['content'].strip()
     # Try to extract JSON if it's wrapped in markdown code blocks
    return json.loads(content)

retrieval_relevance_instructions = '''
You are a teacher grading a quiz. You will be given a QUESTION and a
set of FACTS by the student that relevant to the QUESTION or will help
answer the QUESTION. 
Here is the grading critera to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, 
consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question
as long as (2) is met

Retrieval Relevance:
A retrieval-relevance of of True means that the FACTS contain ANY keywords or 
semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and 
conclusion are correct. Avoid simply stating the correct answer at the outset. 
'''

def retrieval_relevance(example_dataset, embedding_model, language_model) -> dict:
    '''An evaluator for RAG retrieval relevance'''
    retrieval_relevance_results = []
    for i in example_dataset:
        question = i["inputs"]["question"]
         # 1. Get the LLM/RAG retrieved knowledge for the input question
        retrieved_knowledge = retrieve(question, 5, embedding_model)
        # 2. Evaluate relevance
        result = ollama_grade_retrieval_relevance(
            question,
            retrieved_knowledge,  # This is the student LLM retrieved knowledge for the teacher LLM to judge relevant to the question or not
            language_model
        )
        #Add to results dictionary
        retrieval_relevance_results.append({
            'question': question,
            'retrieved_knowledge': retrieved_knowledge,
            'retrieval_relevance': result['retrieval_relevance'],
            'explanation': result['explanation'] 
        })
    return retrieval_relevance_results
