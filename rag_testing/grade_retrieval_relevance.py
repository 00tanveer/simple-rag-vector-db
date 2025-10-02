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

def ollama_grade_retrieval_relevance(question, student_facts, language_model):
    prompt = (
        f"{retrieval_relevance_instructions}\n"
        f"QUESTION: {question}\n"
        f"STUDENT ANSWER: {student_facts}\n"
        "Grade:\n Respond in JSON with keys 'explanation' and 'retrieval-relevance' (True or False).'"
    )
    response = ollama.chat(model=language_model, messages=[{'role': 'user', 'content': prompt}])
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
    retrieval_relevance_results = dict()
    for i in example_dataset:
        question = i["inputs"]["question"]
         # 1. Get the LLM/RAG retrieved knowledge for the input question
        retrieved_knowledge = retrieve(question, 3, embedding_model)
        # 2. Evaluate relevance
        result = ollama_grade_retrieval_relevance(
            question,
            retrieved_knowledge,  # This is the student LLM retrieved knowledge for the teacher LLM to judge relevant to the question or not
            language_model
        )
        print(f"Q: {i['inputs']['question']}")
        print(f"Result: {result}\n")
        #Add to results dictionary
        retrieval_relevance_results[question] = {
            'retrieval_relevance': result['retrieval-relevance'] if result else None,
            'explanation': result['explanation'] if result else None 
        }
    return retrieval_relevance_results
