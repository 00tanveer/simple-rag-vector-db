import sys
import os 
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve
from generation import generate_response_string
import ollama
import json

def ollama_grade_context_recall(question, retrieved_knowledge, reference_retrieved_knowledge, LANGUAGE_MODEL):
    prompt = (
        f"{context_recall_instructions}\n"
        f"QUESTION: {question}\n"
        f"RETRIEVED KNOWLEDGE: {retrieved_knowledge}\n"
        f"REFERENCE RETRIEVED KNOWLEDGE: {reference_retrieved_knowledge}\n"
        "Grade:\n Respond in JSON with keys 'explanation' and 'context_recall' (value between 0 and 1)."
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

context_recall_instructions = '''You are a teacher grading a quiz. You will 
be given a QUESTION, the RETRIEVED KNOWLEDGE or facts the student has used to answer the 
question, and the REFERENCE RETRIEVED KNOWLEDGE or grounded truth that all should 
have been used by the student to answer the question. 
You will grade the RETRIEVED KNOWLEDGE based on how well it matches the 
REFERENCE RETRIEVED KNOWLEDGE by calculating something called context recall.

Here follows the step-by-step procedure to calculate the context recall, and later provide an explanation:
(1) Figure out how many distinct facts are there in the REFERENCE RETRIEVED KNOWLEDGE. Call this number A.
(2) Figure out how many of those facts are also present in the RETRIEVED KNOWLEDGE. Call this number B.
(3) Calculate the context recall as B divided by A. This will be a number between 0 and 1.
(4) Explain your reasoning and how you arrived at the numbers A and B, and the final context recall value.
(5) Avoid simply stating the recall value at the outset.

'''

def context_recall(example_dataset, embedding_model, language_model) -> dict:
    '''An evaluator for RAG answer context recall'''
    context_recall_results = dict()
    for i in example_dataset:
        question = i["inputs"]["question"]
        retrieved_knowledge = retrieve(question, 3, embedding_model)
        reference_retrieved_knowledge = i["reference_retrieved_knowledge"]
        result = ollama_grade_context_recall(
            question,
            '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge]),
            '\n'.join(reference_retrieved_knowledge),
            language_model
        )
        print(f"Q: {i['inputs']['question']}")
        print(f"Result: {result['context_recall']}\n")
        print(f"Result: {result['explanation']}\n")
        context_recall_results[question] = {
            'context_recall': result['context_recall'] if result else None,
            'explanation': result['explanation'] if result else None
        }