import sys
import os 
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve
from generation import generate_response_string
import ollama
import json
from pydantic import BaseModel

class ContextRecallGrade(BaseModel):
    context_recall: float
    explanation: str

def ollama_grade_context_recall(question, retrieved_knowledge, reference_retrieved_knowledge, LANGUAGE_MODEL):
    prompt = (
        f"{context_recall_instructions}\n"
        f"QUESTION: {question}\n"
        f"RETRIEVED KNOWLEDGE: {retrieved_knowledge}\n"
        f"REFERENCE RETRIEVED KNOWLEDGE: {reference_retrieved_knowledge}\n"
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
        format=ContextRecallGrade.model_json_schema()
    )
    content = response['message']['content'].strip()    
    return json.loads(content)

context_recall_instructions = '''
ROLE: You are evaluating context retrieval quality of a retrieval-augmented generation 
application. 

INPUT: You are given:
- QUESTION: The user's question
- RETRIEVED KNOWLEDGE: The facts actually retrieved by the system
- REFERENCE RETRIEVED KNOWLEDGE: The ideal facts that should have been retrieved

TASK: Calculate context recall = (number of reference facts found in retrieved knowledge) / (total reference facts)

STEP-BY-STEP PROCESS:
(1) Observe the factsfrom REFERENCE RETRIEVED KNOWLEDGE. Number 
them clearly (Fact 1, Fact 2, etc.)
(2) For EACH reference fact, check if the SAME INFORMATION appears in RETRIEVED KNOWLEDGE:
    - Facts match if they convey the same meaning (exact wording not required)
    - Example: "Cats get cancer" matches "Cats can develop cancer"
    - Example: "Cats are subject to gum disease" matches "Cats get gum disease."
    - Example: "Cats get tapeworms from mice" does NOT match "Cats get tapeworms" (missing source)
    Mark each fact as:
        - FOUND: if equivalent information exists in RETRIEVED KNOWLEDGE
        - NOT FOUND: if missing or contradicted
(3) Count: A = total reference facts, B = facts found
(4) Calculate: context_recall = B / A (a float between 0 and 1)
(5) In your explanation field, show your work step-by-step before stating the final context recall value.

CRITICAL rules/CONSTRAINTS:
- Two facts are "the same" if they convey the same information, even with different wording
- Do not hallucinate facts not present in the text
- Be consistent: if you mark a fact as found, it must actually be present in RETRIEVED KNOWLEDGE

OUTPUT: Respond in JSON with this EXACT structure:
{
  "context_recall": <float>,                     
  "explanation": "<string>"                       
}

Output in JSON.
'''

def context_recall(example_dataset, embedding_model, language_model) -> dict:
    '''An evaluator for RAG answer context recall'''
    context_recall_results = []
    for i in example_dataset:
        question = i["inputs"]["question"]
        retrieved_knowledge = retrieve(question, 5, embedding_model)
        reference_retrieved_knowledge = i["reference_retrieved_knowledge"]
        result = ollama_grade_context_recall(
            question,
            '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge]),
            '\n'.join(reference_retrieved_knowledge),
            language_model
        )
        # result = ContextRecallGrade.model_validate_json(result)
        # print(result)
        context_recall_results.append({
            'question': question,
            'context_recall': result['context_recall'],
            'explanation': result['explanation']
        })
    return context_recall_results