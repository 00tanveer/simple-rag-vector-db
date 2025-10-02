import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama 
import json 
from retrieval import retrieve
from generation import generate_response_string
from example_dataset import examples
from grade_correctness import correctness 
from grade_relevance import relevance
from grade_groundedness import groundedness
from grade_retrieval import context_recall 
from grade_retrieval_relevance import retrieval_relevance

EMBEDDING_MODEL = 'mxbai-embed-large:latest'
LANGUAGE_MODEL = 'gemma3:4b'

# correctness_results = correctness(examples, EMBEDDING_MODEL, LANGUAGE_MODEL)
# print(_correctness_results)
# relevance_results = relevance(examples, EMBEDDING_MODEL, LANGUAGE_MODEL)
# print(relevance_results)    
# groundedness_results = groundedness(examples, EMBEDDING_MODEL, LANGUAGE_MODEL)
# print(groundedness_results)
# context_recall_results = context_recall(examples, EMBEDDING_MODEL, LANGUAGE_MODEL)
retrieval_relevance_results = retrieval_relevance(examples, EMBEDDING_MODEL, LANGUAGE_MODEL)
