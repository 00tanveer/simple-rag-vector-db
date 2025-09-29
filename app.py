import ollama
import ast

from data_pipeline import *
from db import *
from indexing import *
from retrieval import * 
from generation import *
EMBEDDING_MODEL = 'mxbai-embed-large:latest'
LANGUAGE_MODEL = 'gemma3:4b'

db_init()
db_feed_data_batch(pipeline_get_raw_data())
create_embeddings(EMBEDDING_MODEL)
# main chat loop
input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query, top_n=3, EMBEDDING_MODEL)
# print('Retrieved knowledge:', retrieved_knowledge)
generate_response(input_query, retrieved_knowledge, LANGUAGE_MODEL)