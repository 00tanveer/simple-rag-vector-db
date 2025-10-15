from data_pipeline import pipeline_get_raw_data
from db import db_init, db_feed_data_batch
from indexing import create_embeddings
from retrieval import retrieve
from generation import generate_response_string
EMBEDDING_MODEL = 'mxbai-embed-large:latest'
LANGUAGE_MODEL = 'gemma2:2b'

class RAGSystem:
    def __init__(self):
        # LLM parameters (in the future decouple this from client code)
        self.EMBEDDING_MODEL = 'mxbai-embed-large:latest'
        self.LANGUAGE_MODEL = 'gemma2:2b'

        # db_init()
        # db_feed_data_batch(pipeline_get_raw_data())
        # create_embeddings(self.EMBEDDING_MODEL)

    def retrieve(self, query):
        return retrieve(query, 10, self.EMBEDDING_MODEL)
    
    def generate_response(self, query, retrieved_knowledge):
        return generate_response_string(query, retrieved_knowledge, self.LANGUAGE_MODEL)
