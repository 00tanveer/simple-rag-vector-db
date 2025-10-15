'''
Code for embedding generation and storing embeddings in the database.
Functions to create/update the vector index (e.g., inserting into PostgreSQL with pgvector).
'''
from db import *
import ollama

LANGUAGE_MODEL = 'gemma3:4b'

def create_embeddings(embedding_model):
    print("Creating embeddings for data without embeddings...")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT id, fact FROM cat_facts WHERE embedding IS NULL')
            rows = cur.fetchall()
        for fact_id, fact in rows:
            embedding = ollama.embed(embedding_model, input=fact)['embeddings'][0]
            with conn.cursor() as cur:
                cur.execute('UPDATE cat_facts SET embedding = %s WHERE id = %s', (embedding, fact_id), prepare=False)
        conn.commit()
    finally:
        db_close(conn)