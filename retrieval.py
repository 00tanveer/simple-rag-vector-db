'''
Functions to query the vector database for similar chunks.
Cosine similarity or other retrieval logic.
'''
import ollama
import ast
from db import *

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a**2 for a in vec1) ** 0.5
    norm_b = sum(b**2 for b in vec2) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

# retrieve the top N most similar chunks from the vector database
def retrieve(query, top_n, emb):
    print("Retrieving relevant knowledge from database...")
    query_embedding = ollama.embed(emb, input=query)['embeddings'][0]
    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT fact, embedding FROM cat_facts WHERE embedding IS NOT NULL')
    rows = cur.fetchall()
    cur.close()
    db_close(conn)
    #temporary list to hold (chunk, similarity) tuples
    similarities = []
        
    for fact, emb in rows:
        # Convert embedding from string to list if needed
        if isinstance(emb, str):
            emb = ast.literal_eval(emb)
        similarities.append((fact, cosine_similarity(query_embedding, emb)))
    
    # sort by similarity score in descending order, because higher means more similar
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # finally return the top N most similar chunks
    return similarities[:top_n]