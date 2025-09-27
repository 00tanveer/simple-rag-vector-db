import ollama

# import the necessary postgres library and connecting to local postgres database called cat-facts
import psycopg
from psycopg import sql
import os
import ast
from dotenv import load_dotenv
load_dotenv()
EMBEDDING_MODEL = 'mxbai-embed-large:latest'
LANGUAGE_MODEL = 'gemma3:4b'

# connect to the database
conn = psycopg.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
cur = conn.cursor()
# create a table to store the cat facts if it doesn't exist
cur.execute('''
CREATE TABLE IF NOT EXISTS cat_facts (
    id SERIAL PRIMARY KEY,
    fact TEXT NOT NULL,
    embedding VECTOR(1024)
)
''')
conn.commit()
# insert cat facts into the table
# with open("cat-facts.txt", "r") as file:
#     facts = file.readlines()
#     # for just 50 loops
#     facts = facts[:50]
#     for fact in facts:
#         cur.execute('INSERT INTO cat_facts (fact) VALUES (%s)', (fact.strip(),))
# conn.commit()
# # close the connection
# cur.close()



# index cat facts table by using our embedding model and store the embeddings in a pgvector with a lower count column in the cat-facts table
cur = conn.cursor()
conn.commit()
# fetch all facts that don't have embeddings yet
cur.execute('SELECT id, fact FROM cat_facts WHERE embedding IS NULL')
rows = cur.fetchall()
for row in rows:
    fact_id, fact = row
    embedding = ollama.embed(EMBEDDING_MODEL, input=fact)['embeddings'][0]
    cur.execute('UPDATE cat_facts SET embedding = %s WHERE id = %s', (embedding, fact_id))
conn.commit()
# close the connection
cur.close()
conn.close()



def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a**2 for a in vec1) ** 0.5
    norm_b = sum(b**2 for b in vec2) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

# retrieve the top N most similar chunks from the vector database
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(EMBEDDING_MODEL, input=query)['embeddings'][0]
    
    conn = psycopg.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    cur = conn.cursor()
    cur.execute('SELECT fact, embedding FROM cat_facts WHERE embedding IS NOT NULL')
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(rows)
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

# main chat loop
input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)
print('Retrieved knowledge:')

for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''

stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query}
    ],
    stream=True
)

#print the response from the chatbot in real-time
print('Chatbot response:')
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)