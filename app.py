import ollama

EMBEDDING_MODEL = 'mxbai-embed-large:latest'
LANGUAGE_MODEL = 'gemma3:4b'

dataset = []

with open("cat-facts.txt", "r") as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')

# Each element in the VECTOR_DB will be a tuple of (chunk of text, embedding)
# The embedding is a list of floats, for example: [0.123, 0.456, ...]
VECTOR_DB = []

def add_chunk_to_vector_db(chunk):
    embedding = ollama.embed(EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    add_chunk_to_vector_db(chunk)
    # print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a**2 for a in vec1) ** 0.5
    norm_b = sum(b**2 for b in vec2) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(EMBEDDING_MODEL, input=query)['embeddings'][0]
    
    #temporary list to hold (chunk, similarity) tuples
    similarities = []
    similarities = [(chunk, cosine_similarity(query_embedding, emb)) for chunk, emb in VECTOR_DB]
    # sort by similarity score in descending order, because higher means more similar
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]
    # finally return the top N most similar chunks
    return similarities[:top_n]

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