'''
Code to take retrieved chunks and generate responses using your LLM.
Orchestrates the full RAG pipeline: takes a query, retrieves context, and generates an answer.
'''
import ollama

def generate_response(query, retrieved_knowledge, language_model):
    print("Generating response from language model...")
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk}')
    instruction_prompt = f'''You are a helpful chatbot answering questions on cats.
    Use only the following pieces of context to answer the question. Ignore irrelevant context. Don't make up any new information or inconsistent facts about cats.
    Aggregate similar information from the context. Don't be redundant.
    Be grammatically and semantically correct:
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    '''

    stream = ollama.chat(
        model=language_model,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': query}
        ],
        options={
            "temperature": 0, # Deterministic output
            "top_p": 1, # no nucleus sampling
            "top_k": 1, # only pick most likely token
            "seed": 42  # fixed seed
        },
        stream=True
    )

    #print the response from the chatbot in real-time
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

def generate_response_string(query, retrieved_knowledge, language_model):
    """Generate a response from the language model and return it as a string."""
    # for chunk, similarity in retrieved_knowledge:
    #     print(f' - (similarity: {similarity:.2f}) {chunk}')
    instruction_prompt = f'''You are a helpful chatbot answering questions on cats.
    Use only the following pieces of context to answer the question. Ignore irrelevant context. Don't make up any new information or inconsistent facts about cats.
    Aggregate similar information from the context. Don't be redundant.
    Be grammatically and semantically correct:\n{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}\n'''

    stream = ollama.chat(
        model=language_model,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': query}
        ],
        stream=True
    )

    response = ""
    for chunk in stream:
        response += chunk['message']['content']
    return response