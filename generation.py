'''
Code to take retrieved chunks and generate responses using your LLM.
Orchestrates the full RAG pipeline: takes a query, retrieves context, and generates an answer.
'''
import ollama

def generate_response(query, retrieved_knowledge, language_model):
    print("Generating response from language model...")
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk}')
    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information:
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    '''

    stream = ollama.chat(
        model=language_model,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': query}
        ],
        stream=True
    )

    #print the response from the chatbot in real-time
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)