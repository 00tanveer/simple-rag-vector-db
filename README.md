# INTRODUCTION

This is a simple naive retrieval-augmented generation application created to respond to queries about based on facts about cats in a text file.

# BUILDING BLOCKS OF THE PROJECT
1. Data layer - storing data on cat facts in a local `postgres` database and using `pgvector` as a vector extension in postgres to store vector embeddings
2. Knowledge representation - Indexing fixed-size chunks with a local Ollama embedding model `mxbai-embed-large:latest` and storing the embeddings in postgres
3. Retrieval layer - basic cosine similarity search with top-n ranking
4. Generation layer - generation with a local Ollama embedding model `gemma3:4b` with simple system and user prompts

# PROVENANCE
The previous version of this project had these properties:
1. Data layer - local text file and an in-memory Python tuple to store vector embeddings
2. Knowledge representation - Indexing fixed-size chunks with a local Ollama embedding model `mxbai-embed-large:latest`
3. Retrieval layer - basic cosine similarity search with top-n ranking
4. Generation layer - generation with a local Ollama embedding model `gemma3:4b` with simple system and user prompts

# RAG TESTING FRAMEWORK
## RAG EVALUATION
We can view RAG evaluation as a tuple of "what is being evaluated?" vs "what is it being evaluated against?". There are different dimensions of this evaluation routine.
1. Correctness - Response vs Reference answer (how accurate the response is, agnostic of context)
    - Goal: Measure "how similar/correct is the RAG chain answer relative to a ground-truth answer"
    - Evaluator - LLM-as-judge
2. Relevance - Response vs Input 
    - Goal: Measure "how well does the RAG chain answer address the input question"
    - Evaluator - LLM-as-judge
3. Groundedness - Response vs retrieved docs
    - Goal: Measure "to what extent does the generated response agree with the retrieved context"
    - Evaluator - LLM-as-judge
4. Retrieval Relevance - Retrieved docs vs input
    - Goal: Measure "how relevant are my retrieved docs/results for this query
    - Evaluator - LLM-as-judge

![alt text](image.png)


## APPLICATION PERFORMANCE
1. Latency measurement
2. Throughput