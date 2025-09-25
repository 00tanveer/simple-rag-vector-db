# INTRODUCTION

This is an ultra simple naive retrieval-augmented generation application created to respond to queries about based on facts about cats in a text file.

# BUILDING BLOCKS OF THE PROJECT
1. Data layer - local text file and an in-memory Python tuple to store vector embeddings
2. Knowledge representation - Indexing fixed-size chunks with a local Ollama embedding model `mxbai-embed-large:latest`
3. Retrieval layer - basic cosine similarity search with top-n ranking
4. Generation layer - generation with a local Ollama embedding model `gemma3:4b` with simple system and user prompts
