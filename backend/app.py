from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from rag_api import RAGSystem
import json

app = Flask(__name__)
CORS(app)

# Initialize RAG once on startup
rag = RAGSystem()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    retrieved = rag.retrieve(query)
    response = rag.generate_response(query, retrieved)
    
    return jsonify({
        'query': query,
        'response': response,
        'retrieved_context': retrieved
    })

@app.route('/api/evaluation-results', methods=['GET'])
def get_eval_results():
    try:
        with open('evaluation_results.json', 'r') as f:
            return jsonify(json.load(f))
    except:
        return jsonify({'error': 'Results not found'}), 404

# Serve frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_files(path):
    return send_from_directory('frontend', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)