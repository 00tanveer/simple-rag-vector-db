import { useState } from 'react';
import axios from 'axios';

export default function ChatPage() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    // Add user message
    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/api/chat', {
        query: query
      });

      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: response.data.response,
        context: response.data.retrieved_context
      };
      setMessages(prev => [...prev, assistantMessage]);
      setQuery('');
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        role: 'error',
        content: 'Failed to get response. Is the backend running?'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-white mb-6">RAG Chat Interface</h1>
      
      {/* Messages */}
      <div className="bg-slate-800 rounded-lg p-6 mb-4 h-96 overflow-y-auto">
        {messages.length === 0 ? (
          <p className="text-slate-400 text-center">Ask a question about cats...</p>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`mb-4 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
              <div className={`inline-block p-3 rounded-lg max-w-xl ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white' 
                  : msg.role === 'error'
                  ? 'bg-red-600 text-white'
                  : 'bg-slate-700 text-slate-200'
              }`}>
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className="text-center text-slate-400">
            <p>Thinking...</p>
          </div>
        )}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask me a question..."
          className="flex-1 bg-slate-800 text-white px-4 py-3 rounded-lg border border-slate-700 focus:outline-none focus:border-blue-500"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Send
        </button>
      </form>
    </div>
  );
}