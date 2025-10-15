import { useState } from 'react';
import ChatPage from './pages/ChatPage.jsx';
import DashboardPage from './pages/DashboardPage.jsx';

export default function App() {
  const [page, setPage] = useState('chat'); // 'chat' or 'dashboard'

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Navigation */}
      <nav className="bg-slate-800 border-b border-slate-700 p-4">
        <div className="max-w-7xl mx-auto flex gap-4">
          <button
            onClick={() => setPage('chat')}
            className={`px-4 py-2 rounded ${page === 'chat' ? 'bg-blue-600 text-white' : 'text-slate-300'}`}
          >
            Chat
          </button>
          <button
            onClick={() => setPage('dashboard')}
            className={`px-4 py-2 rounded ${page === 'dashboard' ? 'bg-blue-600 text-white' : 'text-slate-300'}`}
          >
            Evaluation Dashboard
          </button>
        </div>
      </nav>

      {/* Pages */}
      <div className="max-w-7xl mx-auto">
        {page === 'chat' && <ChatPage />}
        {page === 'dashboard' && <DashboardPage />}
      </div>
    </div>
  );
}