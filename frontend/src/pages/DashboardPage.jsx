import { useEffect, useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

export default function DashboardPage() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('http://localhost:5000/api/evaluation-results')
      .then(res => {
        setResults(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading results:', err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="text-white text-center p-8">Loading results...</div>;
  }

  if (!results) {
    return <div className="text-red-400 text-center p-8">Failed to load evaluation results</div>;
  }

  // Calculate summary
  const correctnessPass = results.correctness.filter(r => r.correct).length;
  const groundednessPass = results.groundedness.filter(r => r.grounded).length;
  const relevancePass = results.relevance.filter(r => r.relevant).length;
  const avgRecall = (results.context_recall.reduce((sum, r) => sum + r.context_recall, 0) / results.context_recall.length).toFixed(2);

  const summaryData = [
    { metric: 'Correctness', pass: correctnessPass, total: 3 },
    { metric: 'Groundedness', pass: groundednessPass, total: 3 },
    { metric: 'Relevance', pass: relevancePass, total: 3 }
  ];

  const recallData = results.context_recall.map((r, i) => ({
    question: `Q${i + 1}`,
    recall: r.context_recall
  }));

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold text-white mb-6">Evaluation Dashboard</h1>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-slate-800 rounded-lg p-6">
          <div className="text-slate-400 text-sm mb-2">CORRECTNESS</div>
          <div className="text-3xl font-bold text-green-400">{correctnessPass}/3</div>
        </div>
        <div className="bg-slate-800 rounded-lg p-6">
          <div className="text-slate-400 text-sm mb-2">GROUNDEDNESS</div>
          <div className="text-3xl font-bold text-yellow-400">{groundednessPass}/3</div>
        </div>
        <div className="bg-slate-800 rounded-lg p-6">
          <div className="text-slate-400 text-sm mb-2">RELEVANCE</div>
          <div className="text-3xl font-bold text-orange-400">{relevancePass}/3</div>
        </div>
        <div className="bg-slate-800 rounded-lg p-6">
          <div className="text-slate-400 text-sm mb-2">AVG CONTEXT RECALL</div>
          <div className="text-3xl font-bold text-blue-400">{avgRecall}</div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Pass/Fail by Metric</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={summaryData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="metric" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
              <Bar dataKey="pass" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Context Recall</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={recallData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="question" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
              <Line type="monotone" dataKey="recall" stroke="#3b82f6" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}