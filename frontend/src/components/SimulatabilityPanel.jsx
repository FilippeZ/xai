import { useState } from 'react';
import { apiPost } from '../api';

export default function SimulatabilityPanel({ onResult }) {
  const [pre, setPre] = useState(0.50);
  const [post, setPost] = useState(0.85);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = async () => {
    setLoading(true); setError(null);
    try {
      const data = await apiPost('/api/simulatability', {
        accuracy_pre: pre,
        accuracy_post: post,
      });
      setResult(data);
      if (onResult) onResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const illusion = result?.illusion_of_understanding;

  return (
    <div className="glass" style={{ padding: 24 }}>
      <h3 className="section-title" style={{ marginBottom: 8 }}>
        🧪 Level 3 — Simulatability Engine
      </h3>
      <p style={{ fontSize: '.78rem', color: 'var(--muted)', marginBottom: 20 }}>
        Measures whether explanations actually improve human operator prediction accuracy.
        Formula: <span className="mono" style={{ color: 'var(--cyan)' }}>
          Simulatability = Acc_post − Acc_pre
        </span>. Score ≤ 0 triggers <strong style={{ color: 'var(--red)' }}>Illusion of Understanding</strong>.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 18 }}>
        {/* PRE */}
        <div>
          <label className="label" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span>Accuracy PRE-Explanation</span>
            <span className="mono" style={{ color: 'var(--muted)', fontWeight: 700 }}>{(pre * 100).toFixed(0)}%</span>
          </label>
          <input type="range" min={0} max={1} step={0.05} value={pre}
            onChange={e => setPre(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--blue)' }} />
        </div>
        {/* POST */}
        <div>
          <label className="label" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span>Accuracy POST-Explanation</span>
            <span className="mono" style={{ color: 'var(--cyan)', fontWeight: 700 }}>{(post * 100).toFixed(0)}%</span>
          </label>
          <input type="range" min={0} max={1} step={0.05} value={post}
            onChange={e => setPost(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--cyan)' }} />
        </div>
      </div>

      <button className="btn-ghost" onClick={run} disabled={loading}
        style={{ width: '100%', justifyContent: 'center', marginBottom: 14 }}>
        {loading ? '⏳ Computing…' : '🧮 Compute Simulatability & Illusion Flag'}
      </button>

      {error && (
        <div style={{ background: 'rgba(239,68,68,.1)', border: '1px solid rgba(239,68,68,.3)', borderRadius: 10, padding: 12, fontSize: '.8rem', color: 'var(--red)', marginBottom: 14 }}>
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div style={{
          borderRadius: 12, padding: '16px 18px',
          background: illusion ? 'rgba(239,68,68,.10)' : 'rgba(34,197,94,.10)',
          border: `1px solid ${illusion ? 'rgba(239,68,68,.3)' : 'rgba(34,197,94,.3)'}`,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 14, flexWrap: 'wrap',
        }}>
          <div>
            <div className="label" style={{ marginBottom: 4 }}>Simulatability Score</div>
            <div className="mono" style={{
              fontSize: '1.5rem', fontWeight: 800,
              color: illusion ? 'var(--red)' : 'var(--green)',
            }}>
              {result.simulatability_score >= 0 ? '+' : ''}{result.simulatability_pct?.toFixed(1)}%
            </div>
          </div>
          <span className={`badge ${illusion ? 'badge-red' : 'badge-green'}`} style={{ fontSize: '.8rem', padding: '8px 14px' }}>
            {illusion
              ? '⚠️ REJECT — Illusion of Understanding (Score ≤ 0)'
              : '✅ PASS — Causal Understanding Verified'}
          </span>
        </div>
      )}
      {result && (
        <p style={{ fontSize: '.75rem', color: 'var(--muted)', marginTop: 10 }}>
          {result.regulation_note}
        </p>
      )}
    </div>
  );
}
