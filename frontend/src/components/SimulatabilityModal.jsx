import React, { useState } from 'react';

export default function SimulatabilityModal({ onEvaluate, result }) {
  const [accuracyPre, setAccuracyPre] = useState(0.50);
  const [accuracyPost, setAccuracyPost] = useState(0.85);

  const handleEvaluate = () => {
    onEvaluate(accuracyPre, accuracyPost);
  };

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span>🧪</span> Level 3: Simulatability Engine (Human-in-the-Loop Test)
      </h3>

      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '20px' }}>
        Measures actual human operator prediction improvement ($Simulatability = Accuracy_{post} - Accuracy_{pre}$). Prevents "Illusion of Understanding".
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
        {/* Pre Accuracy */}
        <div>
          <label style={{ display: 'block', fontSize: '0.8rem', marginBottom: '8px' }}>
            Human Accuracy PRE-Explanation: <strong className="font-mono" style={{ color: 'var(--text-main)' }}>{(accuracyPre * 100).toFixed(0)}%</strong>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={accuracyPre}
            onChange={(e) => setAccuracyPre(parseFloat(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>

        {/* Post Accuracy */}
        <div>
          <label style={{ display: 'block', fontSize: '0.8rem', marginBottom: '8px' }}>
            Human Accuracy POST-Explanation: <strong className="font-mono" style={{ color: 'var(--primary-cyan)' }}>{(accuracyPost * 100).toFixed(0)}%</strong>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={accuracyPost}
            onChange={(e) => setAccuracyPost(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--primary-cyan)' }}
          />
        </div>
      </div>

      <button className="btn-outline" onClick={handleEvaluate} style={{ width: '100%', marginBottom: '16px' }}>
        🧮 Compute Simulatability & Check Illusion Flag
      </button>

      {result && (
        <div style={{
          padding: '16px',
          borderRadius: '12px',
          background: result.illusion_of_understanding ? 'rgba(255, 23, 68, 0.12)' : 'rgba(0, 230, 118, 0.12)',
          border: `1px solid ${result.illusion_of_understanding ? 'var(--accent-red)' : 'var(--accent-green)'}`,
          display: 'flex',
          justify: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Simulatability Score</div>
            <div className="font-mono" style={{
              fontSize: '1.4rem',
              fontWeight: 800,
              color: result.simulatability_score > 0 ? 'var(--accent-green)' : 'var(--accent-red)'
            }}>
              {result.simulatability_score > 0 ? '+' : ''}{(result.simulatability_score * 100).toFixed(1)}%
            </div>
          </div>

          <span className={`badge ${result.illusion_of_understanding ? 'badge-red' : 'badge-green'}`} style={{ padding: '8px 16px', fontSize: '0.85rem' }}>
            {result.illusion_of_understanding ? '⚠️ REJECT: Illusion of Understanding (Score ≤ 0)' : '✅ PASS: Causal Understanding Verified'}
          </span>
        </div>
      )}
    </div>
  );
}
