import React from 'react';

export default function CounterfactualView({ data }) {
  if (!data || data.method !== 'COUNTERFACTUAL') return null;

  const originalVals = data.original_values || {};
  const cfVals = data.counterfactual_values || {};
  const changes = data.changes || {};
  const metrics = data.metrics || {};
  const immutableList = data.immutable_features || [];

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>🔄</span> Causal Counterfactual "What-If" Analysis
        </h3>
        <span className="badge badge-green">
          Prediction Flip: {data.original_pred} ➜ {data.new_pred}
        </span>
      </div>

      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '20px' }}>
        Minimal plausible perturbations required to change model decision (EU AI Act Art. 9 & 15 Recourse).
      </p>

      {/* Metrics Banner */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '12px',
        marginBottom: '20px',
        background: 'rgba(15, 23, 42, 0.6)',
        padding: '12px',
        borderRadius: '12px',
        border: '1px solid var(--border-color)'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>L0 Sparsity</div>
          <div className="font-mono" style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--primary-cyan)' }}>
            {metrics.L0 ?? 0} features
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>L1 Manhattan</div>
          <div className="font-mono" style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--primary-blue)' }}>
            {(metrics.L1 ?? 0).toFixed(4)}
          </div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>L2 Euclidean</div>
          <div className="font-mono" style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-purple)' }}>
            {(metrics.L2 ?? 0).toFixed(4)}
          </div>
        </div>
      </div>

      {/* Comparison Table */}
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
            <th style={{ padding: '8px' }}>Feature Name</th>
            <th style={{ padding: '8px' }}>Original Value</th>
            <th style={{ padding: '8px' }}>Counterfactual Target</th>
            <th style={{ padding: '8px' }}>Status</th>
          </tr>
        </thead>
        <tbody>
          {Object.keys(originalVals).map(feature => {
            const isImmutable = immutableList.includes(feature);
            const isChanged = changes.hasOwnProperty(feature);

            return (
              <tr key={feature} style={{
                borderBottom: '1px solid rgba(255,255,255,0.05)',
                background: isChanged ? 'rgba(0, 242, 254, 0.08)' : 'transparent'
              }}>
                <td style={{ padding: '10px 8px', fontWeight: 600 }}>
                  {isImmutable ? '🔒 ' : ''}{feature}
                </td>
                <td className="font-mono" style={{ padding: '10px 8px' }}>
                  {originalVals[feature]}
                </td>
                <td className="font-mono" style={{ padding: '10px 8px', color: isChanged ? 'var(--primary-cyan)' : 'var(--text-main)', fontWeight: isChanged ? 700 : 400 }}>
                  {cfVals[feature]}
                </td>
                <td style={{ padding: '10px 8px' }}>
                  {isImmutable ? (
                    <span className="badge badge-purple" style={{ fontSize: '0.68rem' }}>Immutable</span>
                  ) : isChanged ? (
                    <span className="badge badge-green" style={{ fontSize: '0.68rem' }}>Perturbed ({changes[feature]?.from} ➜ {changes[feature]?.to})</span>
                  ) : (
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Unchanged</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
