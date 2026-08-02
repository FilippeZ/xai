import React from 'react';

export default function AttributionView({ data }) {
  if (!data || !data.feature_attributions) return null;

  const attributions = Object.entries(data.feature_attributions);
  const maxAbs = Math.max(...attributions.map(([_, v]) => Math.abs(v)), 0.01);

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>📊</span> Feature Attribution Analysis ({data.method})
        </h3>
        <div style={{ display: 'flex', gap: '8px' }}>
          <span className="badge badge-purple">
            Prediction: <strong>{data.predicted_label}</strong>
          </span>
          <span className="badge badge-green">
            Confidence: {(data.confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '20px' }}>
        Feature importance weights assigned by {data.method} (GDPR Art. 22 Right to Explanation).
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {attributions.map(([feature, weight]) => {
          const isPositive = weight >= 0;
          const percentage = Math.min(100, Math.round((Math.abs(weight) / maxAbs) * 100));

          return (
            <div key={feature} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                <span style={{ fontWeight: 600 }}>{feature}</span>
                <span className="font-mono" style={{ color: isPositive ? 'var(--accent-green)' : 'var(--accent-red)', fontWeight: 700 }}>
                  {isPositive ? '+' : ''}{weight.toFixed(4)}
                </span>
              </div>
              
              <div style={{
                height: '10px',
                width: '100%',
                background: 'rgba(255, 255, 255, 0.05)',
                borderRadius: '5px',
                overflow: 'hidden',
                position: 'relative'
              }}>
                <div style={{
                  height: '100%',
                  width: `${percentage}%`,
                  background: isPositive
                    ? 'linear-gradient(90deg, #00e676, #00f2fe)'
                    : 'linear-gradient(90deg, #ff1744, #e100ff)',
                  borderRadius: '5px',
                  transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)'
                }} />
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--border-color)', display: 'flex', gap: '12px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', background: 'var(--accent-green)', borderRadius: '2px' }}></span>
          Positive Impact (+): Supports predicted diagnosis
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', background: 'var(--accent-red)', borderRadius: '2px' }}></span>
          Negative Impact (-): Opposes predicted diagnosis
        </div>
      </div>
    </div>
  );
}
