import GradCamView from './GradCamView';

export default function ResultsPane({ data, method }) {
  if (!data) return <EmptyState />;

  const m = (data.method || method || '').toUpperCase().replace('-', '');

  if (m === 'GRADCAM' || data.method === 'GRAD-CAM') return <GradCamView data={data} />;
  if (m === 'COUNTERFACTUAL') return <CounterfactualResult data={data} />;
  if (m === 'LIMETEXT') return <TextLimeResult data={data} />;
  return <AttributionResult data={data} />;  // SHAP or LIME tabular
}

/* ── Empty ── */
function EmptyState() {
  return (
    <div className="glass" style={{ padding: 40, textAlign: 'center', color: 'var(--dim)', fontSize: '.9rem' }}>
      <div style={{ fontSize: '2.8rem', marginBottom: 14 }}>🔬</div>
      <div>Select a technique and click <strong style={{ color: 'var(--cyan)' }}>⚡ RUN</strong> to see live XAI results here.</div>
      <div style={{ marginTop: 10, fontSize: '.78rem', color: 'var(--dim)' }}>
        Supports Tabular (SHAP / LIME / Counterfactual), Image (Grad-CAM), and Text (LIME)
      </div>
    </div>
  );
}

/* ── SHAP / Tabular LIME ── */
function AttributionResult({ data }) {
  const attrs = data.feature_attributions || {};
  const entries = Object.entries(attrs).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.0001);

  return (
    <div className="glass" style={{ padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 18 }}>
        <h3 className="section-title">
          {data.method === 'SHAP' ? '💜' : '🍊'} {data.method} Feature Attribution
        </h3>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <span className="badge badge-purple">Prediction: <strong style={{ marginLeft: 4 }}>{data.predicted_label}</strong></span>
          <span className="badge badge-cyan">True: {data.true_label}</span>
          <span className="badge badge-green">Confidence: {(data.confidence * 100).toFixed(1)}%</span>
          {data.elapsed_ms && <span className="badge badge-amber">⚡ {data.elapsed_ms} ms</span>}
        </div>
      </div>

      <p style={{ fontSize: '.8rem', color: 'var(--muted)', marginBottom: 20 }}>
        Per-feature contribution to the predicted outcome — <span style={{ color: 'var(--purple)' }}>GDPR Art. 22 Right to Explanation</span>.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {entries.map(([feat, val]) => {
          const pos = val >= 0;
          const pct = Math.round((Math.abs(val) / maxAbs) * 100);
          return (
            <div key={feat}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5, fontSize: '.84rem' }}>
                <span style={{ fontWeight: 600 }}>{feat}</span>
                <span className="mono" style={{ color: pos ? 'var(--green)' : 'var(--red)', fontWeight: 700, fontSize: '.82rem' }}>
                  {pos ? '+' : ''}{val.toFixed(5)}
                </span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{
                  width: `${pct}%`,
                  background: pos
                    ? 'linear-gradient(90deg,#22c55e,#00d2dc)'
                    : 'linear-gradient(90deg,#ef4444,#e100ff)',
                }} />
              </div>
            </div>
          );
        })}
      </div>

      {/* Instance values */}
      <div style={{ marginTop: 22, borderTop: '1px solid var(--border)', paddingTop: 14 }}>
        <p className="label" style={{ marginBottom: 10 }}>Original Instance Values</p>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {Object.entries(data.original_values || {}).map(([feat, val]) => (
            <div key={feat} style={{
              background: 'rgba(255,255,255,.04)', border: '1px solid var(--border)',
              borderRadius: 8, padding: '7px 12px', fontSize: '.8rem',
            }}>
              <div style={{ color: 'var(--muted)', fontSize: '.7rem' }}>{feat}</div>
              <div className="mono" style={{ fontWeight: 700, color: 'var(--cyan)' }}>{val}</div>
            </div>
          ))}
        </div>
      </div>

      {/* LIME local prediction */}
      {data.local_prediction && (
        <div style={{ marginTop: 14, fontSize: '.8rem' }}>
          <span className="label">Local Surrogate Prediction: </span>
          {Object.entries(data.local_prediction).map(([cls, p]) => (
            <span key={cls} style={{ marginRight: 12 }}>
              <strong style={{ color: 'var(--text)' }}>{cls}</strong>:
              <span style={{ color: 'var(--muted)' }}> {(p * 100).toFixed(1)}%</span>
            </span>
          ))}
        </div>
      )}

      <div style={{ marginTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {(data.compliance_tags || []).map(tag => (
          <span key={tag} className="badge badge-purple" style={{ fontSize: '.66rem' }}>🏷️ {tag}</span>
        ))}
      </div>
    </div>
  );
}

/* ── COUNTERFACTUAL ── */
function CounterfactualResult({ data }) {
  const origVals = data.original_values || {};
  const cfVals = data.counterfactual_values || {};
  const changes = data.changes || {};
  const metrics = data.metrics || {};
  const immutable = data.immutable_features || [];

  return (
    <div className="glass" style={{ padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 18 }}>
        <h3 className="section-title">🔄 Counterfactual "What-If" Analysis</h3>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <span className={`badge ${data.success ? 'badge-green' : 'badge-red'}`}>
            {data.success ? '✅ Flip Achieved' : '❌ No Flip Found'}
          </span>
          <span className="badge badge-purple">{data.original_pred} ➜ {data.new_pred}</span>
          {data.elapsed_ms && <span className="badge badge-amber">⚡ {data.elapsed_ms} ms</span>}
        </div>
      </div>

      <p style={{ fontSize: '.8rem', color: 'var(--muted)', marginBottom: 16 }}>
        Minimal plausible changes to flip the model decision — <span style={{ color: 'var(--purple)' }}>EU AI Act Art. 9 &amp; 15 Recourse</span>.
      </p>

      {/* Gates */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 18, flexWrap: 'wrap' }}>
        <span className={`badge ${data.plausibility_check ? 'badge-green' : 'badge-red'}`}>
          {data.plausibility_check ? '✅' : '❌'} Plausibility Gate
        </span>
        <span className={`badge ${data.immutable_constraints_satisfied ? 'badge-green' : 'badge-red'}`}>
          {data.immutable_constraints_satisfied ? '🔒' : '⚠️'} Immutable Constraints
        </span>
      </div>

      {/* Metrics */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10,
        background: 'rgba(7,14,32,.7)', borderRadius: 12, padding: '14px 18px',
        border: '1px solid var(--border)', marginBottom: 22,
      }}>
        {[
          { label: 'L₀ Sparsity', val: `${metrics.L0 ?? 0} features`, color: 'var(--cyan)' },
          { label: 'L₁ Manhattan', val: (metrics.L1 ?? 0).toFixed(4), color: 'var(--blue)' },
          { label: 'L₂ Euclidean', val: (metrics.L2 ?? 0).toFixed(4), color: 'var(--purple)' },
        ].map(m => (
          <div key={m.label} style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '.72rem', color: 'var(--muted)', marginBottom: 4 }}>{m.label}</div>
            <div className="mono" style={{ fontWeight: 800, fontSize: '1.05rem', color: m.color }}>{m.val}</div>
          </div>
        ))}
      </div>

      {/* Comparison table */}
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.83rem' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--muted)', textAlign: 'left' }}>
            {['Feature', 'Original', 'Counterfactual', 'Status'].map(h => (
              <th key={h} style={{ padding: '7px 8px' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.keys(origVals).map(feat => {
            const isImm = immutable.includes(feat);
            const changed = feat in changes;
            return (
              <tr key={feat} style={{
                borderBottom: '1px solid rgba(255,255,255,.04)',
                background: changed ? 'rgba(0,210,220,.06)' : 'transparent',
              }}>
                <td style={{ padding: '9px 8px', fontWeight: 600 }}>{isImm ? '🔒 ' : ''}{feat}</td>
                <td className="mono" style={{ padding: '9px 8px' }}>{origVals[feat]}</td>
                <td className="mono" style={{ padding: '9px 8px', color: changed ? 'var(--cyan)' : 'var(--text)', fontWeight: changed ? 700 : 400 }}>
                  {cfVals[feat]}
                </td>
                <td style={{ padding: '9px 8px' }}>
                  {isImm
                    ? <span className="badge badge-purple" style={{ fontSize: '.65rem' }}>Immutable</span>
                    : changed
                      ? <span className="badge badge-green" style={{ fontSize: '.65rem' }}>
                          Δ {changes[feat]?.from} → {changes[feat]?.to}
                        </span>
                      : <span style={{ color: 'var(--dim)', fontSize: '.75rem' }}>Unchanged</span>
                  }
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div style={{ marginTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {(data.compliance_tags || []).map(tag => (
          <span key={tag} className="badge badge-purple" style={{ fontSize: '.66rem' }}>🏷️ {tag}</span>
        ))}
      </div>
    </div>
  );
}

/* ── TEXT LIME ── */
function TextLimeResult({ data }) {
  const wordAttrs = data.word_attributions || {};
  const entries = Object.entries(wordAttrs).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.0001);

  return (
    <div className="glass" style={{ padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <h3 className="section-title">🍊 LIME Text Explanation</h3>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <span className="badge badge-purple">Predicted: <strong style={{ marginLeft: 4 }}>{data.predicted_label}</strong></span>
          <span className="badge badge-cyan">True: {data.true_label}</span>
          <span className="badge badge-green">Confidence: {(data.confidence * 100).toFixed(1)}%</span>
          {data.elapsed_ms && <span className="badge badge-amber">⚡ {data.elapsed_ms} ms</span>}
        </div>
      </div>

      {/* Text preview */}
      {data.text_preview && (
        <div style={{
          background: 'rgba(7,14,32,.7)', border: '1px solid var(--border)',
          borderRadius: 10, padding: '12px 14px', marginBottom: 20,
          fontSize: '.8rem', lineHeight: 1.6, color: 'var(--muted)',
          maxHeight: 100, overflow: 'auto',
        }}>
          {data.text_preview}
        </div>
      )}

      <p className="label" style={{ marginBottom: 10 }}>Top Influential Words (Local Surrogate)</p>

      {/* Word bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {entries.map(([word, val]) => {
          const pos = val >= 0;
          const pct = Math.round((Math.abs(val) / maxAbs) * 100);
          return (
            <div key={word}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: '.84rem' }}>
                <span style={{
                  fontWeight: 700, padding: '2px 10px', borderRadius: 6,
                  background: pos ? 'rgba(34,197,94,.12)' : 'rgba(239,68,68,.12)',
                  color: pos ? 'var(--green)' : 'var(--red)',
                  fontFamily: 'var(--mono)',
                }}>"{word}"</span>
                <span className="mono" style={{ color: pos ? 'var(--green)' : 'var(--red)', fontWeight: 700 }}>
                  {pos ? '+' : ''}{val.toFixed(5)}
                </span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{
                  width: `${pct}%`,
                  background: pos
                    ? 'linear-gradient(90deg,#22c55e,#00d2dc)'
                    : 'linear-gradient(90deg,#ef4444,#e100ff)',
                }} />
              </div>
            </div>
          );
        })}
      </div>

      {/* Local prediction */}
      {data.local_prediction && (
        <div style={{ marginTop: 16, fontSize: '.8rem' }}>
          <span className="label">Local Surrogate Prediction: </span>
          {Object.entries(data.local_prediction).map(([cls, p]) => (
            <span key={cls} style={{ marginRight: 12 }}>
              <strong style={{ color: 'var(--text)' }}>{cls}</strong>
              <span style={{ color: 'var(--muted)' }}> {(p * 100).toFixed(1)}%</span>
            </span>
          ))}
        </div>
      )}

      <div style={{ marginTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {(data.compliance_tags || []).map(tag => (
          <span key={tag} className="badge badge-purple" style={{ fontSize: '.66rem' }}>🏷️ {tag}</span>
        ))}
      </div>
    </div>
  );
}
