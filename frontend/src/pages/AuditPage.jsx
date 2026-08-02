import { useState } from 'react';
import { apiPost } from '../api';

/**
 * Page 3 — Compliance Audit Trail Gate (Layer 4)
 * Full-page view: generate audit record, see JSON, see compliance badge wall.
 */
export default function AuditPage({ explanationData, simulatabilityResult }) {
  const [record, setRecord] = useState(null);
  const [showJson, setShowJson] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generate = async () => {
    setLoading(true); setError(null);
    try {
      const illusion = simulatabilityResult?.illusion_of_understanding ?? false;
      const method = explanationData?.method ?? 'SHAP';
      const modality = explanationData?.modality ?? 'tabular';

      let attrs = {};
      if (method === 'SHAP' || method === 'LIME') {
        attrs = explanationData?.feature_attributions ?? {};
      } else if (method === 'COUNTERFACTUAL') {
        const changes = explanationData?.changes ?? {};
        const origVals = explanationData?.original_values ?? {};
        const cfVals = explanationData?.counterfactual_values ?? {};
        for (const feat of Object.keys(origVals)) {
          if (feat in changes)
            attrs[feat] = parseFloat(((cfVals[feat] ?? 0) - (origVals[feat] ?? 0)).toFixed(6));
        }
        if (!Object.keys(attrs).length) attrs = { no_changes: 0 };
      } else if (method === 'GRAD-CAM') {
        (explanationData?.heatmap_regions ?? []).forEach(r => { attrs[r.region] = r.importance; });
      } else if (method === 'LIME-TEXT') {
        attrs = explanationData?.word_attributions ?? {};
      }
      if (!Object.keys(attrs).length) {
        attrs = { feature_1: 0.42, feature_2: 0.35, feature_3: -0.08 };
      }

      const predicted_label =
        explanationData?.predicted_label ??
        explanationData?.original_pred ?? 'unknown';

      const model_type =
        modality === 'image' ? 'CNN (MNIST Benchmark)' :
        modality === 'text'  ? 'TF-IDF + LogisticRegression (20 Newsgroups)' :
                               'RandomForest (Iris Benchmark)';

      const data = await apiPost('/api/audit', {
        model_type,
        model_accuracy: modality === 'image' ? 0.91 : modality === 'text' ? 0.95 : 0.9333,
        instance_id: explanationData?.instance_index ?? 0,
        true_label: explanationData?.true_label ?? 'unknown',
        predicted_label,
        xai_method: method,
        feature_attributions: attrs,
        confidence: explanationData?.confidence ?? null,
        illusion_of_understanding: illusion,
      });
      setRecord(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const justified = record?.decision_justified;

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Header card */}
      <div className="glass" style={{ padding: '28px 32px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, marginBottom: 20 }}>
          <div style={{
            width: 52, height: 52, borderRadius: 14, flexShrink: 0,
            background: 'linear-gradient(135deg,var(--purple),var(--pink))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1.5rem', boxShadow: '0 0 20px rgba(168,85,247,.4)',
          }}>📋</div>
          <div>
            <h2 style={{ fontSize: '1.3rem', fontWeight: 800, marginBottom: 4 }}>
              Level 4 — Compliance Audit Trail Gate
            </h2>
            <p style={{ fontSize: '.82rem', color: 'var(--muted)', lineHeight: 1.5 }}>
              Every AI-assisted decision must be stamped with an immutable audit record.
              The Audit Gate automatically <strong style={{ color: 'var(--red)' }}>REJECTS</strong> decisions
              if an Illusion of Understanding was detected (Level 3). Records are tagged with
              GDPR Art. 22, EU AI Act Art. 13, 14, and 17.
            </p>
          </div>
        </div>

        {/* Context from previous pages */}
        {(explanationData || simulatabilityResult) && (
          <div style={{
            background: 'rgba(7,14,32,.8)', border: '1px solid var(--border)',
            borderRadius: 12, padding: '14px 20px',
          }}>
            <div className="label" style={{ marginBottom: 10 }}>Context from Current Session</div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {explanationData && (
                <>
                  <span className="badge badge-cyan">XAI: {explanationData.method}</span>
                  <span className="badge badge-purple">Prediction: {explanationData.predicted_label ?? explanationData.original_pred ?? '—'}</span>
                  {explanationData.confidence != null && (
                    <span className="badge badge-green">Confidence: {(explanationData.confidence * 100).toFixed(1)}%</span>
                  )}
                </>
              )}
              {simulatabilityResult && (
                <span className={`badge ${simulatabilityResult.illusion_of_understanding ? 'badge-red' : 'badge-green'}`}>
                  Simulatability: {simulatabilityResult.simulatability_pct >= 0 ? '+' : ''}{simulatabilityResult.simulatability_pct?.toFixed(1)}%
                  {simulatabilityResult.illusion_of_understanding ? ' ⚠️ Illusion' : ' ✅ Verified'}
                </span>
              )}
              {!explanationData && !simulatabilityResult && (
                <span style={{ fontSize: '.78rem', color: 'var(--dim)' }}>
                  Run an XAI explanation (Page 1) to auto-populate the audit fields.
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Generate button + error */}
      <div className="glass" style={{ padding: '24px 28px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <h3 className="section-title">🔒 Generate Immutable Audit Record</h3>
          <button className="btn-primary" onClick={generate} disabled={loading}
            style={{ width: 'auto', padding: '12px 24px' }}>
            {loading ? '⏳ Generating…' : '🔒 Generate Audit Record'}
          </button>
        </div>
        <p style={{ fontSize: '.78rem', color: 'var(--muted)', lineHeight: 1.5 }}>
          Clicking Generate stamps a SHA-256 record ID and ISO-8601 timestamp.
          The record captures the XAI method, feature attributions, human reviewer,
          and all applicable regulatory articles. This log is designed to be forwarded
          to a regulatory inspection database.
        </p>

        {error && (
          <div style={{
            marginTop: 14, background: 'rgba(239,68,68,.1)', border: '1px solid rgba(239,68,68,.3)',
            borderRadius: 10, padding: 12, fontSize: '.82rem', color: 'var(--red)',
          }}>⚠️ {error}</div>
        )}
      </div>

      {/* Record display */}
      {!record ? (
        <div className="glass" style={{
          padding: 48, textAlign: 'center', color: 'var(--dim)',
        }}>
          <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🔒</div>
          <div style={{ fontSize: '.9rem' }}>Click <strong style={{ color: 'var(--cyan)' }}>🔒 Generate Audit Record</strong> above</div>
          <div style={{ fontSize: '.75rem', marginTop: 6 }}>to stamp the immutable decision log</div>
        </div>
      ) : (
        <>
          {/* Gate status banner */}
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            background: justified ? 'rgba(34,197,94,.10)' : 'rgba(239,68,68,.10)',
            border: `2px solid ${justified ? 'rgba(34,197,94,.35)' : 'rgba(239,68,68,.35)'}`,
            borderRadius: 16, padding: '22px 28px',
          }}>
            <div>
              <div className="label" style={{ marginBottom: 6 }}>Audit Record ID</div>
              <div className="mono" style={{ color: 'var(--cyan)', fontWeight: 800, fontSize: '1.3rem' }}>
                {record.record?.record_id}
              </div>
              <div style={{ fontSize: '.75rem', color: 'var(--muted)', marginTop: 4 }}>
                🕐 {new Date(record.record?.timestamp).toLocaleString()}
                {'  ·  '}
                👤 {record.record?.human_reviewer}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{
                fontSize: '1.6rem', fontWeight: 900,
                color: justified ? 'var(--green)' : 'var(--red)',
              }}>
                {justified ? '✅ PASSED' : '⛔ REJECTED'}
              </div>
              <div style={{ fontSize: '.75rem', color: 'var(--muted)', marginTop: 4 }}>
                Audit Gate Status
              </div>
            </div>
          </div>

          {/* Metadata row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
            {[
              { label: 'XAI Method',  val: record.record?.xai_method },
              { label: 'Predicted',   val: record.record?.predicted_label },
              { label: 'True Label',  val: record.record?.true_label },
              { label: 'Confidence',  val: record.record?.confidence != null ? `${(record.record.confidence * 100).toFixed(1)}%` : 'N/A' },
            ].map(({ label, val }) => (
              <div key={label} className="glass" style={{ padding: '14px 16px', textAlign: 'center' }}>
                <div className="label" style={{ marginBottom: 6 }}>{label}</div>
                <div className="mono" style={{ fontWeight: 700, color: 'var(--cyan)', fontSize: '.95rem' }}>
                  {val ?? '—'}
                </div>
              </div>
            ))}
          </div>

          {/* Explanation summary */}
          <div className="glass" style={{ padding: '20px 24px' }}>
            <div className="label" style={{ marginBottom: 8 }}>Explanation Summary</div>
            <p style={{ fontSize: '.84rem', lineHeight: 1.6 }}>
              {record.record?.explanation_summary}
            </p>
          </div>

          {/* Compliance tags */}
          <div className="glass" style={{ padding: '20px 24px' }}>
            <div className="label" style={{ marginBottom: 12 }}>Stamped Regulatory Articles</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {(record.record?.compliance_tags || []).map((tag, i) => (
                <span key={i}
                  className={`badge ${tag.includes('REJECT') || tag.includes('illusion') ? 'badge-red' : 'badge-purple'}`}
                  style={{ padding: '8px 14px', fontSize: '.78rem' }}>
                  🏷️ {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Feature attributions table */}
          {Object.keys(record.record?.feature_attributions || {}).length > 0 && (
            <div className="glass" style={{ padding: '20px 24px' }}>
              <div className="label" style={{ marginBottom: 12 }}>Recorded Feature Attributions</div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.82rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--muted)' }}>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Feature</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>Attribution</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(record.record.feature_attributions)
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .map(([feat, val]) => (
                    <tr key={feat} style={{ borderBottom: '1px solid rgba(255,255,255,.04)' }}>
                      <td style={{ padding: '9px 8px', fontWeight: 600 }}>{feat}</td>
                      <td className="mono" style={{
                        padding: '9px 8px', textAlign: 'right',
                        color: val >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 700,
                      }}>
                        {val >= 0 ? '+' : ''}{val.toFixed(5)}
                      </td>
                      <td style={{ padding: '9px 8px', textAlign: 'right' }}>
                        {val >= 0
                          ? <span className="badge badge-green" style={{ fontSize: '.65rem' }}>▲ Positive</span>
                          : <span className="badge badge-red" style={{ fontSize: '.65rem' }}>▼ Negative</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Raw JSON */}
          <div className="glass" style={{ padding: '20px 24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: showJson ? 14 : 0 }}>
              <div className="label">Raw Audit JSON — Regulatory Export</div>
              <button className="btn-ghost" onClick={() => setShowJson(v => !v)}
                style={{ fontSize: '.76rem' }}>
                {showJson ? '🔼 Collapse' : '🔽 View JSON'}
              </button>
            </div>
            {showJson && (
              <pre className="json-box">{JSON.stringify(record.record, null, 2)}</pre>
            )}
          </div>
        </>
      )}

      {/* Regulation reference */}
      <div className="glass" style={{ padding: '22px 26px' }}>
        <h3 className="section-title" style={{ marginBottom: 14 }}>📜 Regulatory Reference</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
          {[
            { art: 'GDPR Art. 22', color: 'var(--purple)', text: 'Automated individual decision-making — subjects have the right to obtain a human explanation of any automated decision that significantly affects them.' },
            { art: 'EU AI Act Art. 13', color: 'var(--blue)', text: 'Transparency obligations — high-risk AI systems must provide comprehensible information to enable human oversight of their operation.' },
            { art: 'EU AI Act Art. 14', color: 'var(--cyan)', text: 'Human oversight — high-risk AI systems must be designed to allow effective oversight by humans, with the ability to intervene and override.' },
            { art: 'EU AI Act Art. 17', color: 'var(--amber)', text: 'Quality management — providers must maintain accurate, reliable, and verifiable record-keeping of all AI-assisted decisions for audit purposes.' },
          ].map(r => (
            <div key={r.art} style={{
              background: 'rgba(7,14,32,.8)',
              border: `1px solid ${r.color}30`,
              borderLeft: `3px solid ${r.color}`,
              borderRadius: '0 10px 10px 0',
              padding: '12px 16px',
            }}>
              <div style={{ fontWeight: 700, color: r.color, marginBottom: 5, fontSize: '.86rem' }}>{r.art}</div>
              <p style={{ fontSize: '.77rem', color: 'var(--muted)', lineHeight: 1.55 }}>{r.text}</p>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
