import { useState } from 'react';
import { apiPost } from '../api';

/**
 * AuditPanel — Level 4: Compliance Audit Trail Gate.
 * Reads real data from explanationData and simulatabilityResult.
 * Sends real POST to /api/audit.
 */
export default function AuditPanel({ explanationData, simulatabilityResult }) {
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

      // Build feature_attributions from the right field per method/modality
      let attrs = {};
      if (method === 'SHAP' || method === 'LIME') {
        attrs = explanationData?.feature_attributions ?? {};
      } else if (method === 'COUNTERFACTUAL') {
        // Derive attributions from changes (Δ values as proxies)
        const changes = explanationData?.changes ?? {};
        const origVals = explanationData?.original_values ?? {};
        const cfVals = explanationData?.counterfactual_values ?? {};
        for (const feat of Object.keys(origVals)) {
          if (feat in changes) {
            attrs[feat] = parseFloat(((cfVals[feat] ?? 0) - (origVals[feat] ?? 0)).toFixed(6));
          }
        }
        if (Object.keys(attrs).length === 0) {
          attrs = { 'change_placeholder': 0.0 };
        }
      } else if (method === 'GRAD-CAM') {
        // Use top heatmap regions as attribution proxy
        const regions = explanationData?.heatmap_regions ?? [];
        regions.forEach(r => { attrs[r.region] = r.importance; });
      } else if (method === 'LIME-TEXT') {
        attrs = explanationData?.word_attributions ?? {};
      }

      // predicted_label: handle all method types
      const predicted_label =
        explanationData?.predicted_label ??
        explanationData?.original_pred ??
        'unknown';

      const data = await apiPost('/api/audit', {
        model_type: modality === 'image'
          ? 'CNN (MNIST Benchmark)'
          : modality === 'text'
            ? 'TF-IDF + LogisticRegression (20 Newsgroups)'
            : 'RandomForest (Iris Benchmark)',
        model_accuracy: explanationData?.modality === 'image'
          ? 0.91 : explanationData?.modality === 'text' ? 0.95 : 0.9333,
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
    <div className="glass" style={{ padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
        <h3 className="section-title">📋 Level 4 — Compliance Audit Trail Gate</h3>
        <button className="btn-ghost" onClick={generate} disabled={loading}
          style={{ fontSize: '.8rem', whiteSpace: 'nowrap' }}>
          {loading ? '⏳ Generating…' : '🔒 Generate Audit Record'}
        </button>
      </div>

      <p style={{ fontSize: '.78rem', color: 'var(--muted)', marginBottom: 18 }}>
        Immutable decision records stamped with regulatory compliance tags (GDPR Art. 22 &amp; EU AI Act
        Art. 13/14/17). Audit Gate <strong style={{ color: 'var(--red)' }}>REJECTS</strong> decisions
        where Illusion of Understanding is active.
      </p>

      {error && (
        <div style={{ background: 'rgba(239,68,68,.1)', border: '1px solid rgba(239,68,68,.3)', borderRadius: 10, padding: 12, fontSize: '.8rem', color: 'var(--red)', marginBottom: 14 }}>
          ⚠️ {error}
        </div>
      )}

      {!record ? (
        <div style={{ textAlign: 'center', padding: '28px 0', color: 'var(--dim)', fontSize: '.85rem' }}>
          Click <strong style={{ color: 'var(--cyan)' }}>🔒 Generate Audit Record</strong> above to stamp the immutable decision log.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Gate Status banner */}
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            background: justified ? 'rgba(34,197,94,.08)' : 'rgba(239,68,68,.08)',
            border: `1px solid ${justified ? 'rgba(34,197,94,.25)' : 'rgba(239,68,68,.25)'}`,
            borderRadius: 12, padding: '14px 18px',
          }}>
            <div>
              <div className="label" style={{ marginBottom: 4 }}>Record ID</div>
              <div className="mono" style={{ color: 'var(--cyan)', fontWeight: 700, fontSize: '.95rem' }}>
                {record.record?.record_id}
              </div>
            </div>
            <span className={`badge ${justified ? 'badge-green' : 'badge-red'}`}
              style={{ padding: '8px 16px', fontSize: '.85rem' }}>
              {justified ? '✅ AUDIT GATE: PASSED' : '⛔ AUDIT GATE: REJECTED'}
            </span>
          </div>

          {/* Summary */}
          <div style={{ background: 'rgba(255,255,255,.03)', borderRadius: 10, border: '1px solid var(--border)', padding: '12px 14px' }}>
            <div className="label" style={{ marginBottom: 6 }}>Explanation Summary</div>
            <p style={{ fontSize: '.82rem', lineHeight: 1.55, color: 'var(--text)' }}>
              {record.record?.explanation_summary}
            </p>
          </div>

          {/* Meta row */}
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {[
              { label: 'XAI Method', val: record.record?.xai_method },
              { label: 'Predicted', val: record.record?.predicted_label },
              { label: 'True Label', val: record.record?.true_label },
              { label: 'Confidence', val: record.record?.confidence != null ? `${(record.record.confidence * 100).toFixed(1)}%` : 'N/A' },
            ].map(({ label, val }) => (
              <div key={label} style={{
                background: 'rgba(255,255,255,.04)', border: '1px solid var(--border)',
                borderRadius: 8, padding: '7px 12px', fontSize: '.8rem', minWidth: 90,
              }}>
                <div style={{ color: 'var(--muted)', fontSize: '.7rem', marginBottom: 2 }}>{label}</div>
                <div className="mono" style={{ fontWeight: 700, color: 'var(--cyan)' }}>{val ?? '—'}</div>
              </div>
            ))}
          </div>

          {/* Compliance tags */}
          <div>
            <div className="label" style={{ marginBottom: 8 }}>Stamped Regulatory Tags</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
              {(record.record?.compliance_tags || []).map((tag, i) => (
                <span key={i}
                  className={`badge ${tag.includes('REJECT') || tag.includes('illusion') ? 'badge-red' : 'badge-purple'}`}>
                  🏷️ {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Timestamp + reviewer */}
          <div style={{ fontSize: '.74rem', color: 'var(--dim)', display: 'flex', gap: 20 }}>
            <span>🕐 {new Date(record.record?.timestamp).toLocaleString()}</span>
            <span>👤 {record.record?.human_reviewer}</span>
          </div>

          {/* JSON toggle */}
          <div>
            <button className="btn-ghost" onClick={() => setShowJson(v => !v)}
              style={{ fontSize: '.76rem', marginBottom: 8 }}>
              {showJson ? '🔼 Hide JSON' : '🔽 View Raw Audit JSON'}
            </button>
            {showJson && (
              <pre className="json-box">{JSON.stringify(record.record, null, 2)}</pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
