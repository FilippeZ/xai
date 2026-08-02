import React, { useState } from 'react';

export default function AuditTrailCard({ auditData, onGenerateAudit, illusionFlag }) {
  const [showJson, setShowJson] = useState(false);

  const record = auditData?.record;

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>📋</span> Level 4: Compliance Audit Trail Gate
        </h3>
        <button className="btn-neon" onClick={onGenerateAudit} style={{ fontSize: '0.8rem', padding: '8px 14px' }}>
          🔒 Generate Audit Record
        </button>
      </div>

      <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '20px' }}>
        Immutable decision records stamped with regulatory compliance tags (GDPR Art. 22 & EU AI Act Art. 13/14/17).
      </p>

      {record ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Status Header */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            background: 'rgba(15, 23, 42, 0.7)',
            padding: '14px 18px',
            borderRadius: '12px',
            border: '1px solid var(--border-color)'
          }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>RECORD ID</div>
              <div className="font-mono" style={{ fontWeight: 700, color: 'var(--primary-cyan)', fontSize: '0.95rem' }}>
                {record.record_id}
              </div>
            </div>

            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <span className={`badge ${record.decision_justified ? 'badge-green' : 'badge-red'}`} style={{ padding: '8px 16px', fontSize: '0.85rem' }}>
                {record.decision_justified ? '✅ AUDIT GATE: PASSED' : '⛔ AUDIT GATE: REJECTED'}
              </span>
            </div>
          </div>

          {/* Explanation Summary */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '14px', borderRadius: '10px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>EXPLANATION SUMMARY</div>
            <div>{record.explanation_summary}</div>
          </div>

          {/* Compliance Tags */}
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px' }}>STAMPED REGULATORY TAGS</div>
            <div style={{ display: 'flex', wrap: 'wrap', gap: '8px' }}>
              {record.compliance_tags.map((tag, idx) => (
                <span key={idx} className={`badge ${tag.includes('REJECTED') ? 'badge-red' : 'badge-purple'}`}>
                  🏷️ {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Toggle Raw JSON */}
          <div>
            <button
              className="btn-outline"
              onClick={() => setShowJson(!showJson)}
              style={{ fontSize: '0.78rem', padding: '6px 12px' }}
            >
              {showJson ? '🔼 Hide Raw JSON' : '🔽 View Raw JSON Audit Log'}
            </button>

            {showJson && (
              <pre className="font-mono" style={{
                marginTop: '10px',
                background: '#040711',
                padding: '14px',
                borderRadius: '10px',
                fontSize: '0.75rem',
                color: 'var(--primary-cyan)',
                maxHeight: '220px',
                overflow: 'auto',
                border: '1px solid var(--border-color)'
              }}>
                {JSON.stringify(record, null, 2)}
              </pre>
            )}
          </div>
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: '30px', color: 'var(--text-dim)', fontSize: '0.88rem' }}>
          Click "Generate Audit Record" above to stamp the immutable decision log.
        </div>
      )}
    </div>
  );
}
