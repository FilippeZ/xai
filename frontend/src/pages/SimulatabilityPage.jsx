import { useState } from 'react';
import { apiPost } from '../api';

/**
 * Page 2 — Simulatability Engine (Layer 3)
 * Full-page dedicated view with sliders, live score, and regulation note.
 */
export default function SimulatabilityPage({ onResult }) {
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
  const score = result?.simulatability_pct;

  return (
    <div style={{ maxWidth: 860, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Header card */}
      <div className="glass" style={{ padding: '28px 32px' }}>
        <div style={{ display: 'flex', align: 'flex-start', gap: 16, marginBottom: 20 }}>
          <div style={{
            width: 52, height: 52, borderRadius: 14, flexShrink: 0,
            background: 'linear-gradient(135deg,var(--blue),var(--cyan))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1.5rem', boxShadow: '0 0 20px rgba(79,172,254,.4)',
          }}>🧪</div>
          <div>
            <h2 style={{ fontSize: '1.3rem', fontWeight: 800, marginBottom: 4 }}>
              Level 3 — Simulatability Engine
            </h2>
            <p style={{ fontSize: '.82rem', color: 'var(--muted)', lineHeight: 1.5 }}>
              Measures whether XAI explanations genuinely improve human operator prediction accuracy.
              A score ≤ 0 signals an <strong style={{ color: 'var(--red)' }}>Illusion of Understanding</strong> — the
              explanation looked convincing but didn't transfer real causal knowledge.
            </p>
          </div>
        </div>

        {/* Formula display */}
        <div style={{
          background: 'rgba(7,14,32,.8)', border: '1px solid var(--border)',
          borderRadius: 12, padding: '14px 20px',
          display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap',
        }}>
          <div style={{ fontSize: '.8rem', color: 'var(--muted)' }}>Formula:</div>
          <div className="mono" style={{ color: 'var(--cyan)', fontSize: '1.05rem', fontWeight: 700 }}>
            Simulatability = Acc<sub>post</sub> − Acc<sub>pre</sub>
          </div>
          <div style={{ height: 1, background: 'var(--border)', flex: 1, minWidth: 30 }} />
          <div style={{ display: 'flex', gap: 8 }}>
            <span className="badge badge-green" style={{ fontSize: '.68rem' }}>Score &gt; 0 → PASS</span>
            <span className="badge badge-red" style={{ fontSize: '.68rem' }}>Score ≤ 0 → REJECT (Illusion)</span>
          </div>
        </div>
      </div>

      {/* Sliders + controls */}
      <div className="glass" style={{ padding: '28px 32px' }}>
        <h3 className="section-title" style={{ marginBottom: 24 }}>
          🎚️ Human-in-the-Loop Accuracy Inputs
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32, marginBottom: 28 }}>
          {/* PRE */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 10 }}>
              <div>
                <div className="label">Human Accuracy PRE-Explanation</div>
                <div style={{ fontSize: '.72rem', color: 'var(--dim)', marginTop: 2 }}>
                  Before seeing the XAI output
                </div>
              </div>
              <div className="mono" style={{
                fontSize: '1.8rem', fontWeight: 800, color: 'var(--blue)',
                lineHeight: 1,
              }}>
                {(pre * 100).toFixed(0)}%
              </div>
            </div>
            <div style={{ position: 'relative', paddingBottom: 4 }}>
              <input type="range" min={0} max={1} step={0.05} value={pre}
                onChange={e => setPre(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--blue)', height: 6 }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.67rem', color: 'var(--dim)', marginTop: 4 }}>
                <span>0%</span><span>50%</span><span>100%</span>
              </div>
            </div>
          </div>

          {/* POST */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 10 }}>
              <div>
                <div className="label">Human Accuracy POST-Explanation</div>
                <div style={{ fontSize: '.72rem', color: 'var(--dim)', marginTop: 2 }}>
                  After reviewing the XAI output
                </div>
              </div>
              <div className="mono" style={{
                fontSize: '1.8rem', fontWeight: 800, color: 'var(--cyan)',
                lineHeight: 1,
              }}>
                {(post * 100).toFixed(0)}%
              </div>
            </div>
            <div style={{ position: 'relative', paddingBottom: 4 }}>
              <input type="range" min={0} max={1} step={0.05} value={post}
                onChange={e => setPost(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--cyan)', height: 6 }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.67rem', color: 'var(--dim)', marginTop: 4 }}>
                <span>0%</span><span>50%</span><span>100%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Live preview bar */}
        <div style={{
          background: 'rgba(7,14,32,.8)', border: '1px solid var(--border)',
          borderRadius: 12, padding: '16px 20px', marginBottom: 20,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: '.78rem', color: 'var(--muted)' }}>
            <span>PRE ({(pre * 100).toFixed(0)}%)</span>
            <span style={{ color: post > pre ? 'var(--green)' : 'var(--red)' }}>
              Expected Δ = {((post - pre) * 100).toFixed(0)}%
            </span>
            <span>POST ({(post * 100).toFixed(0)}%)</span>
          </div>
          <div style={{ display: 'flex', gap: 6, height: 12, borderRadius: 6, overflow: 'hidden' }}>
            <div style={{
              width: `${pre * 100}%`, background: 'var(--blue)',
              borderRadius: '6px 0 0 6px', transition: 'width .3s',
            }} />
            {post > pre && (
              <div style={{
                width: `${(post - pre) * 100}%`, background: 'var(--green)',
                transition: 'width .3s',
              }} />
            )}
            <div style={{ flex: 1, background: 'rgba(255,255,255,.05)' }} />
          </div>
        </div>

        {error && (
          <div style={{
            background: 'rgba(239,68,68,.1)', border: '1px solid rgba(239,68,68,.3)',
            borderRadius: 10, padding: 12, fontSize: '.82rem', color: 'var(--red)', marginBottom: 14,
          }}>⚠️ {error}</div>
        )}

        <button className="btn-primary" onClick={run} disabled={loading}>
          {loading ? '⏳ Computing…' : '🧮 Compute Simulatability & Check Illusion Flag'}
        </button>
      </div>

      {/* Result card */}
      {result && (
        <div className="glass" style={{ padding: '28px 32px' }}>
          <h3 className="section-title" style={{ marginBottom: 20 }}>📊 Simulatability Result</h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'PRE Accuracy', val: `${(result.accuracy_pre * 100).toFixed(0)}%`, color: 'var(--blue)' },
              { label: 'POST Accuracy', val: `${(result.accuracy_post * 100).toFixed(0)}%`, color: 'var(--cyan)' },
              { label: 'Simulatability Score', val: `${score >= 0 ? '+' : ''}${score?.toFixed(1)}%`, color: illusion ? 'var(--red)' : 'var(--green)' },
            ].map(m => (
              <div key={m.label} style={{
                background: 'rgba(7,14,32,.8)', border: '1px solid var(--border)',
                borderRadius: 12, padding: '18px 20px', textAlign: 'center',
              }}>
                <div className="label" style={{ marginBottom: 8 }}>{m.label}</div>
                <div className="mono" style={{ fontSize: '2.2rem', fontWeight: 800, color: m.color }}>
                  {m.val}
                </div>
              </div>
            ))}
          </div>

          {/* Gate status */}
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            background: illusion ? 'rgba(239,68,68,.10)' : 'rgba(34,197,94,.10)',
            border: `1px solid ${illusion ? 'rgba(239,68,68,.3)' : 'rgba(34,197,94,.3)'}`,
            borderRadius: 14, padding: '18px 24px', marginBottom: 14,
          }}>
            <div style={{ fontSize: '1rem', fontWeight: 700 }}>
              {illusion
                ? '⚠️ Illusion of Understanding Detected'
                : '✅ Genuine Causal Understanding Verified'}
            </div>
            <span className={`badge ${illusion ? 'badge-red' : 'badge-green'}`}
              style={{ padding: '9px 18px', fontSize: '.85rem' }}>
              Audit Gate: {result.audit_gate_status}
            </span>
          </div>

          <p style={{ fontSize: '.76rem', color: 'var(--muted)', lineHeight: 1.6 }}>
            {result.interpretation}
          </p>
          <p style={{ fontSize: '.74rem', color: 'var(--dim)', marginTop: 8 }}>
            📖 {result.regulation_note}
          </p>
        </div>
      )}

      {/* Theory card */}
      <div className="glass" style={{ padding: '24px 28px' }}>
        <h3 className="section-title" style={{ marginBottom: 14 }}>📚 Theoretical Background</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {[
            {
              title: 'Simulatability (Lipton, 2018)',
              text: 'A model is simulatable if a human can mentally simulate its output given the inputs and explanation. Measured by comparing accuracy before and after seeing the explanation.',
              color: 'var(--blue)',
            },
            {
              title: 'Illusion of Understanding',
              text: 'Occurs when human operators feel confident in the explanation but their actual prediction accuracy does not improve. This is a critical safety failure in AI deployment (EU AI Act Art. 14).',
              color: 'var(--red)',
            },
          ].map(c => (
            <div key={c.title} style={{
              background: 'rgba(7,14,32,.8)', border: `1px solid ${c.color}30`,
              borderRadius: 12, padding: '14px 16px',
            }}>
              <div style={{ fontWeight: 700, color: c.color, marginBottom: 6, fontSize: '.86rem' }}>{c.title}</div>
              <p style={{ fontSize: '.78rem', color: 'var(--muted)', lineHeight: 1.55 }}>{c.text}</p>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
