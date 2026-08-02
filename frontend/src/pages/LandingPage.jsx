/**
 * LandingPage — fullscreen hero with real XAIGO logo and demo video.
 */
export default function LandingPage({ onEnter }) {
  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--bg)',
      display: 'flex',
      flexDirection: 'column',
      position: 'relative',
      overflow: 'hidden',
    }}>

      {/* Ambient blobs */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0 }}>
        <div style={{
          position: 'absolute', width: 700, height: 700, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(168,85,247,.18) 0%, transparent 70%)',
          top: '-15%', left: '-10%', filter: 'blur(60px)',
        }} />
        <div style={{
          position: 'absolute', width: 600, height: 600, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(0,210,220,.14) 0%, transparent 70%)',
          bottom: '-10%', right: '-5%', filter: 'blur(50px)',
        }} />
        <div style={{
          position: 'absolute', width: 400, height: 400, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(79,172,254,.10) 0%, transparent 70%)',
          top: '40%', left: '55%', filter: 'blur(40px)',
        }} />
      </div>

      {/* ── Top bar ── */}
      <header style={{
        position: 'relative', zIndex: 10,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '14px 48px',
        borderBottom: '1px solid var(--border)',
        background: 'rgba(7,12,24,0.82)', backdropFilter: 'blur(14px)',
      }}>
        {/* Logo left */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 46, height: 46, borderRadius: 12, overflow: 'hidden',
            border: '1.5px solid var(--border-glow)',
            boxShadow: '0 0 14px rgba(0,210,220,.35)',
            flexShrink: 0, background: '#fff',
          }}>
            <img src="/logo.jpeg" alt="XAIGO Logo"
              style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          </div>
          <div>
            <div style={{ fontWeight: 900, fontSize: '1.05rem', letterSpacing: '-.01em' }}>
              <span style={{ color: 'var(--cyan)' }}>XAIGO</span>
            </div>
            <div style={{ fontSize: '.67rem', color: 'var(--dim)' }}>
              XAI Governance Middleware
            </div>
          </div>
        </div>

        {/* Right badges */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <span className="badge badge-purple" style={{ fontSize: '.67rem' }}>📜 GDPR Art. 22</span>
          <span className="badge badge-cyan"   style={{ fontSize: '.67rem' }}>🛡️ EU AI Act</span>
          <span className="badge badge-green"  style={{ fontSize: '.67rem' }}>⚡ API Live</span>
        </div>
      </header>

      {/* ── Hero ── */}
      <section style={{
        position: 'relative', zIndex: 5,
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        textAlign: 'center', padding: '56px 48px 32px',
      }}>
        {/* Big logo */}
        <div style={{
          width: 130, height: 130, borderRadius: 28, overflow: 'hidden',
          border: '2px solid var(--border-glow)',
          boxShadow: '0 0 50px rgba(0,210,220,.28), 0 0 100px rgba(168,85,247,.18)',
          marginBottom: 28, background: '#fff',
          transition: 'transform .3s, box-shadow .3s',
        }}
          onMouseEnter={e => {
            e.currentTarget.style.transform = 'scale(1.06)';
            e.currentTarget.style.boxShadow = '0 0 70px rgba(0,210,220,.45), 0 0 130px rgba(168,85,247,.28)';
          }}
          onMouseLeave={e => {
            e.currentTarget.style.transform = 'scale(1)';
            e.currentTarget.style.boxShadow = '0 0 50px rgba(0,210,220,.28), 0 0 100px rgba(168,85,247,.18)';
          }}
        >
          <img src="/logo.jpeg" alt="XAIGO"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        </div>



        {/* Headline */}
        <h1 style={{
          fontSize: 'clamp(2rem, 5vw, 3.5rem)', fontWeight: 900,
          lineHeight: 1.12, maxWidth: 840, marginBottom: 18, letterSpacing: '-0.02em',
        }}>
          Governance Middleware for{' '}
          <span style={{
            background: 'linear-gradient(135deg,var(--cyan),var(--blue),var(--purple))',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          }}>
            AI Decisions
          </span>
        </h1>

        <p style={{
          fontSize: '1.02rem', color: 'var(--muted)', maxWidth: 620,
          lineHeight: 1.65, marginBottom: 32,
        }}>
          A 4-layer XAI architecture that <strong style={{ color: 'var(--text)' }}>intercepts</strong>,{' '}
          <strong style={{ color: 'var(--text)' }}>explains</strong>, and{' '}
          <strong style={{ color: 'var(--text)' }}>audits</strong> black-box AI decisions
          before they reach end users — fully compliant with GDPR and the EU AI Act.
        </p>

        {/* Layer pills */}
        <div style={{
          display: 'flex', gap: 10, flexWrap: 'wrap',
          justifyContent: 'center', marginBottom: 36,
        }}>
          {[
            { icon: '📊', label: 'Layer 1 — Input Data',     color: 'var(--blue)' },
            { icon: '💜', label: 'Layer 2 — XAI Controls',   color: 'var(--purple)' },
            { icon: '🧪', label: 'Layer 3 — Simulatability', color: 'var(--cyan)' },
            { icon: '📋', label: 'Layer 4 — Audit Trail',    color: 'var(--amber)' },
          ].map(l => (
            <div key={l.label} style={{
              display: 'flex', alignItems: 'center', gap: 7,
              background: 'rgba(255,255,255,.05)', border: `1px solid ${l.color}30`,
              borderRadius: 99, padding: '7px 16px',
              fontSize: '.78rem', fontWeight: 600, color: 'var(--muted)',
            }}>
              <span>{l.icon}</span>
              <span style={{ color: l.color }}>{l.label}</span>
            </div>
          ))}
        </div>

        {/* CTA */}
        <button onClick={onEnter} style={{
          display: 'inline-flex', alignItems: 'center', gap: 10,
          background: 'linear-gradient(135deg,var(--cyan),var(--blue))',
          color: '#020811', fontWeight: 900, fontSize: '1rem',
          border: 'none', borderRadius: 14, padding: '15px 38px',
          cursor: 'pointer',
          boxShadow: '0 0 32px rgba(0,210,220,.5)',
          transition: 'transform .2s, box-shadow .2s',
          marginBottom: 14,
        }}
          onMouseEnter={e => {
            e.currentTarget.style.transform = 'translateY(-3px)';
            e.currentTarget.style.boxShadow = '0 0 52px rgba(0,210,220,.8)';
          }}
          onMouseLeave={e => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 0 32px rgba(0,210,220,.5)';
          }}
        >
          ⚡ Launch XAIGO Application →
        </button>

        <div style={{ fontSize: '.72rem', color: 'var(--dim)' }}>↓ Scroll to watch the demo</div>
      </section>

      {/* ── VIDEO ── */}
      <section style={{
        position: 'relative', zIndex: 5,
        padding: '8px 48px 56px',
        display: 'flex', flexDirection: 'column', alignItems: 'center',
      }}>
        <div style={{
          width: '100%', maxWidth: 1040,
          borderRadius: 22, overflow: 'hidden',
          border: '1px solid var(--border-glow)',
          boxShadow: '0 0 60px rgba(0,210,220,.18), 0 28px 80px rgba(0,0,0,.7)',
          background: '#000',
          position: 'relative',
        }}>
          <div style={{
            height: 3,
            background: 'linear-gradient(90deg,var(--purple),var(--cyan),var(--blue))',
          }} />

          <video src="/demo.mp4" autoPlay muted loop playsInline controls
            style={{ width: '100%', display: 'block', maxHeight: '62vh', objectFit: 'cover' }} />

          {/* LIVE badge over video */}
          <div style={{
            position: 'absolute', top: 16, left: 16,
            background: 'rgba(7,12,24,.88)', backdropFilter: 'blur(8px)',
            border: '1px solid var(--border)',
            borderRadius: 8, padding: '5px 12px',
            fontSize: '.7rem', color: 'var(--red)', fontWeight: 700,
            display: 'flex', alignItems: 'center', gap: 6, pointerEvents: 'none',
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', background: 'var(--red)',
              display: 'inline-block', animation: 'pulse 1.4s infinite',
            }} />
            LIVE DEMO
          </div>

          {/* Logo watermark inside video frame */}
          <div style={{
            position: 'absolute', top: 14, right: 14,
            width: 34, height: 34, borderRadius: 8, overflow: 'hidden',
            border: '1.5px solid rgba(255,255,255,.2)',
            background: '#fff', pointerEvents: 'none',
            opacity: 0.85,
          }}>
            <img src="/logo.jpeg" alt="XAIGO" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          </div>
        </div>
        <p style={{ marginTop: 14, fontSize: '.78rem', color: 'var(--dim)', textAlign: 'center' }}>
          Full system demonstration — SHAP · LIME · Grad-CAM · Counterfactuals · Simulatability Engine · Audit Trail
        </p>
      </section>

      {/* ── Feature grid ── */}
      <section style={{
        position: 'relative', zIndex: 5,
        padding: '0 48px 72px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: 16,
        maxWidth: 1060, margin: '0 auto', width: '100%',
      }}>
        {[
          { icon: '💜', title: 'SHAP Explanations', desc: 'Shapley Additive values via TreeExplainer. Satisfies GDPR Art. 22 Right to Explanation.' },
          { icon: '🍊', title: 'LIME Surrogate', desc: 'Local Interpretable Model-agnostic Explanations for tabular and text data.' },
          { icon: '🔥', title: 'Grad-CAM Heatmaps', desc: 'Gradient-weighted Class Activation Maps from the CNN last Conv layer. EU AI Act Art. 13.' },
          { icon: '🔄', title: 'Counterfactuals', desc: 'Minimal plausible perturbations with immutable feature constraints and L₀/L₁/L₂ metrics.' },
          { icon: '🧪', title: 'Simulatability Engine', desc: 'Measures genuine user accuracy improvement. Detects Illusion of Understanding.' },
          { icon: '📋', title: 'Compliance Audit Gate', desc: 'SHA-256 stamped records with GDPR, EU AI Act Art. 9/13/14/17 tags. Auto-rejects on illusion.' },
        ].map(f => (
          <div key={f.title} className="glass" style={{ padding: '20px 22px', cursor: 'pointer' }}
            onClick={onEnter}
            onMouseEnter={e => e.currentTarget.querySelector('h3').style.color = 'var(--cyan)'}
            onMouseLeave={e => e.currentTarget.querySelector('h3').style.color = 'var(--text)'}
          >
            <div style={{ fontSize: '1.6rem', marginBottom: 10 }}>{f.icon}</div>
            <h3 style={{ fontWeight: 700, marginBottom: 6, fontSize: '.92rem', transition: 'color .2s' }}>{f.title}</h3>
            <p style={{ fontSize: '.76rem', color: 'var(--muted)', lineHeight: 1.55 }}>{f.desc}</p>
          </div>
        ))}
      </section>

      {/* ── Footer ── */}
      <footer style={{
        position: 'relative', zIndex: 5,
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
        padding: '16px 48px',
        borderTop: '1px solid var(--border)',
        fontSize: '.72rem', color: 'var(--dim)',
        background: 'rgba(7,12,24,.7)', backdropFilter: 'blur(8px)',
        flexWrap: 'wrap', textAlign: 'center',
      }}>
        <img src="/logo.jpeg" alt="XAIGO"
          style={{ width: 22, height: 22, borderRadius: 5, objectFit: 'cover', verticalAlign: 'middle' }} />
        <span>XAIGO · XAI Governance Middleware · © 2026</span>
      </footer>

      <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }`}</style>
    </div>
  );
}
