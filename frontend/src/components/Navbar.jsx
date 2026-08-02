import { API_BASE } from '../api';

const PAGES = [
  { id: 'xai',          icon: '⚡', label: 'XAI Execution',     sub: 'Layers 1 & 2' },
  { id: 'simulatability',icon: '🧪', label: 'Simulatability',    sub: 'Layer 3 — Human-in-the-Loop' },
  { id: 'audit',        icon: '📋', label: 'Audit Trail',        sub: 'Layer 4 — Compliance Gate' },
];

export default function Navbar({ status, currentPage, setPage, onHome }) {
  const connected = status === 'ok';
  return (
    <nav style={{
      position: 'sticky', top: 0, zIndex: 100,
      background: 'rgba(7,12,24,0.92)', backdropFilter: 'blur(16px)',
      borderBottom: '1px solid var(--border)',
    }}>
      {/* Top row */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 32px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10, overflow: 'hidden',
            border: '1.5px solid var(--border-glow)',
            boxShadow: '0 0 14px rgba(0,210,220,.35)',
            flexShrink: 0, background: '#fff',
          }}>
            <img src="/logo.jpeg" alt="XAIGO Logo"
              style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.1rem', fontWeight: 800, lineHeight: 1.2 }}>
              <span style={{ color: 'var(--cyan)' }}>XAIGO</span> <span style={{ color: 'var(--muted)', fontWeight: 500, fontSize: '.85rem' }}>· Governance Middleware</span>
            </h1>
            <p style={{ fontSize: '.7rem', color: 'var(--muted)' }}>
              AI Decision Interception &amp; Regulatory Compliance Engine
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <span className="badge badge-purple" style={{ fontSize: '.68rem' }}>📜 GDPR Art. 22</span>
          <span className="badge badge-purple" style={{ fontSize: '.68rem' }}>🛡️ EU AI Act Art. 13/14</span>
          <span className={`badge ${connected ? 'badge-green' : 'badge-red'}`} style={{ fontSize: '.68rem' }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%', display: 'inline-block',
              background: connected ? 'var(--green)' : 'var(--red)',
              animation: connected ? 'pulse 2s infinite' : 'none',
            }} />
            {connected
              ? `Backend Connected (${API_BASE.split('//')[1]})`
              : 'Backend Disconnected'}
          </span>
        </div>
      </div>

      {/* Page tabs */}
      <div style={{
        display: 'flex', gap: 0, alignItems: 'center',
        padding: '0 24px',
        borderTop: '1px solid var(--border)',
      }}>
        {/* Home button */}
        {onHome && (
          <button
            onClick={onHome}
            title="Back to Landing Page"
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '10px 14px',
              background: 'none', border: 'none', cursor: 'pointer',
              borderBottom: '2px solid transparent',
              color: 'var(--dim)',
              fontFamily: 'var(--sans)', fontSize: '.8rem',
              transition: 'color .2s',
              marginRight: 6,
            }}
            onMouseEnter={e => e.currentTarget.style.color = 'var(--cyan)'}
            onMouseLeave={e => e.currentTarget.style.color = 'var(--dim)'}
          >
            🏠 Home
          </button>
        )}
        <div style={{ width: 1, height: 22, background: 'var(--border)', marginRight: 6 }} />
        {PAGES.map((p, i) => {
          const active = currentPage === p.id;
          return (
            <button
              key={p.id}
              onClick={() => setPage(p.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '10px 22px',
                background: 'none', border: 'none', cursor: 'pointer',
                borderBottom: active ? '2px solid var(--cyan)' : '2px solid transparent',
                color: active ? 'var(--cyan)' : 'var(--muted)',
                fontFamily: 'var(--sans)', fontSize: '.84rem', fontWeight: active ? 700 : 500,
                transition: 'all .2s',
                position: 'relative',
                marginBottom: -1,
              }}
            >
              <span style={{ fontSize: '.95rem' }}>{p.icon}</span>
              <div style={{ textAlign: 'left' }}>
                <div>{p.label}</div>
                <div style={{ fontSize: '.65rem', color: active ? 'var(--cyan)' : 'var(--dim)', marginTop: 1, fontWeight: 400 }}>
                  {p.sub}
                </div>
              </div>
              {active && (
                <div style={{
                  position: 'absolute', bottom: -1, left: 0, right: 0,
                  height: 2, background: 'var(--cyan)',
                  boxShadow: '0 0 10px var(--cyan)',
                }} />
              )}
            </button>
          );
        })}
      </div>

      <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }`}</style>
    </nav>
  );
}
