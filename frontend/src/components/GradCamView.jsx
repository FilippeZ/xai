import { useRef, useEffect } from 'react';

/**
 * GradCamView — renders real PyTorch Grad-CAM heatmap on a canvas.
 * `data.heatmap`        — flat 784-float array [0,1] from backend
 * `data.original_image` — flat 784-float array [0,1] pixel intensities
 */
export default function GradCamView({ data }) {
  const origRef = useRef(null);
  const camRef  = useRef(null);
  const blendRef = useRef(null);

  const SIZE = 28, SCALE = 8;

  // Draw original (grayscale)
  useEffect(() => {
    if (!data?.original_image || !origRef.current) return;
    const ctx = origRef.current.getContext('2d');
    for (let y = 0; y < SIZE; y++)
      for (let x = 0; x < SIZE; x++) {
        const v = Math.round((data.original_image[y * SIZE + x] || 0) * 255);
        ctx.fillStyle = `rgb(${v},${v},${v})`;
        ctx.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
      }
  }, [data]);

  // Draw pure heatmap (jet colormap)
  useEffect(() => {
    if (!data?.heatmap || !camRef.current) return;
    const ctx = camRef.current.getContext('2d');
    for (let y = 0; y < SIZE; y++)
      for (let x = 0; x < SIZE; x++) {
        const h = data.heatmap[y * SIZE + x] || 0;
        ctx.fillStyle = jetColor(h);
        ctx.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
      }
  }, [data]);

  // Draw blended overlay
  useEffect(() => {
    if (!data?.heatmap || !data?.original_image || !blendRef.current) return;
    const ctx = blendRef.current.getContext('2d');
    for (let y = 0; y < SIZE; y++)
      for (let x = 0; x < SIZE; x++) {
        const i = y * SIZE + x;
        const h = data.heatmap[i] || 0;
        const g = (data.original_image[i] || 0) * 255;
        const [r, g2, b] = jetRGB(h);
        ctx.fillStyle = `rgb(${Math.round(g * 0.4 + r * 0.6)},${Math.round(g * 0.4 + g2 * 0.6)},${Math.round(g * 0.4 + b * 0.6)})`;
        ctx.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
      }
  }, [data]);

  if (!data || data.method !== 'GRAD-CAM') return null;

  const regions = data.heatmap_regions || [];
  const classProbs = data.class_probabilities || {};

  return (
    <div className="glass" style={{ padding: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <h3 className="section-title">🔥 Grad-CAM Class Activation Heatmap</h3>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <span className="badge badge-purple">Predicted: <strong style={{ marginLeft: 4 }}>Digit {data.predicted_label}</strong></span>
          <span className="badge badge-cyan">True: Digit {data.true_label}</span>
          <span className="badge badge-green">Confidence: {(data.confidence * 100).toFixed(1)}%</span>
          {data.elapsed_ms && <span className="badge badge-amber">⚡ {data.elapsed_ms} ms</span>}
        </div>
      </div>
      <p style={{ fontSize: '.78rem', color: 'var(--muted)', marginBottom: 20 }}>
        Real gradient-weighted class activation map from the last Conv2d layer —
        <span style={{ color: 'var(--purple)' }}> EU AI Act Art. 13 Visual Inspection</span>.
      </p>

      {/* Three canvases */}
      <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start', flexWrap: 'wrap', marginBottom: 24 }}>
        {[
          { ref: origRef, label: 'Original Digit', border: 'var(--border)' },
          { ref: camRef,  label: 'Activation Heatmap', border: 'rgba(239,68,68,.6)' },
          { ref: blendRef,label: 'Blended Overlay', border: 'var(--cyan)' },
        ].map(({ ref, label, border }) => (
          <div key={label} style={{ textAlign: 'center' }}>
            <p className="label" style={{ marginBottom: 6 }}>{label}</p>
            <canvas ref={ref} width={SIZE * SCALE} height={SIZE * SCALE}
              style={{ borderRadius: 12, border: `2px solid ${border}`,
                       boxShadow: `0 0 18px ${border}40`, display: 'block' }} />
          </div>
        ))}

        {/* Colorbar legend */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <p className="label" style={{ marginBottom: 8 }}>Activation</p>
          <div style={{
            width: 22, height: SIZE * SCALE, borderRadius: 8,
            background: 'linear-gradient(to top, #00008b, #0000ff, #00ffff, #ffff00, #ff0000)',
            border: '1px solid var(--border)',
          }} />
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: SIZE * SCALE, marginTop: -SIZE * SCALE, paddingLeft: 28 }}>
            <span style={{ fontSize: '.65rem', color: 'var(--red)' }}>High</span>
            <span style={{ fontSize: '.65rem', color: 'var(--muted)' }}>Mid</span>
            <span style={{ fontSize: '.65rem', color: 'var(--blue)' }}>Low</span>
          </div>
        </div>
      </div>

      {/* Top Activation Regions */}
      <div style={{ marginBottom: 20 }}>
        <p className="label" style={{ marginBottom: 10 }}>Top Activation Regions</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {regions.map((reg, idx) => (
            <div key={idx} style={{
              background: 'rgba(7,14,32,.7)', borderRadius: 10,
              border: '1px solid var(--border)', padding: '10px 14px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.84rem', marginBottom: 5 }}>
                <span style={{ fontWeight: 600 }}>{reg.region}</span>
                <span className="mono" style={{ color: 'var(--cyan)', fontWeight: 700 }}>
                  {(reg.importance * 100).toFixed(1)}%
                </span>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{
                  width: `${reg.importance * 100}%`,
                  background: 'linear-gradient(90deg,#ef4444,#f59e0b)',
                }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Class probability distribution */}
      <div>
        <p className="label" style={{ marginBottom: 10 }}>Class Probability Distribution</p>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {Object.entries(classProbs)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 6)
            .map(([cls, prob]) => (
              <div key={cls} style={{
                background: 'rgba(255,255,255,.04)', border: '1px solid var(--border)',
                borderRadius: 8, padding: '6px 12px', textAlign: 'center',
                minWidth: 58,
              }}>
                <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--cyan)' }}>{cls}</div>
                <div style={{ fontSize: '.72rem', color: 'var(--muted)' }}>{(prob * 100).toFixed(1)}%</div>
              </div>
          ))}
        </div>
      </div>

      {/* Compliance tags */}
      <div style={{ marginTop: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {(data.compliance_tags || []).map(tag => (
          <span key={tag} className="badge badge-purple" style={{ fontSize: '.66rem' }}>🏷️ {tag}</span>
        ))}
      </div>
    </div>
  );
}

/* ── Jet colormap helpers ── */
function jetRGB(t) {
  // Jet: blue → cyan → green → yellow → red
  const clamp = v => Math.max(0, Math.min(255, Math.round(v)));
  const r = clamp(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3))));
  const g = clamp(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2))));
  const b = clamp(255 * Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1))));
  return [r, g, b];
}
function jetColor(t) {
  const [r, g, b] = jetRGB(t);
  return `rgb(${r},${g},${b})`;
}
