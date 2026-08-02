const MODALITIES = [
  { id: 'tabular', icon: '📊', label: 'Tabular', sub: 'Iris Benchmark' },
  { id: 'image',   icon: '🖼️', label: 'Image',   sub: 'MNIST Digits (CNN)' },
  { id: 'text',    icon: '📝', label: 'Text',    sub: '20 Newsgroups' },
];

const METHODS = {
  tabular: [
    { id: 'SHAP',          icon: '💜', label: 'SHAP',          desc: 'Shapley Values · GDPR Art. 22' },
    { id: 'LIME',          icon: '🍊', label: 'LIME',          desc: 'Local Surrogate Model' },
    { id: 'COUNTERFACTUAL',icon: '🔄', label: 'Counterfactual', desc: 'What-If Scenarios & Recourse' },
  ],
  image: [
    { id: 'GRAD-CAM',      icon: '🔥', label: 'Grad-CAM',      desc: 'CNN Activation Heatmap · Art. 13' },
  ],
  text: [
    { id: 'LIME',          icon: '🍊', label: 'LIME (Text)',   desc: 'Word-level Local Surrogate' },
  ],
};

const INSTANCES = {
  tabular: [
    'Patient Instance #0 (setosa)',
    'Patient Instance #1 (versicolor)',
    'Patient Instance #2 (virginica)',
    'Patient Instance #3 (setosa)',
    'Patient Instance #4 (versicolor)',
  ],
  image: ['Digit Sample #0', 'Digit Sample #1', 'Digit Sample #2',
           'Digit Sample #3', 'Digit Sample #4'],
  text: ['Article #0', 'Article #1', 'Article #2', 'Article #3', 'Article #4'],
};

const IMMUTABLE_OPTS = ['sepal length (cm)', 'sepal width (cm)',
                        'petal length (cm)', 'petal width (cm)'];

export default function ControlPanel({
  modality, setModality,
  method, setMethod,
  instanceIndex, setInstanceIndex,
  immutableFeatures, setImmutableFeatures,
  onRun, loading, error,
}) {
  const methods = METHODS[modality] || [];
  const instances = INSTANCES[modality] || [];

  const handleModalityChange = (m) => {
    setModality(m);
    const first = (METHODS[m] || [])[0];
    if (first) setMethod(first.id);
    setInstanceIndex(0);
  };

  const toggle = (f) =>
    setImmutableFeatures(prev =>
      prev.includes(f) ? prev.filter(x => x !== f) : [...prev, f]
    );

  return (
    <div className="glass" style={{ padding: 24 }}>
      <h2 className="section-title" style={{ marginBottom: 20 }}>
        🎛️ XAI Governance Execution Controls
      </h2>

      {/* Modality */}
      <div style={{ marginBottom: 20 }}>
        <p className="label" style={{ marginBottom: 8 }}>Select Data Modality</p>
        <div style={{ display: 'flex', gap: 8 }}>
          {MODALITIES.map(m => (
            <button
              key={m.id}
              className={`btn-ghost ${modality === m.id ? 'active' : ''}`}
              onClick={() => handleModalityChange(m.id)}
              style={{ flex: 1, flexDirection: 'column', padding: '10px 6px', gap: 3 }}
            >
              <span style={{ fontSize: '1.2rem' }}>{m.icon}</span>
              <span style={{ fontSize: '.75rem', fontWeight: 700 }}>{m.label}</span>
              <span style={{ fontSize: '.65rem', color: 'var(--dim)' }}>{m.sub}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Method */}
      <div style={{ marginBottom: 20 }}>
        <p className="label" style={{ marginBottom: 8 }}>Select XAI Controls Technique</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {methods.map(m => (
            <button
              key={m.id}
              className={`btn-ghost ${method === m.id ? 'active' : ''}`}
              onClick={() => setMethod(m.id)}
              style={{ justifyContent: 'flex-start', textAlign: 'left', padding: '11px 14px' }}
            >
              <span style={{ fontSize: '1rem' }}>{m.icon}</span>
              <div>
                <div style={{ fontWeight: 700, fontSize: '.88rem' }}>{m.label}</div>
                <div style={{ fontSize: '.71rem', color: 'var(--muted)', marginTop: 1 }}>{m.desc}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Instance */}
      <div style={{ marginBottom: 20 }}>
        <p className="label" style={{ marginBottom: 8 }}>Sample Instance</p>
        <select
          value={instanceIndex}
          onChange={e => setInstanceIndex(Number(e.target.value))}
          style={{
            width: '100%', padding: '10px 12px', borderRadius: 10,
            background: 'rgba(10,18,38,.9)', border: '1px solid var(--border)',
            color: 'var(--text)', fontFamily: 'var(--sans)', fontSize: '.86rem',
          }}
        >
          {instances.map((label, i) => (
            <option key={i} value={i}>{label}</option>
          ))}
        </select>
      </div>

      {/* Immutable (only for tabular counterfactual) */}
      {modality === 'tabular' && method === 'COUNTERFACTUAL' && (
        <div style={{ marginBottom: 20 }}>
          <p className="label" style={{ marginBottom: 8 }}>Immutable Features (Plausibility Gate)</p>
          {IMMUTABLE_OPTS.map(f => (
            <label key={f} style={{
              display: 'flex', alignItems: 'center', gap: 9,
              fontSize: '.82rem', marginBottom: 8, cursor: 'pointer',
            }}>
              <input type="checkbox" checked={immutableFeatures.includes(f)}
                onChange={() => toggle(f)}
                style={{ accentColor: 'var(--cyan)', width: 15, height: 15 }} />
              <span>🔒 {f}</span>
            </label>
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          background: 'rgba(239,68,68,.12)', border: '1px solid rgba(239,68,68,.3)',
          borderRadius: 10, padding: '10px 14px', marginBottom: 14,
          fontSize: '.82rem', color: 'var(--red)',
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* Run */}
      <button className="btn-primary" onClick={onRun} disabled={loading}>
        {loading ? '⏳ Computing…' : `⚡ RUN ${method} EXPLANATION`}
      </button>

      {modality === 'image' && !loading && (
        <p style={{ fontSize: '.7rem', color: 'var(--dim)', marginTop: 8, textAlign: 'center' }}>
          ⚠️ First image request trains the CNN (~30 s). Subsequent calls are instant.
        </p>
      )}
    </div>
  );
}
