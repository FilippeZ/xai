import { useState, useEffect } from 'react';
import { apiGet, apiPost } from './api';
import Navbar from './components/Navbar';
import LandingPage from './pages/LandingPage';
import XAIPage from './pages/XAIPage';
import SimulatabilityPage from './pages/SimulatabilityPage';
import AuditPage from './pages/AuditPage';

export default function App() {
  // ── Navigation ────────────────────────────────────────────────────────────
  const [view, setView] = useState('landing');     // 'landing' | 'app'
  const [currentPage, setCurrentPage] = useState('xai');

  // ── Backend state ─────────────────────────────────────────────────────────
  const [healthStatus, setHealthStatus] = useState(null);
  const [modality, setModality] = useState('tabular');
  const [method, setMethod] = useState('SHAP');
  const [instanceIndex, setInstanceIndex] = useState(0);
  const [immutableFeatures, setImmutableFeatures] = useState(['sepal length (cm)']);
  const [explanationData, setExplanationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [simulResult, setSimulResult] = useState(null);

  // ── Health check ──────────────────────────────────────────────────────────
  useEffect(() => {
    apiGet('/api/health')
      .then(d => setHealthStatus(d.status))
      .catch(() => setHealthStatus('error'));
  }, []);

  const handleSetModality = (m) => {
    setModality(m);
    setExplanationData(null);
    setError(null);
  };

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    setExplanationData(null);
    try {
      const data = await apiPost('/api/explain', {
        modality, method,
        instance_index: instanceIndex,
        immutable_features: immutableFeatures,
        num_features: 6,
      });
      setExplanationData(data);
    } catch (e) {
      setError(`Backend error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ── Landing page ──────────────────────────────────────────────────────────
  if (view === 'landing') {
    return <LandingPage onEnter={() => setView('app')} />;
  }

  // ── App shell (with Navbar + pages) ──────────────────────────────────────
  const renderPage = () => {
    switch (currentPage) {
      case 'xai':
        return (
          <XAIPage
            modality={modality}       setModality={handleSetModality}
            method={method}           setMethod={setMethod}
            instanceIndex={instanceIndex} setInstanceIndex={setInstanceIndex}
            immutableFeatures={immutableFeatures} setImmutableFeatures={setImmutableFeatures}
            onRun={handleRun} loading={loading} error={error}
            explanationData={explanationData}
          />
        );
      case 'simulatability':
        return <SimulatabilityPage onResult={setSimulResult} />;
      case 'audit':
        return <AuditPage explanationData={explanationData} simulatabilityResult={simulResult} />;
      default:
        return null;
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        status={healthStatus}
        currentPage={currentPage}
        setPage={setCurrentPage}
        onHome={() => setView('landing')}
      />
      <main style={{
        flex: 1, padding: '28px 32px',
        maxWidth: 1440, margin: '0 auto', width: '100%',
      }}>
        {renderPage()}
      </main>
    </div>
  );
}
