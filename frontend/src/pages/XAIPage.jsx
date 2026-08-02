import ControlPanel from '../components/ControlPanel';
import ResultsPane from '../components/ResultsPane';

/**
 * Page 1 — XAI Execution (Layers 1 & 2)
 * Holds the ControlPanel (left) and the ResultsPane (right).
 */
export default function XAIPage({
  modality, setModality,
  method, setMethod,
  instanceIndex, setInstanceIndex,
  immutableFeatures, setImmutableFeatures,
  onRun, loading, error,
  explanationData,
}) {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '380px 1fr',
      gap: 24,
      alignItems: 'start',
    }}>
      <ControlPanel
        modality={modality}       setModality={setModality}
        method={method}           setMethod={setMethod}
        instanceIndex={instanceIndex} setInstanceIndex={setInstanceIndex}
        immutableFeatures={immutableFeatures} setImmutableFeatures={setImmutableFeatures}
        onRun={onRun} loading={loading} error={error}
      />
      <ResultsPane data={explanationData} method={method} />
    </div>
  );
}
