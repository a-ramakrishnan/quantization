import { QuantApp } from '@/components/quant/QuantApp';
import { QuantizationLesson } from '@/components/quant/QuantizationLesson';
import { WeightComparison } from '@/components/quant/WeightComparison';
import { KvCachePage } from '@/components/quant/KvCachePage';

function StudioPage() {
  const query = new URLSearchParams(window.location.search);
  const view = query.get('view');
  if (view === 'compare') {
    return <WeightComparison initialWeight={Number(query.get('weight'))} />;
  }
  if (view === 'kv-cache') return <KvCachePage />;
  if (view === 'basic' || view === 'advanced') return <QuantApp initialMode={view} />;
  return <QuantizationLesson />;
}

export function App() {
  return (
    <>
      <aside className="academic-notice" aria-label="Academic use notice">
        <div>
          <strong>For academic purposes.</strong>{" "}
          We strive to keep the information up to date. Calculations are illustrative and use the assumptions shown.
        </div>
      </aside>
      <StudioPage />
    </>
  );
}

export default App;
