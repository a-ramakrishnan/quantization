import { QuantApp } from '@/components/quant/QuantApp';
import { QuantizationLesson } from '@/components/quant/QuantizationLesson';
import { WeightComparison } from '@/components/quant/WeightComparison';
import { KvCachePage } from '@/components/quant/KvCachePage';

export function App() {
  const query = new URLSearchParams(window.location.search);
  const view = query.get('view');
  if (view === 'compare') {
    return <WeightComparison initialWeight={Number(query.get('weight'))} />;
  }
  if (view === 'kv-cache') return <KvCachePage />;
  if (view === 'basic' || view === 'advanced') return <QuantApp initialMode={view} />;
  return <QuantizationLesson />;
}

export default App;
