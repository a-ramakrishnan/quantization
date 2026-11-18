import { useState } from 'react';
import { QuantApp } from '@/components/quant/QuantApp';
import { QuantizationLesson } from '@/components/quant/QuantizationLesson';
import { WeightComparison } from '@/components/quant/WeightComparison';
import { KvCachePage } from '@/components/quant/KvCachePage';

export function App() {
  const [studioOpen, setStudioOpen] = useState(false);
  const query = new URLSearchParams(window.location.search);
  if (query.get('view') === 'compare') {
    return <WeightComparison initialWeight={Number(query.get('weight'))} />;
  }
  if (query.get('view') === 'kv-cache') return <KvCachePage />;
  return studioOpen
    ? <QuantApp onOpenLesson={() => setStudioOpen(false)} />
    : <QuantizationLesson onOpenStudio={() => setStudioOpen(true)} />;
}

export default App;
