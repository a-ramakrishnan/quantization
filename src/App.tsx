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
      <StudioPage />
      <footer className="studio-footer">
        <div className="studio-footer-inner">
          <section aria-labelledby="academic-use">
            <h2 id="academic-use">Academic use disclaimer</h2>
            <p>Quantization Studio is provided for academic and educational purposes. We strive to keep the information up to date. Calculations and simulations are illustrative and depend on the inputs and assumptions shown. Model specifications and software support can change, and published benchmarks describe the tested workloads and systems. Verify the source specifications and measure your own workload before sizing a deployment.</p>
          </section>
          <a href="https://github.com/a-ramakrishnan/quantization">View project on GitHub ↗</a>
        </div>
      </footer>
    </>
  );
}

export default App;
