import { useState } from "react";
import { matVec, quantize } from "@/utils/quantization";
import { traceNumber } from "./gpu-weight-trace";
import { INPUT, PROJECTIONS, WEIGHTS } from "./lesson-example";

const MIN_WEIGHT = 0.46875;
const MAX_WEIGHT = 0.65625;
const BOUNDARY = 0.5625;
const SCALE = 0.125;
const EXAMPLE_WEIGHTS = [
  ["Exact", 0.5],
  ["Same code", 0.53125],
  ["At the boundary", BOUNDARY],
  ["Above the boundary", 0.59375],
] as const;
const plotX = (weight: number) => (weight - MIN_WEIGHT) / (MAX_WEIGHT - MIN_WEIGHT) * 100;

export function RoundingBoundary() {
  const [weight, setWeight] = useState(0.53125);
  const result = quantize([[7 * SCALE, weight]], { format: "INT4", symmetric: true, granularity: "tensor", alpha: 1 });
  const code = result.q[0]![1]!;
  const reconstructed = result.deq[0]![1]!;

  return (
    <section id="rounding-boundary" className="lesson-reference lesson-experiment" aria-labelledby="rounding-boundary-title">
      <h3 id="rounding-boundary-title">Cross a rounding boundary</h3>
      <p>Move the blue weight while keeping the INT4 scale fixed at {SCALE}. The gold code changes at the midpoint, {BOUNDARY}.</p>
      <div className="experiment-boundary" aria-hidden="true">
        <div className="boundary-track" />
        <div className="boundary-mark" style={{ left: `${plotX(BOUNDARY)}%` }}><span>{BOUNDARY} midpoint</span></div>
        <div className="boundary-dot" data-kind="original" style={{ left: `${plotX(weight)}%` }} />
        <div className="boundary-dot" data-kind="reconstructed" style={{ left: `${plotX(reconstructed)}%` }} />
        <span className="boundary-label" style={{ left: `${plotX(0.5)}%` }}>q = 4 · 0.5</span>
        <span className="boundary-label" style={{ left: `${plotX(0.625)}%` }}>q = 5 · 0.625</span>
      </div>
      <label className="experiment-range">
        <span>Original weight <strong>{traceNumber(weight)}</strong></span>
        <input type="range" min={MIN_WEIGHT} max={MAX_WEIGHT} step={2 ** -10} value={weight}
          aria-label="Original weight on the INT4 number line" onChange={event => setWeight(Number(event.target.value))} />
      </label>
      <div className="experiment-buttons" aria-label="Rounding examples">
        {EXAMPLE_WEIGHTS.map(([label, value]) => (
          <button type="button" key={value} aria-pressed={weight === value} onClick={() => setWeight(value)}>{label}</button>
        ))}
      </div>
      <p className="experiment-equation" aria-live="polite">round({traceNumber(weight)} / {SCALE}) = <b>{code}</b> → reconstructed <strong>{traceNumber(reconstructed)}</strong> · Δ {traceNumber(weight - reconstructed)}</p>
      <p>Both 0.5 and 0.53125 reconstruct to 0.5. At the halfway value, this integer example rounds toward positive infinity.</p>
    </section>
  );
}

const PRUNE_ORDER = WEIGHTS.flat().map((value, index) => ({ value, index }))
  .sort((a, b) => Math.abs(a.value) - Math.abs(b.value))
  .map(({ index }) => index);

export function CompressionExperiment() {
  const [pruned, setPruned] = useState(4);
  const projection = PROJECTIONS.INT4;
  const mask = projection.source.flat().map((_, i) => !PRUNE_ORDER.slice(0, pruned).includes(i));
  const kept = mask.length - pruned;
  const codeBytes = Math.ceil(kept / 2);
  const maskBytes = Math.ceil(mask.length / 8);
  const sparseBytes = codeBytes + maskBytes + projection.scaleBytes;
  const sparseWeights = projection.result.deq.map((row, r) => row.map((value, c) => mask[r * row.length + c] ? value : 0));
  const sparseOutput = matVec(sparseWeights, INPUT)[0]!;

  return (
    <section id="compression" className="lesson-reference lesson-experiment" aria-labelledby="compression-title">
      <h3 id="compression-title">Quantization, sparsity, and compression</h3>
      <p>Quantization uses fewer bits per weight. Pruning sets selected weights to zero. A sparse encoding can omit those zeros and store a mask so the decoder knows where they belong.</p>
      <label className="experiment-range">
        <span>Prune lowest-magnitude weights <strong>{pruned} of {mask.length}</strong></span>
        <input type="range" min={0} max={6} step={2} value={pruned} aria-label="Number of weights pruned"
          onChange={event => setPruned(Number(event.target.value))} />
      </label>
      <div className="compression-codes" aria-label="Kept and pruned weight codes in row order">
        {projection.result.q.flat().map((code, index) => (
          <div key={index} data-kept={mask[index]}><span>{mask[index] ? code : "–"}</span><small>{mask[index] ? "Keep" : "Prune"}</small></div>
        ))}
      </div>
      <dl className="compression-sizes">
        <div><dt>Dense BF16</dt><dd>{projection.baselineBytes} B</dd></div>
        <div><dt>Dense INT4</dt><dd>{projection.weightBytes + projection.scaleBytes} B</dd></div>
        <div><dt>Sparse INT4</dt><dd>{sparseBytes} B</dd></div>
      </dl>
      <p className="experiment-equation">{codeBytes} B packed codes + {maskBytes} B mask + {projection.scaleBytes} B scale = <strong>{sparseBytes} B</strong> · mask {mask.map(Number).join("")}</p>
      <p>The sparse row 1 output is {traceNumber(sparseOutput)}. The original BF16-weight output is {traceNumber(projection.baseline[0]!)}. Sparse storage pays for position metadata, and skipping pruned weights during inference requires a compatible kernel.</p>
      <p>Entropy coding can further compress a serialized model. It encodes the remaining symbols losslessly. <a href="https://hanlab.mit.edu/projects/deep-compression">Han et al., Deep Compression</a></p>
    </section>
  );
}
