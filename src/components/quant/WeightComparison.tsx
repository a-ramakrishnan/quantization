import { useState } from "react";
import { ArrowLeft, ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from "lucide-react";
import { floatStep, getFormat } from "@/utils/quantization";
import { getGpuWeightTrace, reconstructionFormula, traceNumber } from "./gpu-weight-trace";
import {
  ACTIVATION_CODES, ACTIVATION_SCALE, FP16_INPUT, FP16_WEIGHTS, INFERENCE_MODES, INFERENCE_OUTPUTS,
  INPUT, INT8_SUMS, PROJECTIONS, WEIGHT_FORMATS,
} from "./lesson-example";
import type { LessonMode } from "./lesson-example";
import { useLessonPlayback } from "./useLessonPlayback";
import "./weight-comparison.css";

const MODES = [...WEIGHT_FORMATS, ...INFERENCE_MODES];
const PRESETS = [["W8A16", "W8A8"], ["INT4", "INT8"], ["FP8", "FP16"]] as const;
const STEPS = ["Original", "Encode", "Reconstruct", "Prepare inputs", "Dot product"];
const STEP_SECONDS = 3;
const source = PROJECTIONS.INT8.source;
const columns = source[0]!.length;

function laneData(mode: LessonMode, row: number, column: number) {
  const format = mode === "W8A8" || mode === "W8A16" ? "INT8" : mode;
  const projection = PROJECTIONS[format];
  const trace = getGpuWeightTrace(format, projection.result, [row, column]);
  const spec = getFormat(format);
  const original = source[row]![column]!;
  const originalRow = source[row]!.map(traceNumber);
  const encodedRow = spec.kind === "int" ? projection.result.q[row]!.map(traceNumber) : trace.payload.items;
  const reconstructedRow = projection.result.deq[row]!.map(traceNumber);
  const usesCodes = mode === "W8A8";
  const usesFP16 = mode === "W8A16";
  const weights = usesCodes ? projection.result.q[row]! : usesFP16 ? FP16_WEIGHTS[row]! : projection.result.deq[row]!;
  const inputs = usesCodes ? ACTIVATION_CODES : usesFP16 ? FP16_INPUT : INPUT;
  const inputBytes = usesCodes ? INPUT.length : usesFP16 ? INPUT.length * 2 : INPUT.length * 4;
  const inputScaleBytes = usesCodes ? 4 : 0;
  const output = usesCodes || usesFP16 ? INFERENCE_OUTPUTS[mode][row]! : projection.output[row]!;
  const baseline = projection.baseline[row]!;
  const step = trace.scale === undefined
    ? floatStep(Math.abs(original), spec.exponentBits!, spec.mantissaBits!)
    : undefined;
  const encoding = trace.scale === undefined
    ? `Round to nearest, ties to even, at spacing ${traceNumber(step!)} → ${traceNumber(trace.reconstructed)} → ${trace.stored}.`
    : `w / S ≈ ${traceNumber(original / trace.scale)}. Round to code ${trace.code}, stored as ${trace.stored}.`;
  const activation = usesCodes
    ? `Round x / Sₓ to INT8 codes using Sₓ ≈ ${traceNumber(ACTIVATION_SCALE)}. The selected weight enters the multiply as code ${traceNumber(weights[column]!)}.`
    : usesFP16
      ? `Cast the inputs and the reconstructed weights to FP16. The selected weight enters the multiply as ${traceNumber(weights[column]!)}.`
      : "Use the reconstructed weights and the original FP32 inputs.";
  const product = usesCodes
    ? `INT32 sum ${INT8_SUMS[row]} × S × Sₓ ≈ ${traceNumber(output)}.`
    : `Sum the four products to get ≈ ${traceNumber(output)}.`;

  return {
    format, trace, output, baseline, weightBytes: projection.weightBytes, scaleBytes: projection.scaleBytes,
    inputBytes, inputScaleBytes,
    frames: [
      { value: traceNumber(original), label: "Original BF16 weight", rowLabel: "Original weights", row: originalRow, detail: "Both lanes start with the same four BF16 weights." },
      { value: trace.stored, label: format === "INT4" ? "Packed byte" : "Stored bits", rowLabel: spec.kind === "int" ? "Integer codes" : "Stored bit patterns", row: encodedRow, detail: encoding },
      { value: traceNumber(trace.reconstructed), label: usesCodes ? "Equivalent decoded weight" : "Reconstructed weight", rowLabel: usesCodes ? "Equivalent decoded weights" : "Reconstructed weights", row: reconstructedRow,
        detail: usesCodes ? `${reconstructionFormula(trace)}. W8A8 can defer scaling until after the integer sum.` : reconstructionFormula(trace) },
      { value: traceNumber(inputs[column]!), label: usesCodes ? "INT8 activation code" : usesFP16 ? "FP16 activation" : "FP32 activation", rowLabel: usesCodes ? "Activation codes" : "Activation values", row: inputs.map(traceNumber), detail: activation },
      { value: traceNumber(output), label: "Row output", rowLabel: usesCodes ? "Integer products" : "Products", row: weights.map((w, i) => `${traceNumber(w)} × ${traceNumber(inputs[i]!)}`), detail: product },
    ],
  };
}

function ComparisonLane({ label, mode, setMode, lane, column, frame, position }: {
  label: string;
  mode: LessonMode;
  setMode: (mode: LessonMode) => void;
  lane: ReturnType<typeof laneData>;
  column: number;
  frame: number;
  position: number;
}) {
  const state = lane.frames[frame]!;
  return (
    <section className="comparison-lane" aria-label={label}>
      <label className="comparison-lane-heading">
        <span>{label}</span>
        <select aria-label={`${label} configuration`} value={mode} onChange={event => setMode(event.target.value as LessonMode)}>
          {MODES.map(option => <option key={option}>{option}</option>)}
        </select>
      </label>

      <div className="comparison-track" aria-hidden="true">
        <div className="comparison-track-line" />
        {STEPS.map((step, i) => <i key={step} data-current={i === frame} style={{ left: `${12 + i * 19}%` }} />)}
        <span className="comparison-token" style={{ left: `${12 + position * 19}%` }} data-frame={frame}>{state.value}</span>
      </div>

      <div className="comparison-lane-state">
        <p className="comparison-current"><span>{state.label}</span><strong>{state.value}</strong></p>
        <p className="comparison-detail">{state.detail}</p>
        <div className="comparison-row">
          <p>{state.rowLabel}</p>
          <div className="comparison-cells">
            {state.row.map((value, i) => <span key={i} data-selected={column === i}>{value}</span>)}
          </div>
        </div>
      </div>

      <dl className="comparison-lane-stats">
        <div><dt>Weight storage · 2×4 matrix</dt><dd>{lane.weightBytes + lane.scaleBytes} B{lane.scaleBytes > 0 && <small> · {lane.weightBytes} B weights + {lane.scaleBytes} B scale</small>}</dd></div>
        <div><dt>Input storage · 4 activations</dt><dd>{lane.inputBytes + lane.inputScaleBytes} B{lane.inputScaleBytes > 0 && <small> · {lane.inputBytes} B codes + {lane.inputScaleBytes} B scale</small>}</dd></div>
        {frame === STEPS.length - 1 && <div><dt>Output error · BF16 baseline − output</dt><dd className="comparison-error">{traceNumber(lane.baseline - lane.output)}</dd></div>}
      </dl>
    </section>
  );
}

export function WeightComparison({ initialWeight = 0 }: { initialWeight?: number }) {
  const [left, setLeft] = useState<LessonMode>("W8A16");
  const [right, setRight] = useState<LessonMode>("W8A8");
  const [selected, setSelected] = useState(
    Number.isInteger(initialWeight) && initialWeight >= 0 && initialWeight < source.length * columns ? initialWeight : 0,
  );
  const row = Math.floor(selected / columns);
  const column = selected % columns;
  const playback = useLessonPlayback(STEPS.length, true, 1, STEP_SECONDS);
  const { chapter, chapterTime, time, duration, playing, reducedMotion } = playback;
  const before = Math.max(0, chapter - 1);
  const phase = reducedMotion ? 1 : Math.min(1, chapterTime / 0.7);
  const frame = phase === 1 ? chapter : before;
  const position = before + (chapter - before) * (phase * phase * (3 - 2 * phase));
  const leftLane = laneData(left, row, column);
  const rightLane = laneData(right, row, column);
  const sharedWeights = leftLane.format === rightLane.format;
  const active = STEPS[frame]!;

  function seekStep(step: number) {
    playback.seek(step === 0 ? 0 : step * STEP_SECONDS + 0.75);
  }

  return (
    <main className="quant-lesson weight-comparison">
      <header className="lesson-header">
        <div className="lesson-shell lesson-header-inner">
          <a className="lesson-brand" href="./"><ArrowLeft size={17} /> Back to lesson</a>
          <a className="lesson-lab" href="?view=kv-cache">KV-cache calculator</a>
        </div>
      </header>
      <div className="lesson-shell">
        <div className="comparison-intro">
          <h1>Same weights, two paths</h1>
          <p>Pick a weight. Watch each configuration store it, reconstruct it, and use the row to compute an output.</p>
        </div>

        <div className="comparison-setup">
          <div>
            <h2>Compare configurations</h2>
            <div className="experiment-buttons" role="group" aria-label="Comparison presets">
              {PRESETS.map(([a, b]) => (
                <button type="button" key={a} aria-pressed={left === a && right === b} onClick={() => { setLeft(a); setRight(b); }}>
                  {a} vs {b}
                </button>
              ))}
            </div>
          </div>
          <div>
            <h2>Weight to follow <span>W[{row + 1},{column + 1}]</span></h2>
            <div className="comparison-weight-picker" role="group" aria-label="Source weights">
              {source.flat().map((value, i) => (
                <button type="button" key={i} aria-pressed={selected === i}
                  aria-label={`Row ${Math.floor(i / columns) + 1}, column ${i % columns + 1}, weight ${traceNumber(value)}`}
                  onClick={() => setSelected(i)}>{Math.abs(value) < 1e-4 ? value.toExponential(1) : traceNumber(value)}</button>
              ))}
            </div>
          </div>
        </div>

        <section className="lesson-player comparison-player" aria-label="Synchronized comparison animation">
          <nav className="lesson-chapters comparison-chapters" aria-label="Comparison steps">
            {STEPS.map((step, i) => (
              <button type="button" key={step} aria-current={frame === i ? "step" : undefined}
                onClick={() => playback.selectChapter(i)}>{step}</button>
            ))}
          </nav>
          <p className="comparison-shared" aria-live={playing ? "off" : "polite"}>
            {sharedWeights
              ? <>Both lanes store the same {leftLane.format} weights. The selected payload is <strong>{leftLane.trace.stored}</strong> in each lane.</>
              : <>Both lanes quantize the same BF16 row. Their stored payloads are <strong>{leftLane.trace.stored}</strong> and <strong>{rightLane.trace.stored}</strong>.</>}
          </p>
          <div className="comparison-lanes">
            <ComparisonLane label="Left lane" mode={left} setMode={setLeft} lane={leftLane} column={column} frame={frame} position={position} />
            <ComparisonLane label="Right lane" mode={right} setMode={setRight} lane={rightLane} column={column} frame={frame} position={position} />
          </div>

          <div className="lesson-controls">
            <button className="lesson-icon" type="button" aria-label="Restart comparison" onClick={() => seekStep(0)}><RotateCcw size={17} /></button>
            <button className="lesson-icon" type="button" aria-label="Previous comparison step" disabled={frame === 0} onClick={() => seekStep(frame - 1)}><ChevronLeft size={20} /></button>
            <button className="lesson-play" type="button" aria-label={playing ? "Pause comparison" : "Play comparison"}
              disabled={reducedMotion} onClick={playback.togglePlayback}>{playing ? <Pause size={17} /> : <Play size={17} />}</button>
            <button className="lesson-icon" type="button" aria-label="Next comparison step" disabled={frame === STEPS.length - 1}
              onClick={() => seekStep(frame + 1)}><ChevronRight size={20} /></button>
            <input type="range" min={0} max={duration} step={playback.scrubStep} value={time} aria-label="Comparison progress"
              aria-valuetext={active} onChange={event => playback.seek(Number(event.target.value))} />
            <span className="lesson-time">{Math.floor(time)} / {duration}s</span>
          </div>
        </section>
        <p className="comparison-note">Row {row + 1} · Original inputs [{INPUT.map(traceNumber).join(", ")}] · BF16-weight baseline ≈ {traceNumber(leftLane.baseline)}. Byte counts use this teaching example and include any shared scales.</p>
      </div>
    </main>
  );
}
