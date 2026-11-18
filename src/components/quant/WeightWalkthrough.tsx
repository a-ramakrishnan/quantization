import { ArrowRight, Pause, Play } from "lucide-react";
import type { Matrix, QuantConfig, QuantResult } from "@/types/quantization";
import { floatStep, getFormat, matVec } from "@/utils/quantization";
import { getGpuWeightTrace, reconstructionFormula, traceNumber } from "./gpu-weight-trace";
import { useLessonPlayback } from "./useLessonPlayback";
import { TraceReadout } from "./TraceReadout";

const STAGES = ["Calibrate", "Quantize", "GPU transfer", "Dot product"];
const STEP_SECONDS = 2.5;
const center = (stage: number) => 12.5 + stage * 25;

function weightSteps(W: Matrix, cfg: QuantConfig, result: QuantResult, x: number[], selected: [number, number]) {
  const [row, col] = selected;
  const w = W[row]![col]!;
  const format = getFormat(cfg.format);
  const trace = getGpuWeightTrace(cfg.format, result, selected);
  const { scale, zero, code, stored, reconstructed } = trace;
  const label = `W[${row},${col}]`;
  const activation = x[col]!;
  const product = reconstructed * activation;
  const sum = matVec(result.deq, x)[row]!;
  const calibration = scale === undefined
    ? `The target ${format.label} format sets the exponent and fraction widths. The source weight is still ${traceNumber(w)}.`
    : `Calibration chose S = ${traceNumber(scale)} and Z = ${zero} for this block. The source weight stays ${traceNumber(w)}.`;
  const conversion = scale === undefined
    ? `Round to nearest-even on the ${format.label} grid. Its spacing near this magnitude is ${traceNumber(floatStep(Math.min(Math.abs(w), format.maxValue!), format.exponentBits!, format.mantissaBits!))}. Values beyond the format’s range saturate at its limit.`
    : `Round w / S ≈ ${traceNumber(w / scale)}, add Z = ${zero}, and clamp to [${result.params!.qmin}, ${result.params!.qmax}] to obtain code ${code}.`;
  const storage = cfg.format === "INT4"
    ? `The code occupies the ${col % 2 === 0 ? "high" : "low"} nibble of ${stored}. The other nibble belongs to the adjacent weight.`
    : `Encode the ${format.label} value in ${format.bits} bits.`;

  return [
    { stage: 0, operation: "Read original", name: label, value: traceNumber(w), why: "This is the original value in the selected matrix cell." },
    { stage: 0, operation: "Calibrate", name: label, value: traceNumber(w), why: calibration },
    { stage: 1, operation: scale === undefined ? "Round the weight" : "Compute the code", name: scale === undefined ? "Rounded weight" : "Integer code q", value: traceNumber(code), why: conversion },
    { stage: 1, operation: "Encode for storage", name: "Stored bits", value: stored, why: storage },
    { stage: 2, operation: "Copy to HBM", name: "Stored bits in HBM", value: stored, why: "The GPU copies the encoded bits into its main memory. The bit pattern stays the same." },
    { stage: 2, operation: "Fetch through L2 and shared memory", name: "Stored bits at the SM", value: stored, why: "The kernel fetches the same bits through cache and shared memory." },
    ...(scale === undefined ? [] : [
      { stage: 2, operation: "Read the integer code", name: "Decoded code q", value: traceNumber(code), why: "The registers interpret the bits as the integer code for the selected weight." },
      { stage: 2, operation: "Subtract the zero-point", name: "Centered code q − Z", value: traceNumber(code - zero), why: `Subtract Z = ${zero} so the code for zero maps to a numerical zero.` },
    ]),
    { stage: 2, operation: "Reconstruct in registers", name: "Reconstructed weight ŵ", value: traceNumber(reconstructed), why: `${reconstructionFormula(trace)}. The original was ${traceNumber(w)}.` },
    { stage: 3, operation: "Multiply by activation", name: "Selected product", value: traceNumber(product), why: `Multiply ŵ = ${traceNumber(reconstructed)} by x[${col}] = ${traceNumber(activation)}.` },
    { stage: 3, operation: "Add the row products", name: `Output ŷ[${row}]`, value: traceNumber(sum), why: `Add the reconstructed products from all ${W[row]!.length} columns in this row.` },
  ];
}

export function WeightWalkthrough({ W, cfg, result, x, selected, onSelect, enabled }: {
  W: Matrix;
  cfg: QuantConfig;
  result: QuantResult;
  x: number[];
  selected: [number, number];
  onSelect: (cell: [number, number]) => void;
  enabled: boolean;
}) {
  const steps = weightSteps(W, cfg, result, x, selected);
  const playback = useLessonPlayback(steps.length, enabled, 1, STEP_SECONDS);
  const { chapter, chapterTime, duration, playing, reducedMotion, time } = playback;
  const previous = Math.max(0, chapter - 1);
  const phase = reducedMotion ? 1 : Math.min(1, chapterTime / 0.55);
  const activeIndex = phase >= 1 ? chapter : previous;
  const current = steps[activeIndex]!;
  const next = steps[activeIndex + 1];
  const from = center(steps[previous]!.stage);
  const to = center(steps[chapter]!.stage);
  const position = from + (to - from) * phase;
  const columns = W[0]!.length;

  function seekOperation(index: number) {
    playback.seek(index * STEP_SECONDS + (index === 0 ? 0 : 0.6));
  }

  return (
    <section className="math-theater operation-trace mb-10 rounded-xl border border-theater-grid p-4 sm:p-6" aria-label="Trace one weight">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-serif text-3xl">Trace one weight</h3>
        <label className="flex items-center gap-2 text-sm text-theater-muted">
          Weight
          <select className="max-w-full rounded border border-theater-grid bg-theater p-2 font-mono text-xs text-theater-ink"
            value={selected[0] * columns + selected[1]} onChange={event => {
              const index = Number(event.target.value);
              onSelect([Math.floor(index / columns), index % columns]);
            }}>
            {W.flatMap((row, r) => row.map((value, c) => (
              <option key={`${r},${c}`} value={r * columns + c}>W[{r},{c}] · {traceNumber(value)}</option>
            )))}
          </select>
        </label>
      </div>

      <div className="operation-rail" aria-hidden="true">
        <span className="operation-token" data-stage={current.stage} style={{ left: `${position}%` }}>{current.value}</span>
      </div>
      <div className="mb-4 grid grid-cols-4 gap-1 sm:gap-2" aria-label="Trace stages">
        {STAGES.map((stage, i) => (
          <button type="button" key={stage} aria-current={current.stage === i ? "step" : undefined}
            className="operation-stage rounded border border-theater-grid p-2 text-xs text-theater-muted sm:text-sm"
            onClick={() => seekOperation(steps.findIndex(step => step.stage === i))}>
            {stage}
          </button>
        ))}
      </div>

      <TraceReadout value={current.value} name={current.name} why={current.why} playing={playing}
        next={next?.operation ?? "The reconstructed row output is ready."} />

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button type="button" aria-label={playing ? "Pause trace" : "Play trace"} onClick={playback.togglePlayback}
          disabled={reducedMotion} className="operation-control operation-play rounded p-2">
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <button type="button" className="operation-control" disabled={activeIndex === 0}
          onClick={() => seekOperation(activeIndex - 1)}>Previous operation</button>
        <button type="button" className="operation-control" disabled={!next}
          onClick={() => seekOperation(activeIndex + 1)}>Next operation <ArrowRight size={15} /></button>
        <input type="range" className="min-w-24 flex-1 accent-theater-gold" min={0} max={duration}
          step={playback.scrubStep} value={time} aria-label="Weight trace progress"
          aria-valuetext={current.operation} onChange={event => playback.seek(Number(event.target.value))} />
      </div>
    </section>
  );
}
