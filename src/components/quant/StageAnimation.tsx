import { useEffect, useId, useRef, useState } from "react";
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Matrix, QuantConfig, QuantResult } from "@/types/quantization";
import { getFormat, matVec, matrixRange, packNibbles, twosComplement } from "@/utils/quantization";
import { useLessonPlayback } from "./useLessonPlayback";
import { GpuWeightTrace } from "./GpuWeightTrace";
import { traceNumber } from "./gpu-weight-trace";

const LESSONS = [
  { title: "Find the scale of the weights", chapters: ["The matrix", "The range", "The scale"] },
  { title: "Turn a weight into a stored code", chapters: ["Divide & round", "Store the bits", "See the error"] },
  { title: "Follow one weight through the GPU", chapters: ["Upload to HBM", "Move to the SM", "Reconstruct"] },
  { title: "Watch the linear layer compute", chapters: ["Multiply", "Accumulate", "Compare"] },
];
const n = (v: number) => Number(v.toFixed(4)).toString();
const ease = (t: number) => t * t * (3 - 2 * t);

export function StageAnimation({ stage, W, cfg, result, x, selected, onSelect, active }: {
  stage: number; W: Matrix; cfg: QuantConfig; result: QuantResult; x: number[];
  selected: [number, number]; onSelect: (cell: [number, number]) => void; active: boolean;
}) {
  const lesson = LESSONS[stage];
  const ref = useRef<HTMLElement>(null);
  const grid = useId().replace(/:/g, "");
  const [onscreen, setOnscreen] = useState(false);
  const [speed, setSpeed] = useState(1);
  const playback = useLessonPlayback(3, active && onscreen, speed);
  const { chapter, time, duration, playing, reducedMotion: reduced } = playback;
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new IntersectionObserver(([entry]) => setOnscreen(entry?.isIntersecting ?? false), { threshold: 0.1 });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  if (!lesson) return null;
  const phase = ease(Math.min(1, playback.chapterTime / 4));
  const spec = getFormat(cfg.format);
  const [row, col] = selected;
  const w = W[row]?.[col] ?? 0;
  const q = result.q[row]?.[col] ?? 0;
  const deq = result.deq[row]?.[col] ?? 0;
  const block = result.blockOf(row, col);
  const scale = result.params?.scales[block] ?? 1;
  const zero = result.params?.zeros[block] ?? 0;
  const range = matrixRange(W);
  const max = Math.max(...W.flat().map(Math.abs), 0.001);
  const px = (v: number) => 480 + v / max * 330;
  const y = matVec(W, x);
  const yq = matVec(result.deq, x);
  const narrative = [
    ["The blue numbers are the sixteen original weights. The selected cell is the one we will follow through the next stages.", `The observed range is ${n(range.min)} to ${n(range.max)}. Outliers stretch the grid.`, spec.kind === "int" ? `The selected weight uses block ${block}, with S = ${n(scale)} and Z = ${zero}.` : `${spec.label} stores sign, exponent and mantissa instead of integer scales.`],
    [spec.kind === "int" ? `Divide ${n(w)} by S to get about ${n(w / scale)}. Round, add the zero-point ${zero}, and clamp the result to get code ${q}.` : `Round the selected FP32 weight to ${spec.label} precision.`, cfg.format === "INT4" ? "Two 4-bit codes share one byte. The first occupies the high nibble." : `Each stored weight occupies ${spec.bits} bits.`, `The reconstructed value is ${n(deq)}. The original minus reconstruction is ${n(w - deq)}.`],
    [
      `The CPU copies the stored representation of W[${row},${col}] to GPU memory. Watch the value on the gold card.`,
      "The same bits travel from HBM through L2 and shared memory into the registers. The stored value stays the same on this part of the journey.",
      spec.kind === "int"
        ? `Read code ${q}, subtract the zero-point ${zero}, then multiply by S to recover ${traceNumber(deq)}. The original was ${traceNumber(w)}. Their difference comes from quantizing the original weight.`
        : `The registers decode the stored ${spec.label} bits as ${traceNumber(deq)}. This is the already-rounded value used for the dot product.`,
    ],
    [`For the selected cell, multiply the original weight ${n(w)} by input ${n(x[col] ?? 0)} to get ${n(w * (x[col] ?? 0))}. Repeat for the four columns.`, "Add the four products. This is one output of the linear layer.", `For the selected row, the original output is ${n(y[row] ?? 0)} and the reconstructed output is ${n(yq[row] ?? 0)}. Pink shows the original minus reconstructed difference.`],
  ][stage]?.[chapter];
  const rowCodes = result.q[row] ?? [];
  const pairStart = col - col % 2;

  return <section ref={ref} className="math-theater mb-7" aria-label={`Stage ${stage + 1} animated lesson`}>
    <div className="px-4 pt-6 sm:px-6">
      <div className="flex justify-between gap-2 font-mono text-xs text-theater-muted"><span>STAGE 0{stage + 1} · {spec.label}</span><span className="text-theater-gold">0{chapter + 1} / 03</span></div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-serif text-2xl sm:text-3xl">{lesson.title}</h3>
        {stage === 2 && (
          <label className="flex flex-wrap items-center gap-2 text-sm text-theater-muted">
            Weight to trace
            <select className="rounded-md border border-theater-grid bg-theater px-2 py-1 font-mono text-theater-ink"
              value={row * W[row]!.length + col}
              onChange={event => {
                const index = Number(event.target.value);
                const columns = W[0]!.length;
                onSelect([Math.floor(index / columns), index % columns]);
              }}>
              {W.flatMap((values, r) => values.map((value, c) => (
                <option key={`${r},${c}`} value={r * values.length + c}>
                  W[{r},{c}] = {traceNumber(value)}
                </option>
              )))}
            </select>
          </label>
        )}
      </div>
      {stage === 2 ? (
        <GpuWeightTrace chapter={chapter} progress={Math.min(1, playback.chapterTime / 4)}
          format={cfg.format} result={result} selected={selected} original={w} grid={grid} playing={playing} />
      ) : (
      <div className="stage-math-viewport mt-4">
        <svg viewBox="0 0 960 400" className="h-full w-full" role="img" aria-label={`${lesson.chapters[chapter]} for Stage ${stage + 1}`}>
          <defs><pattern id={grid} width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" className="stroke-theater-grid" strokeWidth="0.5" /></pattern></defs>
          <rect width="960" height="400" fill={`url(#${grid})`} />
          {stage === 0 && <g>
            {chapter === 0 ? <g>
              <text x="480" y="48" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="30">W ∈ ℝ⁴ˣ⁴</text>
              <path d="M285 82H270V330H285 M675 82H690V330H675" fill="none" className="stroke-theater-blue" strokeWidth="2" />
              {W.flat().map((v, i) => <text key={i} x={330 + i % 4 * 100} y={126 + Math.floor(i / 4) * 58} textAnchor="middle" fontSize="22" opacity={0.65 + 0.35 * Math.min(1, phase * 2 + (15 - i) / 16)} className="fill-theater-blue font-mono" fontWeight={i === row * 4 + col ? 700 : 400}>{n(v)}</text>)}
            </g> : chapter === 1 ? <g>
              <text x="480" y="54" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="28">The range sets the spacing</text>
              <line x1="100" y1="240" x2="860" y2="240" className="stroke-theater-muted" />
              <line x1={px(range.min)} y1="250" x2={px(range.max)} y2="250" className="stroke-theater-gold" strokeWidth="3" />
              {W.flat().map((v, i) => <circle key={i} cx={480 + (px(v) - 480) * phase} cy={216 - i % 3 * 15} r="6" className="fill-theater-blue" />)}
              <text x={px(range.min)} y="294" textAnchor="middle" className="fill-theater-rose font-mono" fontSize="20">{n(range.min)}</text><text x={px(range.max)} y="294" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="20">{n(range.max)}</text>
              <text x="480" y="354" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="17">α = {cfg.alpha.toFixed(2)} · {cfg.granularity} granularity</text>
            </g> : <g>
              <text x="480" y="100" textAnchor="middle" className="fill-theater-blue font-serif" fontSize="30">w[{row},{col}] = {n(w)}</text>
              <text x="480" y="190" textAnchor="middle" className="fill-theater-gold font-mono" fontSize="34">{spec.kind === "int" ? `S = ${n(scale)}     Z = ${zero}` : `1 + ${spec.exponentBits} + ${spec.mantissaBits} bits`}</text>
              <text x="480" y="272" textAnchor="middle" className="fill-theater-green font-serif" fontSize="26">{spec.kind === "int" ? `q ∈ [${result.params?.qmin}, ${result.params?.qmax}]` : "Sign + exponent + mantissa"}</text>
              <text x="480" y="342" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="17">{spec.kind === "int" ? `${cfg.symmetric ? "Symmetric" : "Asymmetric"} · block ${block}` : `${spec.label} · ${spec.bits} bits per weight`}</text>
            </g>}
          </g>}
          {stage === 1 && <g>
            {chapter === 0 ? <g>
              <text x="480" y="60" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="28">{spec.kind === "int" ? "q = clamp(round(w / S) + Z)" : `FP32 → ${spec.label}`}</text>
              <text x="215" y="183" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="38">{n(w)}</text>
              <path d="M320 174H640 M630 166L640 174L630 182" fill="none" className="stroke-theater-muted" />
              <circle cx={340 + phase * 280} cy="174" r="6" className="fill-theater-gold" />
              <text x="752" y="183" textAnchor="middle" className="fill-theater-gold font-mono" fontSize="38">{n(q)}</text>
              <text x="480" y="284" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="19">{spec.kind === "int" ? `w / S = ${n(w / scale)} · Z = ${zero}` : `${spec.mantissaBits} mantissa bits retained`}</text>
            </g> : chapter === 1 ? <g>
              <text x="480" y="70" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="29">{cfg.format === "INT4" ? "Two weights. One byte." : `${spec.bits} bits per stored weight`}</text>
              {cfg.format === "INT4" ? <g>
                <text x="480" y="143" textAnchor="middle" className="fill-theater-gold font-mono" fontSize="32">{twosComplement(rowCodes[pairStart] ?? 0, 4)} {twosComplement(rowCodes[pairStart + 1] ?? 0, 4)}</text>
                <path d="M480 155V193" className="stroke-theater-muted" />
                <circle cx="480" cy={158 + phase * 31} r="5" className="fill-theater-gold" />
                <text x="480" y="235" textAnchor="middle" className="fill-theater-gold font-mono" fontSize="34" opacity={0.5 + phase * 0.5}>{packNibbles(rowCodes)[pairStart / 2]?.hex}</text>
              </g> : <g>{Array.from({length:spec.bits}, (_, i) => <rect key={i} x={150 + i * (660 / spec.bits)} y="145" width={600 / spec.bits} height="55" rx="2" className={i < Math.round(phase * spec.bits) ? "fill-theater-gold" : "fill-theater-grid"} />)}</g>}
              <text x="480" y="302" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="20">FP32: 64 B weights → {16 * spec.bits / 8} B weight payload</text>
              <text x="480" y="347" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="16">Integer scales and zero-points are additional metadata</text>
            </g> : <g>
              <text x="480" y="67" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="30">Rounding leaves a small gap</text>
              <line x1="100" y1="234" x2="860" y2="234" className="stroke-theater-muted" />
              <circle cx={px(w)} cy="215" r="9" className="fill-theater-blue" /><circle cx={px(w + (deq - w) * phase)} cy="250" r="9" className="fill-theater-green" />
              <text x="250" y="145" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="25">w = {n(w)}</text><text x="710" y="145" textAnchor="middle" className="fill-theater-green font-mono" fontSize="25">ŵ = {n(deq)}</text>
              <text x="480" y="332" textAnchor="middle" className="fill-theater-rose font-mono" fontSize="28">w − ŵ = {n(w - deq)}</text>
            </g>}
          </g>}
          {stage === 3 && <g>
            <text x="480" y="52" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="28">{chapter === 0 ? `Row ${row}: multiply each weight by x` : chapter === 1 ? `y[${row}] = Σⱼ w[${row},j] x[j]` : "y = Wx    vs    ŷ = Ŵx"}</text>
            {chapter < 2 ? <g>
              {(W[row] ?? []).map((v, i) => <g key={i}>
                <text x={170+i*205} y="119" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="23">{n(v)} × {n(x[i] ?? 0)}</text>
                <path d={`M ${170+i*205} 140 V ${174+phase*20}`} className="stroke-theater-gold" />
                <text x={170+i*205 + (480-(170+i*205))*(chapter === 1 ? phase : 0)} y={215+(chapter === 1 ? phase*58 : 0)} textAnchor="middle" opacity={chapter === 1 ? 1-phase : 1} className="fill-theater-gold font-mono" fontSize="25">{n(v*(x[i] ?? 0))}</text>
              </g>)}
              <text x="480" y="320" textAnchor="middle" className="fill-theater-green font-mono" fontSize="36" opacity={chapter === 1 ? phase : 0.5}>y[{row}] = {n(y[row] ?? 0)}</text>
            </g> : <g>
              <text x="240" y="105" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="20">FP32 baseline</text><text x="570" y="105" textAnchor="middle" className="fill-theater-green font-mono" fontSize="20">{spec.label} weights</text><text x="810" y="105" textAnchor="middle" className="fill-theater-rose font-mono" fontSize="20">Δ</text>
              {y.map((v, i) => <g key={i}>
                <text x="240" y={155+i*54} textAnchor="middle" className="fill-theater-blue font-mono" fontSize="25">{v.toFixed(4)}</text>
                <text x="570" y={155+i*54} textAnchor="middle" className="fill-theater-green font-mono" fontSize="25" opacity={0.55 + phase * 0.45}>{(yq[i] ?? 0).toFixed(4)}</text>
                <text x="810" y={155+i*54} textAnchor="middle" className="fill-theater-rose font-mono" fontSize="21">{(v-(yq[i] ?? 0)).toFixed(5)}</text>
              </g>)}
            </g>}
          </g>}
        </svg>
      </div>
      )}
      <p className="mx-auto min-h-16 max-w-3xl text-center font-serif text-lg leading-relaxed text-theater-ink" aria-live={playing ? "off" : "polite"}>{narrative}</p>
      <div className="mt-3 grid grid-cols-3 gap-2 border-b border-theater-grid pb-3">
        {lesson.chapters.map((label, i) => <Button key={label} variant="ghost" aria-pressed={chapter === i} onClick={() => playback.selectChapter(i)} className={cn("h-auto min-h-10 whitespace-normal px-1 text-xs sm:text-sm", chapter === i ? "text-theater-gold" : "text-theater-muted")}>{label}</Button>)}
      </div>
      <div className="flex flex-wrap items-center gap-2 py-4 sm:gap-3">
        <Button variant="ghost" size="icon" title="Restart animation" aria-label="Restart animation" onClick={() => playback.seek(0)}><RotateCcw /></Button>
        <Button variant="ghost" size="icon" title="Previous chapter" aria-label="Previous chapter" disabled={chapter === 0} onClick={() => playback.selectChapter(chapter - 1)}><SkipBack /></Button>
        <Button variant="secondary" size="icon" title={playing ? "Pause animation" : "Play animation"} aria-label={playing ? "Pause animation" : "Play animation"} disabled={reduced} onClick={playback.togglePlayback}>{playing ? <Pause /> : <Play />}</Button>
        <Button variant="ghost" size="icon" title="Next chapter" aria-label="Next chapter" disabled={chapter === 2} onClick={() => playback.selectChapter(chapter + 1)}><SkipForward /></Button>
        <input type="range" min="0" max={duration} step={playback.scrubStep} value={time} aria-label="Animation progress" className="min-w-20 flex-1 accent-theater-gold" onChange={e => playback.seek(Number(e.target.value))} />
        <span className="w-20 text-right font-mono text-xs text-theater-muted">{time.toFixed(1)} / {duration}s</span>
        <select aria-label="Playback speed" value={speed} onChange={e => setSpeed(Number(e.target.value))} className="rounded-md border border-theater-grid bg-theater px-2 py-1 text-xs text-theater-ink"><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select>
        {reduced && <span className="w-full text-xs text-theater-muted">Reduced motion: select a chapter to view each step.</span>}
      </div>
    </div>
  </section>;
}