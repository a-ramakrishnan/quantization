import { useState } from "react";
import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { BasicFormat, ProjectionId } from "@/utils/basic-quantization";
import { basicProjection } from "@/utils/basic-quantization";
import { getFormat, packNibbles } from "@/utils/quantization";
import { useLessonPlayback } from "./useLessonPlayback";

const CHAPTERS = ["The weights", "Fewer bits", "Back to floats", "The result"];
const f = (v: number) => Number(v.toFixed(3)).toString();
const clamp = (n: number) => Math.max(0, Math.min(1, n));

export function QuantizationScene({ projection, format, selected, cell, active }: {
  projection: ReturnType<typeof basicProjection>;
  format: BasicFormat;
  selected: ProjectionId;
  cell: [number, number];
  active: boolean;
}) {
  const [speed, setSpeed] = useState(1);
  const playback = useLessonPlayback(CHAPTERS.length, active, speed);
  const { chapter, time, duration, playing, reducedMotion: reduced } = playback;
  const phase = clamp(playback.chapterTime / 4);
  const spec = getFormat(format);
  const values = projection.source.flat();
  const stored = projection.result.q.flat();
  const reconstructed = projection.result.deq.flat();
  const scale = projection.result.params?.scales[0];
  const index = cell[0] * 4 + cell[1];
  const w = values[index]!;
  const q = stored[index]!;
  const reconstructedWeight = reconstructed[index]!;
  const max = Math.max(...values.map(Math.abs), ...reconstructed.map(Math.abs), 0.01) * 1.2;
  const position = (value: number) => 480 + value / max * 330;
  const explanation = [
    `The ${selected} projection has two rows of four weights. Each row uses the same four inputs and produces one output. The highlighted weight is ${f(w)}.`,
    scale !== undefined ? `The highlighted weight ${f(w)} is about ${f(w / scale)} scale steps from zero. Rounding gives code ${q}. The gold destination is where that code lands on the weight axis.` : `The highlighted weight ${f(w)} becomes ${f(q)} in the ${spec.label} model. The other weights are converted independently.`,
    scale !== undefined ? `Multiply the gold code ${q} by S to obtain the green weight ${f(reconstructedWeight)}. The same scale reconstructs all eight weights in this matrix.` : `The GPU reads the ${spec.label} values. The highlighted value used in this dot product is ${f(reconstructedWeight)}.`,
    `Both dot products use the same inputs. For row ${cell[0] + 1}, the baseline is ${f(projection.baseline[cell[0]]!)} and the reconstructed output is ${f(projection.output[cell[0]]!)}. Pink shows the signed difference.`,
  ][chapter];
  const equation = [
    `W${selected} · x = ${selected}`,
    scale !== undefined ? `q = round(w / S)    S = ${f(scale)}` : `BF16 → ${spec.label}`,
    scale !== undefined ? `ŵ = q × S` : `ŵ = stored ${spec.label} value`,
    `Δ${selected} = ${selected} − ${selected}̂`,
  ][chapter];

  return (
    <section className="math-theater -mx-4 mb-8 sm:-mx-6" aria-label="Animated quantization walkthrough">
      <div className="mx-auto max-w-6xl px-4 pt-8 sm:px-6 sm:pt-10">
        <div className="flex items-center justify-between gap-4 text-xs">
          <span className="font-mono text-theater-muted">A SMALL LESSON IN PRECISION</span>
          <span className="font-mono text-theater-gold">{String(chapter + 1).padStart(2, "0")} / 04</span>
        </div>
        <h2 className="mt-4 font-serif text-3xl sm:text-4xl">How do the weights change?</h2>
        <p className="mt-2 text-sm text-theater-muted">Eight weights. Fewer bits. Almost the same answer.</p>
        <div className="relative mt-4 aspect-[960/440] min-h-64 w-full overflow-hidden sm:min-h-0">
          <svg viewBox="0 0 960 440" className="h-full w-full" role="img" aria-label={`${CHAPTERS[chapter]}: ${equation}`}>
            <defs>
              <pattern id="math-grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" className="stroke-theater-grid" strokeWidth="0.5" /></pattern>
            </defs>
            <rect width="960" height="440" fill="url(#math-grid)" />
            <text x="480" y="55" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="32">{equation}</text>
            {chapter === 0 && <g>
              <text x="90" y="122" className="fill-theater-blue font-serif" fontSize="26">W_{selected}</text>
              <path d="M 205 140 H 190 V 300 H 205 M 565 140 H 580 V 300 H 565" fill="none" className="stroke-theater-blue" strokeWidth="2" />
              {values.map((value, i) => <g key={i} transform={`translate(${240 + (i % 4) * 90}, ${185 + Math.floor(i / 4) * 70})`} opacity={0.65 + 0.35 * clamp(phase * 2 - i / 12)}>
                {i === index && <rect x="-40" y="-25" width="80" height="36" rx="5" fill="none" className="stroke-theater-blue" />}
                <text textAnchor="middle" className="fill-theater-blue font-mono" fontSize="22">{f(value)}</text>
              </g>)}
              <text x="637" y="227" className="fill-theater-gold font-serif" fontSize="40">× x</text>
              <text x="766" y="227" className="fill-theater-green font-serif" fontSize="40">= {selected}</text>
              <text x="385" y="350" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="15">2 rows × 4 columns · 16 bits per weight</text>
            </g>}
            {chapter === 1 && <g>
              <line x1="100" y1="230" x2="860" y2="230" className="stroke-theater-muted" />
              {scale !== undefined && Array.from({length: 15}, (_, i) => {
                const value = (i - 7) * scale! * (format === "INT8" ? 18 : 1);
                const px = position(value);
                return <g key={i}><line x1={px} y1="220" x2={px} y2="240" className="stroke-theater-muted" /><text x={px} y="264" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="12">{(i - 7) * (format === "INT8" ? 18 : 1)}</text></g>;
              })}
              {values.map((value, i) => {
                const target = reconstructed[i] ?? value;
                const px = position(value + (target - value) * phase);
                return <g key={i}>
                  <line x1={position(value)} y1={100 + i % 3 * 30} x2={px} y2="225" className="stroke-theater-blue" strokeDasharray="3 5" opacity="0.4" />
                  <circle cx={position(target)} cy="230" r={i === index ? 7 : 4} opacity={i === index ? 1 : 0.45} className={scale !== undefined ? "fill-theater-gold" : "fill-theater-green"} />
                  <circle cx={px} cy={212 - (i % 2) * 14} r={i === index ? 8 : 5} opacity={i === index ? 1 : 0.65} className="fill-theater-blue" />
                  <text x={position(value)} y={95 + i % 3 * 30} textAnchor="middle" className="fill-theater-ink font-mono" fontSize="12">{f(value)}</text>
                </g>;
              })}
              <text x="480" y="322" textAnchor="middle" className="fill-theater-gold font-serif" fontSize="25">{spec.bits} bits per weight · {projection.payloadRatio.toFixed(1)}× smaller weight payload</text>
              <text x="480" y="356" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="14">{scale !== undefined ? "Continuous weights → discrete integer codes" : "More compact floating-point precision"}</text>
            </g>}
            {chapter === 2 && <g>
              <text x="210" y="121" textAnchor="middle" className="fill-theater-gold font-mono" fontSize="17">GPU HBM · {format}</text>
              <text x="742" y="121" textAnchor="middle" className="fill-theater-green font-mono" fontSize="17">SM REGISTERS · Ŵ</text>
              <path d="M 385 223 H 580" className="stroke-theater-muted" fill="none" strokeDasharray="5 8" />
              <path d="M 570 216 L 580 223 L 570 230" fill="none" className="stroke-theater-muted" />
              {values.map((_, i) => <g key={i}>
                <text x={125 + i % 4 * 60} y={188 + Math.floor(i / 4) * 66} textAnchor="middle" className="fill-theater-gold font-mono" fontSize="20">{f(stored[i] ?? 0)}</text>
                <circle cx={400 + 165 * ((phase + i / 8) % 1)} cy={210 + Math.floor(i / 4) * 25} r="3" className="fill-theater-gold" opacity={playing ? 0.8 : 0.4} />
                <text x={614 + i % 4 * 80} y={188 + Math.floor(i / 4) * 66} textAnchor="middle" className="fill-theater-green font-mono" fontSize="18">{f(reconstructed[i] ?? 0)}</text>
              </g>)}
              {format === "INT4" && <text x="210" y="300" textAnchor="middle" className="fill-theater-gold font-mono" fontSize="15">{packNibbles(stored).map(b => b.hex).join("  ")}</text>}
              <text x="480" y="367" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="15">{projection.weightBytes} B weights{scale !== undefined ? ` + ${projection.scaleBytes} B scale` : ""} → floating-point multiply</text>
            </g>}
            {chapter === 3 && <g>
              <text x="260" y="125" textAnchor="middle" className="fill-theater-blue font-serif" fontSize="25">BF16 baseline · {selected}</text>
              <text x="690" y="125" textAnchor="middle" className="fill-theater-green font-serif" fontSize="25">Quantized · {selected}̂</text>
              {projection.baseline.map((value, i) => <g key={i}>
                <text x="260" y={190 + i * 74} textAnchor="middle" className="fill-theater-blue font-mono" fontSize="36">{value.toFixed(4)}</text>
                <path d={`M 390 ${180 + i * 74} H 548`} className="stroke-theater-muted" strokeDasharray="3 6" />
                <text x="690" y={190 + i * 74} textAnchor="middle" className="fill-theater-green font-mono" fontSize="36" opacity={0.55 + 0.45 * phase}>{(projection.output[i] ?? 0).toFixed(4)}</text>
                <text x="690" y={218 + i * 74} textAnchor="middle" className="fill-theater-rose font-mono" fontSize="15">Δ = {(projection.delta[i] ?? 0).toFixed(5)}</text>
              </g>)}
              <text x="480" y="370" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="26">Less memory. A small approximation.</text>
            </g>}
          </svg>
        </div>
        <div className="mx-auto min-h-16 max-w-2xl text-center"><p className="font-serif text-lg leading-relaxed text-theater-ink" aria-live={playing ? "off" : "polite"}>{explanation}</p></div>
        <div className="mt-4 grid grid-cols-4 gap-2 border-b border-theater-grid pb-4">
          {CHAPTERS.map((label, i) => <Button key={label} variant="ghost" onClick={() => playback.selectChapter(i)} aria-pressed={chapter === i} className={cn("h-auto min-h-10 whitespace-normal px-1 text-xs sm:text-sm", chapter === i ? "text-theater-gold" : "text-theater-muted")}><span className="hidden font-mono sm:inline">0{i + 1}</span>{label}</Button>)}
        </div>
        <div className="flex flex-wrap items-center gap-3 py-4">
          <Button size="icon" variant="ghost" title="Restart animation" aria-label="Restart animation" onClick={() => playback.seek(0)}><RotateCcw /></Button>
          <Button size="icon" variant="ghost" title="Previous chapter" aria-label="Previous chapter" disabled={chapter === 0} onClick={() => playback.selectChapter(chapter - 1)}><SkipBack /></Button>
          <Button size="icon" variant="secondary" title={playing ? "Pause animation" : "Play animation"} aria-label={playing ? "Pause animation" : "Play animation"} disabled={reduced} onClick={playback.togglePlayback}>{playing ? <Pause /> : <Play />}</Button>
          <Button size="icon" variant="ghost" title="Next chapter" aria-label="Next chapter" disabled={chapter === 3} onClick={() => playback.selectChapter(chapter + 1)}><SkipForward /></Button>
          <input className="min-w-24 flex-1 accent-theater-gold" type="range" min="0" max={duration} step={playback.scrubStep} value={time} aria-label="Animation progress" onChange={e => playback.seek(Number(e.target.value))} />
          <span className="w-20 text-right font-mono text-xs text-theater-muted">{time.toFixed(1)} / {duration}s</span>
          <select aria-label="Playback speed" className="rounded-md border border-theater-grid bg-theater px-2 py-1 text-xs text-theater-ink" value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select>
          {reduced && <p className="w-full text-xs text-theater-muted">Reduced motion: select a chapter to view each step.</p>}
        </div>
      </div>
    </section>
  );
}