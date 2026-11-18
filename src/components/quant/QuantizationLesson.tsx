import { useState } from "react";
import { ArrowRight, AudioLines, ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from "lucide-react";
import { floatBitsFor, floatStep, getFormat, packNibbles, twosComplement } from "@/utils/quantization";
import { useLessonPlayback } from "./useLessonPlayback";
import { ModelStorage, SymbolLegend } from "./LessonReference";
import { CompressionExperiment, RoundingBoundary } from "./LessonExperiments";
import { TraceReadout } from "./TraceReadout";
import {
  ACTIVATION_CODES, ACTIVATION_SCALE, FP16_INPUT, FP16_WEIGHTS, INFERENCE_MODES, INFERENCE_OUTPUTS,
  INPUT, INT8_SUMS, MAX_WEIGHT, PROJECTIONS, WEIGHT_FORMATS,
} from "./lesson-example";
import type { LessonMode } from "./lesson-example";
import "./quantization-lesson.css";

const short = (value: number) => Number(value.toPrecision(7)).toString();

function reveal(progress: number, index: number) {
  const visibility = Math.max(0, Math.min(1, progress * 2 - index * 0.13));
  return {
    opacity: 0.7 + 0.3 * visibility,
    transform: `translateY(${7 * (1 - visibility)}px)`,
  };
}

export function QuantizationLesson({ onOpenStudio }: { onOpenStudio: () => void }) {
  const [mode, setMode] = useState<LessonMode>("INT4");
  const activationBits = mode === "W8A8" ? 8 : mode === "W8A16" ? 16 : undefined;
  const format = mode === "W8A16" || mode === "W8A8" ? "INT8" : mode;
  const [cell, setCell] = useState(0);
  const spec = getFormat(format);
  const chapters = [spec.kind === "int" ? "Set the scale" : "Read the format", "Round the weight", "Bring it back", "Multiply again"];
  const playback = useLessonPlayback(chapters.length);
  const { chapter, time, duration, playing, reducedMotion } = playback;
  const phase = Math.min(1, playback.chapterTime / 3.5);
  const progress = phase * phase * (3 - 2 * phase);
  const projection = PROJECTIONS[format];
  const { result, source, baseline, output, weightBytes, scaleBytes } = projection;
  const scale = result.params?.scales[0];
  const qmax = result.params?.qmax;
  const { bits } = spec;
  const weights = source.flat();
  const codes = result.q.flat();
  const weight = weights[cell]!;
  const code = codes[cell]!;
  const row = Math.floor(cell / source[0]!.length);
  const column = cell % source[0]!.length;
  const reconstructed = result.deq[row]![column]!;
  const baselineValue = baseline[row]!;
  const outputValue = mode === "W8A16" || mode === "W8A8" ? INFERENCE_OUTPUTS[mode][row]! : output[row]!;
  const activation = INPUT[column]!;
  const activationCode = ACTIVATION_CODES[column]!;
  const activationValue = activationBits === 8 ? activationCode * ACTIVATION_SCALE : FP16_INPUT[column]!;
  const dotWeights = activationBits === 8 ? result.q[row]! : activationBits === 16 ? FP16_WEIGHTS[row]! : result.deq[row]!;
  const dotInputs = activationBits === 8 ? ACTIVATION_CODES : activationBits === 16 ? FP16_INPUT : INPUT;
  const float = spec.kind === "float" ? floatBitsFor(format, code) : undefined;
  const storedBits = float?.binary ?? twosComplement(code, bits);
  const packedByte = format === "INT4" ? packNibbles(codes)[Math.floor(cell / 2)] : undefined;
  const exponentBias = spec.exponentBits ? 2 ** (spec.exponentBits - 1) - 1 : 0;
  const exponent = float ? (float.exp || 1) - exponentBias : 0;
  const significand = float ? Number(float.exp !== 0) + float.man / 2 ** spec.mantissaBits! : 0;
  const roundingStep = float ? floatStep(Math.abs(weight), spec.exponentBits!, spec.mantissaBits!) : scale!;
  const weightError = weight - reconstructed;
  const delta = baselineValue - outputValue;

  const plotX = (value: number) => 50 + (value / MAX_WEIGHT) * 45;
  const codeMarks = bits === 4
    ? Array.from({ length: 15 }, (_, i) => i - 7)
    : [...[-127, -96, -64, -32, 0, 32, 64, 96, 127].filter(mark => Math.abs(mark - code) >= 8), code]
        .sort((a, b) => a - b);
  const marks = scale === undefined
    ? [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75].map(value => ({ label: value, weight: value }))
    : codeMarks.map(mark => ({ label: mark, weight: mark * scale }));
  const descriptions = [
    scale !== undefined
      ? `The blue dots are the original weights. The largest magnitude, ${MAX_WEIGHT}, sets the edge of the grid. Divide it by ${qmax} to find S, the difference in weight represented by consecutive codes.`
      : `${spec.label} stores each weight in ${bits} bits. The sign says whether it is positive or negative, the exponent sets its magnitude, and the fraction fills in the detail. The spacing grows with magnitude.`,
    scale !== undefined
      ? `The selected weight is ${short(weight)}. Dividing by S gives about ${short(weight / scale)} grid steps. Rounding chooses integer code ${code}.`
      : `At this magnitude, the ${spec.label} spacing is ${short(roundingStep)}. Divide by that step, round to the nearest even integer at a tie, then multiply by the step. ${weightError === 0 ? (format === "FP16" ? "This weight is exact. Try W[2, 4] to see rounding near zero." : "This weight is exact.") : `The rounded value is ${short(reconstructed)}.`}`,
    scale !== undefined
      ? `Read the stored bits as the signed integer ${code}. Multiply that code by the shared scale S to recover ${short(reconstructed)}. The original was ${short(weight)}.`
      : `The bits give a significand of ${short(significand)} and a power of ${exponent}. Apply the sign and multiply by that power of two to recover ${short(reconstructed)}. ${weightError === 0 ? "It matches the original weight." : "Widening preserves the rounded value."}`,
    activationBits === 8
      ? `Quantize the inputs using their own scale Sₓ ≈ ${short(ACTIVATION_SCALE)}. Multiply weight codes by input codes and accumulate in INT32. Apply S and Sₓ once to the sum of ${INT8_SUMS[row]}.`
      : activationBits === 16
        ? "Read INT8 weight codes, multiply by S, and round the reconstructed weights to FP16. Round each input to FP16 as well, then add their products."
        : `Multiply each reconstructed weight by its matching input. Add the four products to get about ${short(outputValue)}. Pink shows the original output minus the reconstructed output.`,
  ];
  const equations = [
    scale !== undefined
      ? <>S = {MAX_WEIGHT} / {qmax} <span>≈ {scale.toPrecision(5)}</span></>
      : <>{spec.label} <span>1 + {spec.exponentBits} + {spec.mantissaBits} bits</span></>,
    scale !== undefined
      ? <>round({short(weight)} / {short(roundingStep)}) <span>= {code}</span></>
      : <>roundEven({short(weight / roundingStep)}) × {short(roundingStep)} <span>≈ {short(reconstructed)}</span></>,
    float
      ? <>ŵ = {float.sign ? "−" : ""}{short(significand)} × 2<sup>{exponent}</sup> <span>≈ {short(reconstructed)}</span></>
      : <>ŵ = S × {code} <span>≈ {short(reconstructed)}</span></>,
    activationBits === 8
      ? <>ŷ = {INT8_SUMS[row]} × S × Sₓ <span>≈ {short(outputValue)}</span></>
      : <>ŷ = Σ ŵⱼ {activationBits === 16 ? "x₁₆,ⱼ" : "xⱼ"} <span>≈ {short(outputValue)}</span></>,
  ];
  const readout = chapter === 0 || (chapter === 1 && progress < 1)
    ? {
      value: short(weight), name: "Original weight",
      why: "This is the selected BF16 weight before rounding.",
      next: scale === undefined ? `Round to a representable ${spec.label} value.` : "Divide by S and round to a code.",
    }
    : chapter === 1 || (chapter === 2 && progress < 0.65)
      ? {
        value: chapter === 2 ? storedBits : short(code),
        name: chapter === 2 ? "Stored bits" : scale === undefined ? "Rounded weight" : "Stored integer code",
        why: scale === undefined ? `The format rounded the original weight to ${short(reconstructed)}.` : `Rounding w / S selected code ${code}.`,
        next: chapter === 1 ? `Encode in ${bits} bits.`
          : scale === undefined ? "Decode the sign, exponent, and fraction." : "Read the code and multiply by S.",
      }
      : chapter === 2 || progress < 0.65
        ? {
          value: short(reconstructed), name: "Reconstructed weight",
          why: scale === undefined ? "The stored bits decode to the rounded value." : `Multiply code ${code} by S to reconstruct the weight.`,
          next: activationBits === 8 ? "Multiply weight and activation codes, then rescale the sum." : "Multiply by the matching input and add the row products.",
        }
        : {
          value: short(outputValue), name: `Output for row ${row + 1}`,
          why: activationBits === 8 ? "Apply both scales to the integer sum." : "Add the products from this row.",
          next: "Compare with the BF16-weight output.",
        };

  return (
    <main className="quant-lesson">
      <header className="lesson-header">
        <div className="lesson-shell lesson-header-inner">
          <a className="lesson-brand" href="./" aria-label="Quantization Studio home"><AudioLines size={19} /> Quantization Studio</a>
          <button className="lesson-lab" onClick={onOpenStudio}>Open lab <ArrowRight size={16} /></button>
        </div>
      </header>

      <div className="lesson-shell">
        <div className="lesson-intro">
          <div>
            <h1>Where does the precision go?</h1>
            <p>Quantization stores model weights with fewer bits. Follow one weight and see how the approximation changes the answer.</p>
          </div>
          <div className="lesson-mode-groups">
            <div className="lesson-format" role="group" aria-label="Weight precision">
              {WEIGHT_FORMATS.map(value => (
                <button key={value} onClick={() => setMode(value)} aria-pressed={mode === value}>{value}</button>
              ))}
            </div>
            <div className="lesson-format" role="group" aria-label="Weight and activation precision">
              {INFERENCE_MODES.map(value => (
                <button key={value} onClick={() => setMode(value)} aria-pressed={mode === value}>{value}</button>
              ))}
            </div>
          </div>
        </div>

        <nav className="lesson-experiment-links" aria-label="Lesson experiments">
          <a href="#rounding-boundary">Try a rounding boundary</a>
          <a href={`?view=compare&weight=${cell}`}>Compare weight configurations</a>
          <a href="#compression">Explore model compression</a>
          <a href="?view=kv-cache">KV cache &amp; GPU memory</a>
        </nav>

        <section className="lesson-player" aria-label="Interactive quantization lesson">
          <nav className="lesson-chapters" aria-label="Lesson chapters">
            {chapters.map((label, index) => (
              <button key={label} type="button" onClick={() => playback.selectChapter(index)} aria-current={chapter === index ? "step" : undefined}>
                <span>{index + 1}</span>{label}
              </button>
            ))}
          </nav>

          <div className="lesson-stage">
            <div className="lesson-canvas">
              {chapter < 2 ? (
                <>
                  <div className="lesson-legend">
                    <span><i /> Original weight</span>
                    <span><i /> {scale === undefined ? "Rounded value" : "Integer grid"}</span>
                  </div>
                  <div className="lesson-plot" role="img" aria-label={`Original weight ${short(weight)} maps to ${short(reconstructed)} in ${spec.label}.`}>
                    <div className="lesson-axis" />
                    {marks.map(mark => (
                      <div key={mark.label} className={`lesson-tick${mark.weight === reconstructed ? " lesson-tick-selected" : ""}`} style={{ left: `${plotX(mark.weight)}%` }}>
                        <span>{mark.label}</span>
                      </div>
                    ))}
                    {weights.map((value, index) => (
                      index !== cell && <i key={index} className="lesson-context-dot" style={{ left: `${plotX(value)}%`, top: `${28 + (index % 3) * 14}%` }} />
                    ))}
                    <div className="lesson-original" style={{ left: `${plotX(weight)}%` }} />
                    <div className="lesson-target" style={{ left: `${plotX(reconstructed)}%` }} />
                    <div className="lesson-dot" style={{
                      left: `${plotX(chapter === 1 ? weight + (reconstructed - weight) * progress : weight)}%`,
                      top: `${chapter === 1 ? 26 + 40 * progress : 26}%`,
                    }} />
                    <span className="lesson-axis-label">{scale === undefined ? "weight value" : "integer code q"}</span>
                  </div>
                  <div className="lesson-pair">
                    <span>Original <b>{short(weight)}</b></span>
                    <ArrowRight size={17} />
                    <span>Reconstructed <b data-role="reconstructed">{short(reconstructed)}</b></span>
                  </div>
                </>
              ) : chapter === 2 ? (
                <div className="lesson-data-scene">
                  <h3>From stored bits to a weight</h3>
                  <div className="lesson-reconstruction">
                    <div data-role="original">
                      <small>Original BF16 · 16 bits</small>
                      <strong>{short(weight)}</strong>
                    </div>
                    <ArrowRight size={17} aria-hidden="true" />
                    <div data-role="code" style={reducedMotion ? undefined : reveal(progress, 0)}>
                      <small>Stored {format} · {bits} bits</small>
                      <strong className="lesson-bit-string" aria-label={`Bits ${storedBits}`}>
                        {float ? <>
                          <span data-field="sign">{storedBits[0]}</span>
                          <span data-field="exponent">{storedBits.slice(1, 1 + spec.exponentBits!)}</span>
                          <span data-field="fraction">{storedBits.slice(1 + spec.exponentBits!)}</span>
                        </> : storedBits}
                      </strong>
                      <small>{float ? float.hex : `q = ${code}`}{packedByte ? ` · packed ${packedByte.hex}` : ""}</small>
                    </div>
                    <span className="lesson-decode-arrow" aria-hidden="true">
                      <ArrowRight size={17} />
                      {playing && !reducedMotion && <i style={{ left: `${progress * 100}%`, opacity: 1 - progress }} />}
                    </span>
                    <div data-role="reconstructed" style={reducedMotion ? undefined : reveal(progress, 3)}>
                      <small>Reconstructed</small>
                      <strong>{short(reconstructed)}</strong>
                    </div>
                  </div>
                  {float && <div className="lesson-bit-legend" aria-label="Floating-point bit fields">
                    <span data-field="sign">Sign</span>
                    <span data-field="exponent">Exponent</span>
                    <span data-field="fraction">Fraction</span>
                  </div>}
                  <p className="lesson-error" data-role="error">Δ weight = {short(weightError)}</p>
                  <p>{weightBytes} B stored weights{scaleBytes > 0 && ` + ${scaleBytes} B for S`} · <strong>{weightBytes + scaleBytes} B total</strong> vs {projection.baselineBytes} B BF16</p>
                </div>
              ) : (
                <div className="lesson-data-scene lesson-product">
                  <h3>{activationBits === 8 ? "Multiply and accumulate the codes" : `Dot product for row ${row + 1}`}</h3>
                  {activationBits !== undefined && (
                    <p className="lesson-activation">
                      Input x[{column + 1}] · FP32 {short(activation)}
                      {activationBits === 8
                        ? <> → round(x / Sₓ) = <b data-role="code">{activationCode}</b> → {activationCode} × Sₓ ≈ <b>{short(activationValue)}</b></>
                        : <> → FP16 <b>{short(activationValue)}</b></>}
                    </p>
                  )}
                  <div className="lesson-terms" aria-label={activationBits === 8 ? "Weight codes times input codes" : "Reconstructed weights times inputs"}>
                    {dotWeights.map((value, index) => (
                      <div key={index} className={index === column ? "active" : ""} style={reducedMotion ? undefined : reveal(progress, index)}>
                        <strong data-role={activationBits === 8 ? "code" : "reconstructed"}>{short(value)}</strong>
                        <span>× {short(dotInputs[index]!)}</span>
                      </div>
                    ))}
                  </div>
                  {activationBits === 8 && <p className="lesson-activation">Integer sum {INT8_SUMS[row]} × S × Sₓ ≈ <strong>{short(outputValue)}</strong></p>}
                  <div className="lesson-pair">
                    <span>BF16 output <b>{short(baselineValue)}</b></span><ArrowRight size={17} />
                    <span>{mode} output <b data-role="reconstructed">{short(outputValue)}</b></span>
                  </div>
                  <p className="lesson-error" data-role="error">Δ = {short(delta)}</p>
                </div>
              )}
            </div>

            <aside className="lesson-inspector">
              <div>
                <h2>{chapters[chapter]}</h2>
                <p aria-live={playing ? "off" : "polite"}>{descriptions[chapter]}</p>
              </div>
              <div className="lesson-equation">{equations[chapter]}</div>
              <div>
                <p className="lesson-weight-label">Select a weight <span>W[{row + 1}, {column + 1}]</span></p>
                <div className="lesson-weights" role="group" aria-label="Projection weights">
                  {weights.map((value, index) => (
                    <button key={index} onClick={() => setCell(index)} aria-pressed={cell === index}
                      aria-label={`Row ${Math.floor(index / 4) + 1}, column ${index % 4 + 1}, weight ${value}`}>
                      {Math.abs(value) < 1e-4 ? value.toExponential(1) : short(value)}
                    </button>
                  ))}
                </div>
              </div>
            </aside>
          </div>

          <div className="lesson-player-readout">
            <TraceReadout {...readout} playing={playing} />
          </div>
          <div className="lesson-controls">
            <button className="lesson-icon" aria-label="Restart lesson" onClick={() => playback.seek(0)} title="Restart lesson"><RotateCcw size={17} /></button>
            <button className="lesson-icon" aria-label="Previous chapter" disabled={chapter === 0} onClick={() => playback.selectChapter(chapter - 1)}><ChevronLeft size={20} /></button>
            <button className="lesson-play" aria-label={playing ? "Pause lesson" : "Play lesson"} disabled={reducedMotion} onClick={playback.togglePlayback}>{playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />}</button>
            <button className="lesson-icon" aria-label="Next chapter" disabled={chapter === chapters.length - 1} onClick={() => playback.selectChapter(chapter + 1)}><ChevronRight size={20} /></button>
            <input type="range" aria-label="Lesson progress" aria-valuetext={`${chapters[chapter]}, ${Math.round(time)} seconds`}
              min={0} max={duration} step={playback.scrubStep} value={time}
              onChange={event => playback.seek(Number(event.target.value))} />
            <span className="lesson-time">{Math.floor(time)} / {duration}s</span>
          </div>
          {reducedMotion && <p className="lesson-motion-note">Select a chapter or use the scrubber to step through the lesson.</p>}
        </section>
        <div className="lesson-references">
          <ModelStorage initial="BF16" formats={WEIGHT_FORMATS} selected={format} />
          <SymbolLegend />
        </div>
        <section className="lesson-reference lesson-inference" aria-label="Compression and inference">
          <h3>What changes at inference?</h3>
          {activationBits !== undefined && <p>
            {mode} uses INT8 weights and {activationBits === 8 ? "INT8" : "FP16"} activations. Both modes store the same INT8 weights.
            The four inputs in this example go from 16 B in FP32 to {activationBits === 8 ? "4 B of codes plus a 4 B activation scale" : "8 B in FP16"}.
          </p>}
          <p>
            {bits === 16
              ? "FP16 uses the same 16 bits per weight as BF16, so it saves no weight bandwidth in this example."
              : `Each stored weight goes from 16 to ${bits} bits. That cuts weight traffic when the kernel streams those weights from memory.${scale === undefined ? "" : " Scale metadata adds some bytes back."}`}
            {" "}
            {activationBits === 8
              ? "A supported INT8 kernel can also do the dot product on integer codes before rescaling. Input quantization and rescaling add work."
              : "A weight-only kernel can unpack or widen values for the dot product. Smaller transfers help most when loading weights is the bottleneck."}
          </p>
          <p>Actual inference speed depends on the hardware, kernel, and workload. The chart measures storage. <a href="https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/inference-library/explicit-quantization.html">TensorRT quantization reference</a></p>
        </section>
        <RoundingBoundary />
        <CompressionExperiment />
        <p className="lesson-footnote">{scale === undefined ? "Direct float conversion · round to nearest, ties to even · widening preserves the stored value" : "Symmetric integer quantization · one scale per matrix · halfway codes round toward +∞"}</p>
      </div>
    </main>
  );
}
