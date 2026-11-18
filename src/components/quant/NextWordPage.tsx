import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { ArrowLeft, ArrowRight, Pause, Play } from "lucide-react";
import { ACTIVATION, MAX_PROMPT_CHARS, predictionTrace, WEIGHT_SCALE, WEIGHT_START } from "@/utils/next-word";
import type { NextWords, PredictionMessage } from "@/utils/next-word";
import { TraceReadout } from "./TraceReadout";
import { useLessonPlayback } from "./useLessonPlayback";
import "./next-word.css";

const samplePrompt = "The sky is";
const sample: NextWords = {
  continuation: " blue today", wordCount: 2,
  candidates: [{ text: " blue", logit: 0.08 }, { text: " clear", logit: 0 }],
};
const phases = ["Compare scores", "Round the weight", "Read the result"];
const phaseSeconds = 4;
const number = (value: number) => Number(value.toFixed(6)).toString();
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const displayToken = (text: string) => text.trim() || JSON.stringify(text);

export default function NextWordPage() {
  const worker = useRef<Worker | null>(null);
  const downloadDialog = useRef<HTMLDialogElement>(null);
  const pendingPrompt = useRef<string | null>(null);
  const [prompt, setPrompt] = useState(samplePrompt);
  const [result, setResult] = useState<{ prompt: string; prediction: NextWords; isSample: boolean } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [run, setRun] = useState(0);

  useEffect(() => () => worker.current?.terminate(), []);

  function abandonWorker() {
    worker.current?.terminate();
    worker.current = null;
  }

  function showExample() {
    setResult({ prompt: samplePrompt, prediction: sample, isSample: true });
    setRun(previous => previous + 1);
  }

  function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || !prompt.trim()) return;
    if (worker.current) {
      startPrediction(prompt);
    } else {
      pendingPrompt.current = prompt;
      downloadDialog.current?.showModal();
    }
  }

  function answerDownload(approved: boolean) {
    const submitted = pendingPrompt.current;
    pendingPrompt.current = null;
    downloadDialog.current?.close();
    if (submitted === null) return;
    if (approved) {
      startPrediction(submitted);
    } else {
      setResult(null);
      setError("");
      setStatus("Download declined. The animation can’t run without DistilGPT2.");
    }
  }

  function startPrediction(submitted: string) {
    setBusy(true);
    setResult(null);
    setError("");
    setStatus("Starting the local model…");
    try {
      const engine = worker.current ??= new Worker(new URL("../../utils/next-word.worker.ts", import.meta.url), { type: "module" });
      engine.onmessage = ({ data }: MessageEvent<PredictionMessage>) => {
        if (worker.current !== engine) return;
        if (data.kind === "progress") {
          setStatus(data.percent === undefined ? "Loading model files…" : `Downloading model weights… ${data.percent}%`);
        } else if (data.kind === "generating") {
          setStatus("Predicting in your browser…");
        } else {
          setBusy(false);
          setStatus("");
          if (data.kind === "error") {
            abandonWorker();
            setError(data.message);
          } else {
            setResult({ prompt: submitted, prediction: data.prediction, isSample: false });
            setRun(previous => previous + 1);
          }
        }
      };
      engine.onerror = () => {
        if (worker.current !== engine) return;
        abandonWorker();
        setBusy(false);
        setStatus("");
        setError("The browser could not start local inference. Try again to run the animation.");
      };
      engine.postMessage(submitted);
    } catch {
      abandonWorker();
      setBusy(false);
      setStatus("");
      setError("The animation can’t run because this browser could not start the model worker.");
    }
  }

  return (
    <main className="quant-lesson next-word-page">
      <header className="lesson-header">
        <div className="lesson-shell lesson-header-inner">
          <a className="lesson-brand" href="./"><ArrowLeft size={17} /> Back to studio</a>
          <a className="lesson-lab" href="./#lesson">Weight walkthrough</a>
        </div>
      </header>
      <div className="lesson-shell">
        <div className="next-word-intro">
          <h1>One rounded weight,<br />one different word?</h1>
          <p>Predict two words on your device. Then move one weight across an INT4 rounding boundary and see whether the leading token changes.</p>
        </div>
        <form className="lesson-reference next-word-prompt" onSubmit={predict}>
          <label htmlFor="next-word-input">Your text</label>
          <textarea id="next-word-input" rows={2} maxLength={MAX_PROMPT_CHARS} value={prompt}
            onChange={event => setPrompt(event.target.value)} placeholder="Start a sentence…" />
          <div className="next-word-actions">
            <button type="submit" className="next-word-primary" disabled={busy || !prompt.trim()}>Predict two words <ArrowRight size={16} /></button>
            <button type="button" className="next-word-secondary" onClick={() => setPrompt(samplePrompt)}>Use example prompt</button>
            {busy && <button type="button" className="next-word-secondary" onClick={() => {
              abandonWorker(); setBusy(false); setStatus("Prediction canceled.");
            }}>Cancel</button>}
          </div>
          <p>DistilGPT2 runs in your browser and works best with English prompts. Your text stays on your device. <a className="next-word-source" href="https://huggingface.co/distilbert/distilgpt2">Model details</a></p>
          {status && <p className="next-word-status" role="status">{status}</p>}
          {error && <p className="next-word-error" role="alert">{error}</p>}
        </form>
        <dialog ref={downloadDialog} className="next-word-download" aria-labelledby="next-word-download-title"
          aria-describedby="next-word-download-details" onCancel={event => { event.preventDefault(); answerDownload(false); }}>
          <h2 id="next-word-download-title">Download DistilGPT2?</h2>
          <p id="next-word-download-details">DistilGPT2 predicts the next two words in your browser. The animation uses its first-token scores to show how rounding a weight could change the leading token. The first run downloads about 85 MB of model weights plus the runtime.</p>
          <div className="next-word-actions">
            <button type="button" className="next-word-secondary" onClick={() => answerDownload(false)}>No</button>
            <button type="button" className="next-word-primary" onClick={() => answerDownload(true)}>Download and predict</button>
          </div>
        </dialog>
        {result && <>
          <div className="next-word-result" aria-live="polite">
            <p className="next-word-label">{result.isSample ? "Illustrative continuation" : "Local model continuation"}</p>
            <p className="next-word-phrase">{result.prompt}<strong>{result.prediction.continuation}</strong></p>
            {result.prediction.wordCount < 2 && <p>The model produced {result.prediction.wordCount === 1 ? "one word" : "no words"} within its generation limit.</p>}
          </div>
          <WeightToWord key={run} prediction={result.prediction} isSample={result.isSample} />
          {!result.isSample && <div className="next-word-actions">
            <button type="button" className="next-word-secondary" onClick={showExample}>Show near-tie example</button>
          </div>}
        </>}
      </div>
    </main>
  );
}

function WeightToWord({ prediction, isSample }: { prediction: NextWords; isSample: boolean }) {
  const [weight, setWeight] = useState(WEIGHT_START);
  const playback = useLessonPlayback(phases.length, true, 1, phaseSeconds);
  const { chapter, chapterTime, playing, reducedMotion, seek, time, duration } = playback;
  const progress = chapter === 0 ? 0 : chapter === 2 || reducedMotion ? 1 : Math.min(1, chapterTime / (phaseSeconds * 0.7));
  const start = predictionTrace(weight, prediction.candidates, 0);
  const end = predictionTrace(weight, prediction.candidates, 1);
  const frame = predictionTrace(weight, prediction.candidates, progress);
  const tokens = prediction.candidates.map(candidate => displayToken(candidate.text));
  const winner = (probability: number) => Math.abs(probability - 0.5) < 1e-8 ? "Tie" : tokens[probability > 0.5 ? 0 : 1]!;
  const firstWinner = winner(start.firstProbability);
  const finalWinner = winner(end.firstProbability);
  const readout = [
    { value: number(weight), name: "original weight", why: "The starting scores are the reference. One weight contributes w × x to the first score.",
      next: "Divide the weight by the INT4 scale and round to the nearest code." },
    { value: `${number(weight)} → ${number(end.restored)}`, name: `INT4 code ${end.code}`,
      why: `round(${number(weight)} / ${WEIGHT_SCALE}) = ${end.code}. Multiply by the scale to reconstruct the weight.`,
      next: "Apply the change in the weight’s contribution and compare the two scores." },
    { value: finalWinner, name: finalWinner === "Tie" ? "equal simulated scores" : "leading token after rounding",
      why: firstWinner === finalWinner ? "The score change kept the same ranking." : `The score change moved the lead from ${firstWinner} to ${finalWinner}.`,
      next: "Drag the weight across the midpoint and test where the outcome changes." },
  ][chapter]!;

  return (
    <section className="lesson-reference next-word-stage" aria-labelledby="next-word-trace">
      <h2 id="next-word-trace">Follow one weight into the prediction</h2>
      <p>{isSample
        ? "This example uses two hand-picked scores."
        : "The starting scores come from the local model’s first token. The INT4 change below simulates one weight contribution."} Probability bars compare only these two tokens.</p>
      <div className="experiment-buttons" role="group" aria-label="Prediction animation steps">
        {phases.map((phase, index) => (
          <button key={phase} type="button" aria-pressed={chapter === index}
            onClick={() => seek(index * phaseSeconds + (index === 1 ? phaseSeconds * 0.7 : 0))}>{phase}</button>
        ))}
      </div>
      <div className="next-word-math">
        <div className="next-word-weight">
          <h3>One weight</h3>
          <div className="next-word-ruler" aria-hidden="true">
            <span className="next-word-midpoint">0.0625 midpoint</span>
            {progress > 0 && <span className="next-word-dot next-word-origin" style={{ left: `${100 * weight / WEIGHT_SCALE}%` }} />}
            <span className="next-word-dot" style={{ left: `${100 * frame.movingWeight / WEIGHT_SCALE}%` }} data-rounded={progress > 0} />
            <span className="next-word-tick">0</span><span className="next-word-tick">0.125</span>
          </div>
          <label className="experiment-range">
            <span>Original weight <strong>{number(weight)}</strong></span>
            <input type="range" step={1 / 512} min={0} max={WEIGHT_SCALE} value={weight}
              onChange={event => { setWeight(Number(event.target.value)); seek(2 * phaseSeconds); }} />
          </label>
          <p className="next-word-equation">S = {WEIGHT_SCALE} · x = {ACTIVATION}</p>
        </div>
        <div className="next-word-scores">
          <h3>Two possible next tokens</h3>
          {[frame.firstProbability, 1 - frame.firstProbability].map((probability, index) => (
            <div key={index} className="next-word-score" data-leading={probability > 0.5 + 1e-8}>
              <div><strong>{tokens[index]}</strong><span>{percent(probability)}</span></div>
              <div className="next-word-bar" aria-hidden="true"><span style={{ width: percent(probability) }} /></div>
              <small>Score {number(index === 0 ? frame.firstLogit : frame.secondLogit)}</small>
            </div>
          ))}
          <p className="next-word-equation">Final score shift = (ŵ − w) × x = {number((end.restored - weight) * ACTIVATION)}</p>
        </div>
      </div>
      <TraceReadout {...readout} playing={playing} />
      <div className="next-word-controls" role="group" aria-label="Prediction playback">
        <button type="button" className="next-word-secondary" disabled={reducedMotion} onClick={playback.togglePlayback}>
          {playing ? <Pause size={16} /> : <Play size={16} />} {playing ? "Pause animation" : "Play animation"}
        </button>
        <label><span>Animation progress</span>
          <input type="range" min={0} max={duration} step={playback.scrubStep} value={time}
            aria-valuetext={phases[chapter]} onChange={event => seek(Number(event.target.value))} />
        </label>
      </div>
    </section>
  );
}
