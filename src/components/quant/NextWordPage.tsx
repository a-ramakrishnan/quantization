import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { ArrowLeft, ArrowRight, Pause, Play } from "lucide-react";
import { ACTIVATION, MAX_PROMPT_CHARS, predictionTrace, WEIGHT_SCALE, WEIGHT_START } from "@/utils/next-word";
import type { NextWords, PredictionMessage, PredictionRequest, WordPath } from "@/utils/next-word";
import { TraceReadout } from "./TraceReadout";
import { useLessonPlayback } from "./useLessonPlayback";
import "./next-word.css";

const examplePrompt = "She opened the";
const samplePrompt = "The sky is";
const sample: NextWords = {
  continuation: " blue today", wordCount: 2,
  candidates: [
    { word: "blue", completion: " blue", score: -0.62, pieces: [] },
    { word: "clear", completion: " clear", score: -0.7, pieces: [] },
  ],
};
const phases = ["Compare words", "Round the weight", "Read the result"];
const branchIndices = [0, 1] as const;
type BranchIndex = typeof branchIndices[number];
type BranchResult = { continuation: string } | { error: string };
const phaseSeconds = 4;
const number = (value: number) => Number(value.toFixed(6)).toString();
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export default function NextWordPage() {
  const worker = useRef<Worker | null>(null);
  const downloadDialog = useRef<HTMLDialogElement>(null);
  const pendingPrompt = useRef<string | null>(null);
  const [prompt, setPrompt] = useState(examplePrompt);
  const [result, setResult] = useState<{ prompt: string; prediction: NextWords; isSample: boolean } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [run, setRun] = useState(0);
  const [following, setFollowing] = useState<BranchIndex | null>(null);
  const [branches, setBranches] = useState<Partial<Record<BranchIndex, BranchResult>>>({});

  useEffect(() => () => worker.current?.terminate(), []);

  function abandonWorker() {
    worker.current?.terminate();
    worker.current = null;
  }

  function showExample() {
    setBranches({});
    setResult({ prompt: samplePrompt, prediction: sample, isSample: true });
    setRun(previous => previous + 1);
  }

  function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || following !== null || !prompt.trim()) return;
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
    setBranches({});
    setError("");
    setStatus("Starting the local model…");
    try {
      const engine = worker.current ??= new Worker(new URL("../../utils/next-word.worker.ts", import.meta.url), { type: "module" });
      requestPrediction(engine, { kind: "predict", prompt: submitted });
    } catch {
      abandonWorker();
      setBusy(false);
      setStatus("");
      setError("The animation can’t run because this browser could not start the model worker.");
    }
  }

  function followWord(index: BranchIndex, candidate: WordPath) {
    const engine = worker.current;
    if (!engine || !result || following !== null || busy) return;
    setFollowing(index);
    requestPrediction(engine, { kind: "follow", prompt: result.prompt, completion: candidate.completion }, index);
  }

  function requestPrediction(engine: Worker, request: PredictionRequest, branch?: BranchIndex) {
    function finishBranch(response: BranchResult) {
      if (branch === undefined) return;
      setBranches(previous => ({ ...previous, [branch]: response }));
      setFollowing(null);
    }

    engine.onmessage = ({ data }: MessageEvent<PredictionMessage>) => {
      if (worker.current !== engine) return;
      if (branch !== undefined) {
        if (data.kind === "follow-result") finishBranch({ continuation: data.continuation });
        if (data.kind === "error") finishBranch({ error: data.message });
        return;
      }
      if (data.kind === "progress") {
        setStatus(data.percent === undefined ? "Loading model files…" : `Downloading model weights… ${data.percent}%`);
      } else if (data.kind === "generating") {
        setStatus("Predicting in your browser…");
      } else if (data.kind === "result" || data.kind === "error") {
        setBusy(false);
        setStatus("");
        if (data.kind === "error") {
          abandonWorker();
          setError(data.message);
        } else {
          setResult({ prompt: request.prompt, prediction: data.prediction, isSample: false });
          setRun(previous => previous + 1);
        }
      }
    };

    function workerFailed() {
      if (worker.current !== engine) return;
      abandonWorker();
      if (branch !== undefined) {
        finishBranch({ error: "The local model stopped. Predict two words again to restart it." });
      } else {
        setBusy(false);
        setStatus("");
        setError("The browser could not start local inference. Try again to run the animation.");
      }
    }
    engine.onerror = workerFailed;
    try {
      engine.postMessage(request);
    } catch {
      workerFailed();
    }
  }

  const candidates = result?.prediction.candidates;

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
          <p>Predict two words on your device. Then move one weight across an INT4 rounding boundary and see whether the leading word changes.</p>
        </div>
        <form className="lesson-reference next-word-prompt" onSubmit={predict}>
          <label htmlFor="next-word-input">Your text</label>
          <textarea id="next-word-input" rows={2} maxLength={MAX_PROMPT_CHARS} value={prompt}
            onChange={event => setPrompt(event.target.value)} placeholder="Start a sentence…" />
          <div className="next-word-actions">
            <button type="submit" className="next-word-primary" disabled={busy || following !== null || !prompt.trim()}>Predict two words <ArrowRight size={16} /></button>
            <button type="button" className="next-word-secondary" onClick={() => setPrompt(examplePrompt)}>Use example prompt</button>
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
          <p id="next-word-download-details">DistilGPT2 predicts the next two words in your browser. The animation uses the model’s scores for two complete word paths to simulate how rounding one weight could change the leading word. The first run downloads about 85 MB of model weights plus the runtime.</p>
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
          {candidates
            ? <WeightToWord key={run} candidates={candidates} isSample={result.isSample} />
            : <p role="status">The model didn’t find two distinct complete words to compare. Try a different prompt or use the near-tie example.</p>}
          {candidates && <section className="lesson-reference next-word-stage next-word-branches" aria-labelledby="next-word-branches">
            <h2 id="next-word-branches">What happens after each word?</h2>
            <p>Choose a word to add it to the original prompt. DistilGPT2 predicts up to two more words from that context. Run both branches to compare the continuations.</p>
            <div className="next-word-math">
              {branchIndices.map(index => {
                const candidate = candidates[index];
                const branch = branches[index];
                return <div key={index} className="next-word-branch">
                  <button type="button" className="next-word-secondary" disabled={!worker.current || following !== null || Boolean(branch && "continuation" in branch)}
                    onClick={() => followWord(index, candidate)}>
                    <span>Follow “{candidate.word}”</span><ArrowRight size={16} />
                  </button>
                  <div aria-live="polite" aria-atomic="true">
                    {following === index && <p className="next-word-status">Predicting what follows “{candidate.word}”…</p>}
                    {branch && "error" in branch && <p className="next-word-error">{branch.error}</p>}
                    {branch && "continuation" in branch && (branch.continuation
                      ? <p className="next-word-branch-result">{candidate.word}<strong>{branch.continuation}</strong></p>
                      : <p>The model didn’t produce another complete word.</p>)}
                  </div>
                </div>;
              })}
            </div>
          </section>}
          {!result.isSample && <div className="next-word-actions">
            <button type="button" className="next-word-secondary" onClick={showExample} disabled={following !== null}>Show near-tie example</button>
          </div>}
        </>}
      </div>
    </main>
  );
}

function WeightToWord({ candidates, isSample }: { candidates: [WordPath, WordPath]; isSample: boolean }) {
  const [weight, setWeight] = useState(WEIGHT_START);
  const playback = useLessonPlayback(phases.length, true, 1, phaseSeconds);
  const { chapter, chapterTime, playing, reducedMotion, seek, time, duration } = playback;
  const progress = chapter === 0 ? 0 : chapter === 2 || reducedMotion ? 1 : Math.min(1, chapterTime / (phaseSeconds * 0.7));
  const start = predictionTrace(weight, candidates, 0);
  const end = predictionTrace(weight, candidates, 1);
  const frame = predictionTrace(weight, candidates, progress);
  const winner = (probability: number) => Math.abs(probability - 0.5) < 1e-8 ? "Tie" : candidates[probability > 0.5 ? 0 : 1].word;
  const firstWinner = winner(start.firstProbability);
  const finalWinner = winner(end.firstProbability);
  const readout = [
    { value: number(weight), name: "original weight", why: "Each word has a starting score. The simulation follows one weight’s contribution to the first word’s score.",
      next: "Divide the weight by the INT4 scale and round to the nearest code." },
    { value: number(frame.movingWeight), name: progress < 1 ? "weight in motion" : "reconstructed weight",
      why: `round(${number(weight)} / ${WEIGHT_SCALE}) = ${end.code}. Multiplying the code by the scale gives ${number(end.restored)}.`,
      next: "Apply the change in the weight’s contribution and compare the two scores." },
    { value: finalWinner, name: finalWinner === "Tie" ? "equal simulated scores" : "leading word after rounding",
      why: firstWinner === finalWinner ? "The score change kept the same ranking." : `The score change moved the lead from ${firstWinner} to ${finalWinner}.`,
      next: "Drag the weight across the midpoint and test where the outcome changes." },
  ][chapter]!;

  return (
    <section className="lesson-reference next-word-stage" aria-labelledby="next-word-trace">
      <h2 id="next-word-trace">Follow one weight into the next word</h2>
      <p>{isSample
        ? "This example uses two hand-picked word scores."
        : "The first row follows the model’s continuation. The second follows an alternate branch until it reaches a different complete word."} The bars show the relative chances of these two paths.</p>
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
          <h3>Two possible next words</h3>
          {[frame.firstProbability, 1 - frame.firstProbability].map((probability, index) => (
            <div key={index} className="next-word-score" data-leading={probability > 0.5 + 1e-8}>
              <div><strong>{candidates[index]!.word}</strong><span>{percent(probability)}</span></div>
              <div className="next-word-bar" aria-hidden="true"><span style={{ width: percent(probability) }} /></div>
              <small>Comparison score {number(index === 0 ? frame.firstScore : frame.secondScore)}</small>
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
      <details className="next-word-explanation">
        <summary>How the word scores work</summary>
        <p>A word can span several model tokens. Add each token’s log probability through the word’s end, including any spaces or punctuation generated before it. This gives the score for the specific token path shown. The bars are normalized across these two paths and sum to 100%.</p>
        {!isSample && <dl>
          {candidates.map(path => <div key={path.word}>
            <dt>{path.word}</dt>
            <dd>{path.pieces.map(piece => `${piece.text} (${number(piece.logProbability)})`).join(" + ")} = {number(path.score)}</dd>
          </div>)}
        </dl>}
        <p>S is the spacing between INT4 values. x is the input activation used in the simulation. The weight change multiplied by x shifts the first word’s comparison score. The displayed continuation comes from DistilGPT2.</p>
        {!isSample && <p>If the model’s first attempt ends before reaching a complete word, the demo tries another model-scored path.</p>}
      </details>
    </section>
  );
}
