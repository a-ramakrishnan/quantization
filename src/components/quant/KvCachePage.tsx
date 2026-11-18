import { useState } from "react";
import { ArrowLeft, CheckCircle2, CircleAlert } from "lucide-react";
import {
  ATTENTION_LAYERS, estimateQwenMemory, GIB, GPU_PRESETS, KV_OPTIONS, QWEN,
  RECURRENT_LAYERS, WEIGHT_OPTIONS,
} from "@/utils/qwen-memory";
import "./kv-cache.css";

const contexts = [8_192, 32_768, 131_072, QWEN.maxContext];
const rowKinds = ["Weights", "Attention KV", "DeltaNet state", "Runtime reserve"];
const formatCount = new Intl.NumberFormat("en-US");
const formatSize = (bytes: number) => Math.abs(bytes) >= GIB
  ? `${(bytes / GIB).toFixed(2)} GiB`
  : `${(bytes / 2 ** 20).toFixed(2)} MiB`;

function NumberField({ label, value, setValue, min, max, step = 1 }: {
  label: string;
  value: string;
  setValue: (value: string) => void;
  min: number;
  max: number;
  step?: number;
}) {
  return (
    <label className="kv-field">
      <span>{label}</span>
      <input type="number" value={value} min={min} max={max} step={step}
        onChange={event => setValue(event.target.value)}
        onBlur={() => setValue(String(readNumber(value, min, max, step === 1)))} />
    </label>
  );
}

function readNumber(value: string, min: number, max: number, integer = true) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(max, Math.max(min, integer ? Math.floor(parsed) : parsed)) : min;
}

export function KvCachePage() {
  const [gpu, setGpu] = useState<keyof typeof GPU_PRESETS>("a10080");
  const [weight, setWeight] = useState<keyof typeof WEIGHT_OPTIONS>("bf16");
  const [kv, setKv] = useState<keyof typeof KV_OPTIONS>("bf16");
  const [gpuGiB, setGpuGiB] = useState("80");
  const [context, setContext] = useState("8192");
  const [batch, setBatch] = useState("16");
  const [gpuCount, setGpuCount] = useState("1");
  const [utilization, setUtilization] = useState("90");
  const [overhead, setOverhead] = useState("2");
  const inputs = {
    weight, kv,
    context: readNumber(context, 1, QWEN.maxContext),
    batch: readNumber(batch, 1, 1024),
    gpuCount: readNumber(gpuCount, 1, 32),
    gpuGiB: readNumber(gpuGiB, 1, 80, false),
    utilizationPct: readNumber(utilization, 1, 100),
    overheadGiB: readNumber(overhead, 0, 80, false),
  };
  const result = estimateQwenMemory(inputs);
  const segments = [
    result.weightBytes,
    result.kvPerSequence * result.sequencesPerGpu,
    result.statePerSequence * result.sequencesPerGpu,
    result.overhead,
  ];
  const chartLimit = Math.max(result.capacity, result.peak);
  const rows = [
    ["Attention KV per token", `${result.kvPerToken / 1024} KiB`],
    ["Attention KV per sequence", formatSize(result.kvPerSequence)],
    ["Attention KV for batch", formatSize(result.kvForBatch)],
    ["DeltaNet state per sequence", formatSize(result.statePerSequence)],
    ["Weights per GPU", formatSize(result.weightBytes)],
    ["Max concurrent sequences by memory", formatCount.format(result.maxSequences)],
  ];

  function selectGpu(next: keyof typeof GPU_PRESETS) {
    setGpu(next);
    setGpuGiB(String(GPU_PRESETS[next].nominalGiB));
  }

  return (
    <main className="quant-lesson kv-cache-page">
      <header className="lesson-header">
        <div className="lesson-shell lesson-header-inner">
          <a className="lesson-brand" href="./"><ArrowLeft size={17} /> Back to lesson</a>
          <a className="lesson-lab" href="?view=compare">Weight comparison</a>
        </div>
      </header>
      <div className="lesson-shell">
        <div className="kv-intro">
          <p>Qwen3.5-9B</p>
          <h1>KV cache &amp; GPU memory</h1>
          <p>Estimate how context length, concurrent sequences, and storage precision change VRAM use on an NVIDIA A100 or L4.</p>
        </div>

        <div className="kv-grid">
          <section className="lesson-reference kv-panel" aria-labelledby="kv-settings-title">
            <h2 id="kv-settings-title">Configuration</h2>
            <div className="kv-fields">
              <label className="kv-field kv-full">
                <span>GPU preset</span>
                <select value={gpu} onChange={event => selectGpu(event.target.value as keyof typeof GPU_PRESETS)}>
                  {Object.entries(GPU_PRESETS).map(([id, preset]) => <option key={id} value={id}>{id === "l4" ? "NVIDIA L4 24 GB" : preset.label}</option>)}
                </select>
              </label>
              <label className="kv-field">
                <span>Weight storage</span>
                <select value={weight} onChange={event => setWeight(event.target.value as keyof typeof WEIGHT_OPTIONS)}>
                  {Object.entries(WEIGHT_OPTIONS).map(([id, option]) => <option key={id} value={id}>{option.label}</option>)}
                </select>
              </label>
              <label className="kv-field">
                <span>KV-cache storage</span>
                <select value={kv} onChange={event => setKv(event.target.value as keyof typeof KV_OPTIONS)}>
                  {Object.entries(KV_OPTIONS).map(([id, option]) => <option key={id} value={id}>{option.label}</option>)}
                </select>
              </label>
              <NumberField label="Context per sequence (tokens)" value={context} setValue={setContext} min={1} max={QWEN.maxContext} />
              <NumberField label="Concurrent sequences" value={batch} setValue={setBatch} min={1} max={1024} />
              <div className="experiment-buttons kv-full kv-contexts" role="group" aria-label="Context presets">
                {contexts.map(tokens => (
                  <button type="button" key={tokens} aria-pressed={inputs.context === tokens} onClick={() => setContext(String(tokens))}>{tokens / 1024}K</button>
                ))}
              </div>
              <NumberField label="VRAM per GPU (GiB)" value={gpuGiB} setValue={setGpuGiB} min={1} max={80} step={0.1} />
              <NumberField label="GPU replicas" value={gpuCount} setValue={setGpuCount} min={1} max={32} />
              <NumberField label="VRAM budget (%)" value={utilization} setValue={setUtilization} min={1} max={100} />
              <NumberField label="Runtime reserve (GiB/GPU)" value={overhead} setValue={setOverhead} min={0} max={80} step={0.1} />
            </div>
            <p>Each replica holds a full copy of the weights. The batch is spread evenly across replicas.</p>
          </section>

          <section className="lesson-reference kv-panel" aria-labelledby="kv-results-title">
            <div className="kv-results-heading">
              <h2 id="kv-results-title">Peak memory per GPU</h2>
              <span className="kv-verdict" data-fits={result.fits} role="status">
                {result.fits ? <CheckCircle2 size={15} /> : <CircleAlert size={15} />}
                {result.fits ? "Fits estimate" : "Over budget"}
              </span>
            </div>
            <p className="kv-peak"><strong>{formatSize(result.peak)}</strong> / {formatSize(result.budget)} budget</p>
            <p>{formatCount.format(result.sequencesPerGpu)} of {formatCount.format(inputs.batch)} sequences on the busiest GPU at {formatCount.format(inputs.context)} tokens each.</p>
            <div className="kv-stack" aria-hidden="true">
              {segments.map((bytes, i) => <div key={rowKinds[i]} className={`kv-segment kv-segment-${i}`} style={{ width: `${100 * bytes / chartLimit}%` }} />)}
              <span className="kv-budget-mark" style={{ left: `${100 * result.budget / chartLimit}%` }} />
            </div>
            <dl className="kv-legend">
              {segments.map((bytes, i) => (
                <div key={rowKinds[i]}>
                  <dt><i className={`kv-segment-${i}`} />{rowKinds[i]}</dt><dd>{formatSize(bytes)}</dd>
                </div>
              ))}
            </dl>
            <dl className="kv-summary" aria-label="KV cache and capacity calculation">
              {rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
            </dl>
          </section>
        </div>

        <section className="lesson-reference kv-panel kv-architecture" aria-labelledby="kv-architecture-title">
          <h2 id="kv-architecture-title">Why Qwen’s cache grows differently</h2>
          <div className="kv-layer-stack" role="img" aria-label="Eight repeated groups of three DeltaNet layers and one full-attention layer">
            {Array.from({ length: ATTENTION_LAYERS }, (_, group) => (
              <div key={group}>{Array.from({ length: QWEN.fullAttentionEvery }, (_, layer) => (
                <i key={layer} className={layer === QWEN.fullAttentionEvery - 1 ? "kv-segment-1" : "kv-segment-2"} />
              ))}</div>
            ))}
          </div>
          <div className="kv-explain">
            <div><h3><i className="kv-segment-1" />{ATTENTION_LAYERS} full-attention layers</h3>
              <p className="kv-equation">2 × {ATTENTION_LAYERS} layers × {QWEN.kvHeads} KV heads × {QWEN.headDim} values × {KV_OPTIONS[kv].bits / 8} B = <strong>{result.kvPerToken / 1024} KiB/token</strong></p>
              <p>The factor of two counts keys and values. This memory grows with context length and active sequences.</p>
            </div>
            <div><h3><i className="kv-segment-2" />{RECURRENT_LAYERS} Gated DeltaNet layers</h3>
              <p>Each sequence holds {formatSize(result.statePerSequence)} of fixed state under this estimate, using BF16 convolution buffers and FP32 recurrent state. KV-cache precision applies to the full-attention layers.</p>
            </div>
          </div>
        </section>

        <details className="lesson-reference kv-panel kv-assumptions">
          <summary>Assumptions and sources</summary>
          <ul>
            <li>Weight storage uses 9 billion parameters. Group-128 adds one FP16 scale per group. Actual checkpoints can mix precisions across tensors.</li>
            <li>KV numbers are raw payload sizes. Cache scales, block padding, prefix sharing, and speculative decoding can change physical allocations. The DeltaNet estimate holds one current state per sequence, using the vLLM state shapes.</li>
            <li>The GPU presets use nominal capacity. Enter measured VRAM if available. The reserve represents runtime allocations, including activation workspace. W8A16 and W8A8 describe compute precision. With the same weight and KV storage, they have the same persistent memory estimate.</li>
            <li>A fit is a memory estimate. Cache format support and performance depend on the runtime and hardware. NVIDIA lists native FP8 Tensor Core throughput for L4. A100’s FP8 cache estimate assumes a runtime path that supports that storage format.</li>
          </ul>
          <p className="kv-sources">
            <a href="https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json">Qwen model config</a>
            <a href="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py">vLLM state shapes</a>
            <a href="https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/">vLLM KV precision</a>
            <a href="https://www.nvidia.com/en-us/data-center/a100/">NVIDIA A100</a>
            <a href="https://www.nvidia.com/en-us/data-center/l4/">NVIDIA L4</a>
          </p>
        </details>
      </div>
    </main>
  );
}
