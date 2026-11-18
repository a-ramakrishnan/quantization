import { useRef, useState } from "react";
import { ArrowLeft, CheckCircle2, CircleAlert } from "lucide-react";
import {
  attentionKvBytesPerToken, estimateModelMemory, GIB, GPU_PRESETS, KV_OPTIONS, MODEL_PRESETS, tensorParallelOptions, WEIGHT_OPTIONS,
} from "@/utils/model-memory";
import type { GpuId, ModelId, ModelPreset } from "@/utils/model-memory";
import { ServingBenchmarks, ServingCapacity } from "./ServingCapacity";
import "./kv-cache.css";

const contextShortcuts = [8_192, 32_768, 131_072];
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
  const [model, setModel] = useState<ModelId>("qwen35");
  const selectedModel: ModelPreset = MODEL_PRESETS[model];
  const { attentionLayers, kvHeads, headDim, maxContext, recurrent, latent, sliding } = selectedModel;
  const contexts = [...contextShortcuts.filter(tokens => tokens < maxContext), maxContext];
  const [gpu, setGpu] = useState<GpuId>("a10080");
  const [tensorParallel, setTensorParallel] = useState(1);
  const resultsHeading = useRef<HTMLHeadingElement>(null);
  const [weight, setWeight] = useState<keyof typeof WEIGHT_OPTIONS>("bf16");
  const [kv, setKv] = useState<keyof typeof KV_OPTIONS>("bf16");
  const [gpuGiB, setGpuGiB] = useState("80");
  const [context, setContext] = useState("8192");
  const [batch, setBatch] = useState("16");
  const [gpuCount, setGpuCount] = useState("1");
  const [utilization, setUtilization] = useState("90");
  const [overhead, setOverhead] = useState("2");
  const inputs = {
    model, weight, kv, tensorParallel,
    context: readNumber(context, 1, maxContext),
    batch: readNumber(batch, 1, 1024),
    gpuCount: readNumber(gpuCount, 1, 1024),
    gpuGiB: readNumber(gpuGiB, 1, 512, false),
    utilizationPct: readNumber(utilization, 1, 100),
    overheadGiB: readNumber(overhead, 0, 512, false),
  };
  const result = estimateModelMemory(inputs);
  const segments = [
    result.weightPerGpu,
    result.kvPerGpuSequence * result.sequencesPerGpu,
    result.statePerSequence * result.sequencesPerGpu,
    result.overhead,
  ];
  const chartLimit = Math.max(result.capacity, result.peak);
  const rows = [
    ["Growing attention KV per token", `${result.kvPerToken / 1024} KiB`],
    ["Attention KV per sequence", formatSize(result.kvPerSequence)],
    ["Logical attention KV for batch", formatSize(result.kvForBatch)],
    ...(recurrent ? [["DeltaNet state per sequence", formatSize(result.statePerSequence)]] : []),
    ["Weights per GPU", formatSize(result.weightPerGpu)],
    ["Total GPUs configured", formatCount.format(inputs.gpuCount * tensorParallel)],
    ["Max concurrent sequences by memory", formatCount.format(result.maxSequences)],
  ];

  function selectModel(next: ModelId) {
    setModel(next);
    if (!tensorParallelOptions(next).includes(tensorParallel)) setTensorParallel(1);
    setContext(String(readNumber(context, 1, MODEL_PRESETS[next].maxContext)));
  }

  function selectGpu(next: GpuId) {
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
          <h1>KV cache &amp; GPU memory</h1>
          <p>Estimate VRAM and GPU counts across context lengths, then compare published time-to-first-token and throughput measurements.</p>
        </div>

        <div className="kv-grid">
          <section className="lesson-reference kv-panel" aria-labelledby="kv-settings-title">
            <h2 id="kv-settings-title">Configuration</h2>
            <div className="kv-fields">
              <label className="kv-field kv-full">
                <span>Model preset</span>
                <select value={model} onChange={event => selectModel(event.target.value as ModelId)}>
                  {Object.entries(MODEL_PRESETS).map(([id, preset]) => <option key={id} value={id}>{preset.label}</option>)}
                </select>
              </label>
              <label className="kv-field kv-full">
                <span>GPU preset</span>
                <select value={gpu} onChange={event => selectGpu(event.target.value as GpuId)}>
                  {Object.entries(GPU_PRESETS).map(([id, preset]) => <option key={id} value={id}>{preset.label} · {preset.nominalGiB} GB</option>)}
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
              <NumberField label="Cached tokens per request" value={context} setValue={setContext} min={1} max={maxContext} />
              <NumberField label="Concurrent users (active requests)" value={batch} setValue={setBatch} min={1} max={1024} />
              <div className="experiment-buttons kv-full kv-contexts" role="group" aria-label="Context presets">
                {contexts.map(tokens => (
                  <button type="button" key={tokens} aria-pressed={inputs.context === tokens} onClick={() => setContext(String(tokens))}>
                    {tokens % 1024 === 0 ? `${tokens / 1024}K` : formatCount.format(tokens)}
                  </button>
                ))}
              </div>
              <NumberField label="VRAM per GPU (GiB)" value={gpuGiB} setValue={setGpuGiB} min={1} max={512} step={0.1} />
              <NumberField label="Model replicas" value={gpuCount} setValue={setGpuCount} min={1} max={1024} />
              <label className="kv-field">
                <span>GPUs per replica (tensor parallelism)</span>
                <select value={tensorParallel} onChange={event => setTensorParallel(Number(event.target.value))}>
                  {tensorParallelOptions(model).map(size => <option key={size} value={size}>{size}</option>)}
                </select>
              </label>
              <NumberField label="VRAM budget (%)" value={utilization} setValue={setUtilization} min={1} max={100} />
              <NumberField label="Runtime reserve (GiB/GPU)" value={overhead} setValue={setOverhead} min={0} max={512} step={0.1} />
            </div>
            <p>Budget one active request per concurrent user. Cached tokens include the prompt and room for generated tokens. Each replica shares a copy of the weights across its GPUs.</p>
          </section>

          <section className="lesson-reference kv-panel" aria-labelledby="kv-results-title">
            <div className="kv-results-heading">
              <h2 id="kv-results-title" ref={resultsHeading} tabIndex={-1}>Peak memory per GPU</h2>
              <span className="kv-verdict" data-fits={result.fits} role="status">
                {result.fits ? <CheckCircle2 size={15} /> : <CircleAlert size={15} />}
                {result.fits ? "Fits estimate" : "Over budget"}
              </span>
            </div>
            <p className="kv-peak"><strong>{formatSize(result.peak)}</strong> / {formatSize(result.budget)} budget</p>
            <p>The busiest replica handles {formatCount.format(result.sequencesPerGpu)} of {formatCount.format(inputs.batch)} requests at {formatCount.format(inputs.context)} cached tokens each.</p>
            <div className="kv-stack" aria-hidden="true">
              {segments.map((bytes, i) => <div key={rowKinds[i]} className={`kv-segment kv-segment-${i}`} style={{ width: `${100 * bytes / chartLimit}%` }} />)}
              <span className="kv-budget-mark" style={{ left: `${100 * result.budget / chartLimit}%` }} />
            </div>
            <dl className="kv-legend">
              {segments.map((bytes, i) => (i !== 2 || recurrent) && (
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

        <ServingCapacity inputs={inputs} formatSize={formatSize} apply={(nextGpu, plan, nextContext) => {
          selectGpu(nextGpu);
          setTensorParallel(plan.tensorParallel);
          setGpuCount(String(plan.replicas));
          setContext(String(nextContext));
          resultsHeading.current?.focus();
        }} />
        <ServingBenchmarks key={model} model={model} context={inputs.context} batch={inputs.batch} selectModel={selectModel} />

        <section className="lesson-reference kv-panel kv-architecture" aria-labelledby="kv-architecture-title">
          <h2 id="kv-architecture-title">{selectedModel.label} cache layout</h2>
          <p>{formatCount.format(maxContext)}-token native context limit · {selectedModel.parameters / 1e9}B parameters in this estimate</p>
          {recurrent && (
            <div className="kv-layer-stack" role="img" aria-label={`${attentionLayers} groups of ${recurrent.interval - 1} DeltaNet layers and one full-attention layer`}>
              {Array.from({ length: attentionLayers }, (_, group) => (
                <div key={group}>{Array.from({ length: recurrent.interval }, (_, layer) => (
                  <i key={layer} className={layer === recurrent.interval - 1 ? "kv-segment-1" : "kv-segment-2"} />
                ))}</div>
              ))}
            </div>
          )}
          <div className="kv-explain">
            <div><h3><i className="kv-segment-1" />{attentionLayers} {latent ? "latent attention (MLA)" : "full-attention"} layers</h3>
              <p className="kv-equation">{latent
                ? `${attentionLayers} layers × (${latent.rank} latent + ${latent.rotaryDim} positional values)`
                : `2 × ${attentionLayers} layers × ${kvHeads} KV heads × ${headDim} values`} × {KV_OPTIONS[kv].bits / 8} B = <strong>{result.kvPerToken / 1024} KiB/token</strong></p>
              <p>{latent ? "The compressed cache stores one latent vector and one positional key vector per token per layer. Each tensor-parallel GPU holds this cache."
                : "The factor of two counts keys and values. This memory grows with context length and active requests."}</p>
            </div>
            {sliding && <div><h3><i className="kv-segment-1" />{sliding.layers} sliding-window layers</h3>
              <p className="kv-equation">2 × {sliding.layers} layers × {kvHeads} KV heads × {headDim} values × {KV_OPTIONS[kv].bits / 8} B × min({formatCount.format(inputs.context)}, {formatCount.format(sliding.window)}) = <strong>{formatSize(result.slidingKvPerSequence)}</strong></p>
              <p>These layers only cache the most recent {formatCount.format(sliding.window)} tokens per request in this estimate.</p>
            </div>}
            {recurrent && (
              <div><h3><i className="kv-segment-2" />{recurrent.layers} Gated DeltaNet layers</h3>
                <p>Each sequence holds {formatSize(result.statePerSequence)} of fixed state under this estimate, using BF16 convolution buffers and FP32 recurrent state. KV-cache precision applies to the full-attention layers.</p>
              </div>
            )}
          </div>
          {selectedModel.deploymentNote && <p>{selectedModel.deploymentNote}</p>}
        </section>

        <div className="kv-grid kv-comparisons">
          <section className="lesson-reference kv-panel" aria-labelledby="kv-model-compare-title">
            <h2 id="kv-model-compare-title">Compare model layouts</h2>
            <p>Growing KV per token uses {KV_OPTIONS[kv].label}. Windowed cache is shown in the context planner.</p>
            <div className="kv-table-wrap" tabIndex={0} role="region" aria-label="Model comparison table">
              <table className="kv-comparison-table">
                <thead><tr><th scope="col">Model</th><th scope="col">Growing cache layers</th><th scope="col">Context limit</th><th scope="col">Growing KV/token</th></tr></thead>
                <tbody>
                  {Object.entries(MODEL_PRESETS).map(([id, preset]) => (
                    <tr key={id} data-current={id === model}>
                      <th scope="row"><button type="button" onClick={() => selectModel(id as ModelId)} aria-pressed={id === model}>{preset.label}</button></th>
                      <td>{preset.attentionLayers} layers</td>
                      <td>{formatCount.format(preset.maxContext)}</td>
                      <td>{attentionKvBytesPerToken(preset, kv) / 1024} KiB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          <section className="lesson-reference kv-panel" aria-labelledby="kv-gpu-compare-title">
            <h2 id="kv-gpu-compare-title">Compare GPU capacity</h2>
            <p>Same model and workload, using each preset’s nominal VRAM.</p>
            <div className="kv-table-wrap" tabIndex={0} role="region" aria-label="GPU comparison table">
              <table className="kv-comparison-table">
                <thead><tr><th scope="col">GPU preset</th><th scope="col">Max sequences</th><th scope="col">Current batch</th></tr></thead>
                <tbody>
                  {Object.entries(GPU_PRESETS).map(([id, preset]) => {
                    const estimate = estimateModelMemory({ ...inputs, gpuGiB: preset.nominalGiB });
                    return (
                      <tr key={id} data-current={id === gpu && inputs.gpuGiB === preset.nominalGiB}>
                        <th scope="row"><button type="button" onClick={() => selectGpu(id as GpuId)} aria-pressed={id === gpu && inputs.gpuGiB === preset.nominalGiB}>{preset.label} · {preset.nominalGiB} GB</button></th>
                        <td>{formatCount.format(estimate.maxSequences)}</td>
                        <td className="kv-table-verdict" data-fits={estimate.fits}>{estimate.fits ? "Fits" : "Over budget"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </div>

        <details className="lesson-reference kv-panel kv-assumptions">
          <summary>Assumptions and sources</summary>
          <ul>
            <li>Weight storage applies one format to the estimated total parameter count, including all experts. Group-128 adds one FP16 scale per group. MXFP8 and NVFP4 include their 8-bit block scales and omit tensor-scale and padding overhead. Actual checkpoints can mix precisions across tensors.</li>
            <li>Context caps use native settings. The Qwen2.5-7B and Qwen3-32B presets use a 32,768-token cap. Longer YaRN deployments have their own runtime configuration.</li>
            <li>KV numbers are raw payload sizes. Cache scales, block padding, prefix sharing, and speculative decoding can change physical allocations. The DeltaNet estimate holds one current state per sequence, using the vLLM state shapes.</li>
            <li>Tensor parallelism evenly divides weight payloads. Grouped-query KV shards by KV head and replicates when TP exceeds the number of KV heads. Compressed MLA and fixed DeltaNet state are modeled as replicated on each GPU in a model replica. The planner compares TP 1–16 where the query heads divide evenly.</li>
            <li>The GPU presets use nominal capacity. Enter measured VRAM if available. The reserve represents runtime allocations, including activation workspace. W8A16 and W8A8 describe compute precision. With the same weight and KV storage, they have the same persistent memory estimate.</li>
            <li>A fit is a memory estimate. Cache format support and performance depend on the runtime and hardware. Each selected storage format assumes a compatible runtime path.</li>
          </ul>
          <p className="kv-sources">
            <a href={selectedModel.source}>{selectedModel.label} architecture</a>
            {selectedModel.card && <a href={selectedModel.card}>{selectedModel.label} model details</a>}
            {recurrent && <a href="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py">vLLM state shapes</a>}
            <a href="https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/">vLLM KV precision</a>
            <a href="https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/">vLLM cache sharding</a>
            <a href="https://docs.nvidia.com/cudnn/latest/operations/BlockScaling/">NVIDIA block scale formats</a>
            <a href={GPU_PRESETS[gpu].source}>{GPU_PRESETS[gpu].label} specifications</a>
          </p>
        </details>
      </div>
    </main>
  );
}
