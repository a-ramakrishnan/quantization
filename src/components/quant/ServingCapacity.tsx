import { useState } from "react";
import { estimateModelMemory, GPU_PRESETS, MODEL_PRESETS, planGpuCapacity } from "@/utils/model-memory";
import type { CapacityInputs, GpuId, ModelId } from "@/utils/model-memory";
import { SERVING_BENCHMARKS } from "@/utils/serving-benchmarks";

const count = new Intl.NumberFormat("en-US");
const metric = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
type Plan = NonNullable<ReturnType<typeof planGpuCapacity>>;

export function ServingCapacity({ inputs, formatSize, apply }: {
  inputs: CapacityInputs;
  formatSize: (bytes: number) => string;
  apply: (gpu: GpuId, plan: Plan, context: number) => void;
}) {
  const max = MODEL_PRESETS[inputs.model].maxContext;
  const contexts = [...new Set([inputs.context, 4_096, 8_192, 32_768, 131_072, max])]
    .filter(value => value <= max).sort((a, b) => a - b);
  const gpus = Object.keys(GPU_PRESETS) as GpuId[];
  const scenarios = contexts.map(context => {
    const workload = { ...inputs, context };
    const kvPerSequence = estimateModelMemory({ ...workload, gpuGiB: 1, gpuCount: 1 }).kvPerSequence;
    const plans = gpus.map(gpu => planGpuCapacity(workload, GPU_PRESETS[gpu].nominalGiB));
    const fewest = Math.min(...plans.map(plan => plan?.totalGpus ?? Infinity));
    return { context, kvPerSequence, plans, fewest };
  });

  return (
    <section className="lesson-reference kv-panel kv-serving" aria-labelledby="kv-serving-title">
      <h2 id="kv-serving-title">GPU capacity planner</h2>
      <p>Compare the GPUs needed for {count.format(inputs.batch)} active requests. Each cell uses the selected precisions and gives the smallest GPU count for that hardware. Highlighted cells use the fewest GPUs in their context row.</p>
      <div className="kv-table-wrap" tabIndex={0} role="region" aria-label="Required GPUs by context length">
        <table className="kv-comparison-table kv-planner-table">
          <thead><tr><th scope="col">Context</th><th scope="col">KV / request</th>
            {gpus.map(gpu => <th scope="col" key={gpu}>{GPU_PRESETS[gpu].label.replace("NVIDIA ", "")}<small>{GPU_PRESETS[gpu].nominalGiB} GB</small></th>)}
          </tr></thead>
          <tbody>{scenarios.map(({ context, kvPerSequence, plans, fewest }) => (
            <tr key={context} data-current={context === inputs.context}>
              <th scope="row">{count.format(context)} tokens{context === inputs.context && <small>Selected context</small>}</th>
              <td>{formatSize(kvPerSequence)}</td>
              {plans.map((plan, index) => {
                const gpu = gpus[index];
                return (
                  <td key={gpu} data-best={plan?.totalGpus === fewest}>
                    {plan ? <button type="button" onClick={() => apply(gpu, plan, context)}
                      aria-label={`Apply ${plan.totalGpus} ${GPU_PRESETS[gpu].label} ${GPU_PRESETS[gpu].nominalGiB} GB GPUs at ${count.format(context)} tokens`}>
                      <strong>{count.format(plan.totalGpus)} GPU{plan.totalGpus === 1 ? "" : "s"}</strong>
                      <small>TP {plan.tensorParallel} × {count.format(plan.replicas)} {plan.replicas === 1 ? "replica" : "replicas"}</small>
                      <small>{formatSize(plan.kvPerGpu)} KV / GPU</small>
                    </button> : <small>No fit</small>}
                  </td>
                );
              })}
            </tr>
          ))}</tbody>
        </table>
      </div>
      <p>Choose a cell to load its configuration. TP is the number of GPUs sharing each model copy. A replica takes a share of active requests. Counts are based on memory and assume compatible hardware, interconnects and serving kernels. TP 16 may require multiple nodes.</p>
    </section>
  );
}

export function ServingBenchmarks({ model, context, batch, selectModel }: {
  model: ModelId;
  context: number;
  batch: number;
  selectModel: (model: ModelId) => void;
}) {
  const benchmark = SERVING_BENCHMARKS[model];
  const [scenarioIndex, setScenarioIndex] = useState("");
  const [target, setTarget] = useState("");
  const closest = benchmark?.scenarios.reduce((best, scenario, index, scenarios) => {
    const distance = (item: typeof scenario) =>
      Math.abs(Math.log((item.prompt + item.output) / context)) + Math.abs(Math.log(item.concurrency / batch));
    return distance(scenario) < distance(scenarios[best]) ? index : best;
  }, 0) ?? 0;
  const scenario = benchmark?.scenarios[scenarioIndex === "" ? closest : Number(scenarioIndex)];
  const targetMs = Number(target) > 0 ? Number(target) : Infinity;
  const eligible = scenario?.measurements.filter(row => row.ttftMs <= targetMs)
    .sort((a, b) => b.outputTokensPerSecond - a.outputTokensPerSecond);
  const best = eligible?.[0];

  return (
    <section className="lesson-reference kv-panel kv-serving" aria-labelledby="kv-bench-title">
      <h2 id="kv-bench-title">Measured serving performance</h2>
      <p>TTFT measures the time until the first output token. Output throughput counts generated tokens per second across the tested server, and inter-token latency measures the time between streamed tokens.</p>
      {benchmark && scenario ? <>
        <div className="kv-fields">
          <label className="kv-field"><span>Benchmark workload</span>
            <select value={scenarioIndex} onChange={event => setScenarioIndex(event.target.value)}>
              <option value="">Closest measured workload</option>
              {benchmark.scenarios.map((row, index) => <option key={index} value={index}>
                {count.format(row.prompt)} in / {count.format(row.output)} out · {row.concurrency} concurrent
              </option>)}
            </select>
          </label>
          <label className="kv-field"><span>Mean TTFT target (ms, optional)</span>
            <input type="number" min="1" step="1" value={target} onChange={event => setTarget(event.target.value)} />
          </label>
        </div>
        <p><strong>{count.format(scenario.prompt)} input + {count.format(scenario.output)} output tokens · {scenario.concurrency} concurrent requests.</strong> {benchmark.setup}.</p>
        <div className="kv-table-wrap" tabIndex={0} role="region" aria-label="Measured serving metrics">
          <table className="kv-comparison-table">
            <thead><tr><th scope="col">GPU</th><th scope="col">Mean TTFT</th><th scope="col">Output tokens/s</th><th scope="col">Mean inter-token latency</th></tr></thead>
            <tbody>{scenario.measurements.map(row => (
              <tr key={row.gpu} data-current={row === best && scenario.measurements.length > 1}>
                <th scope="row">{GPU_PRESETS[row.gpu].label}</th>
                <td>{metric.format(row.ttftMs)} ms</td>
                <td>{metric.format(row.outputTokensPerSecond)}</td>
                <td>{metric.format(row.itlMs)} ms</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        {Number.isFinite(targetMs) && <p role="status">{best
          ? `${GPU_PRESETS[best.gpu].label} has the highest measured output throughput within this mean TTFT target.`
          : "The measured configurations exceed this mean TTFT target."}</p>}
        <p className="kv-sources"><a href={benchmark.source}>Benchmark methodology and results</a></p>
        <p>These measurements apply to the listed workload and server configuration. Changing the capacity inputs leaves the published measurements intact.</p>
      </> : <>
        <p>Published measurements for precise GPU configurations are available here for the following presets.</p>
        <div className="experiment-buttons" role="group" aria-label="Presets with benchmark results">
          {(Object.keys(SERVING_BENCHMARKS) as ModelId[]).map(id => (
            <button key={id} type="button" onClick={() => selectModel(id)}>{MODEL_PRESETS[id].label}</button>
          ))}
        </div>
      </>}
    </section>
  );
}
