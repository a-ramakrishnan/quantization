import { describe, expect, it } from "vitest";
import { estimateModelMemory, GIB, planGpuCapacity, tensorParallelOptions } from "./model-memory";

const base = {
  model: "qwen35", weight: "bf16", kv: "bf16", context: 8_192, batch: 16,
  gpuCount: 1, gpuGiB: 80, utilizationPct: 90, overheadGiB: 2,
} as const;

describe("model memory estimator", () => {
  it("counts full-attention KV and one fixed DeltaNet state per sequence", () => {
    const result = estimateModelMemory(base);
    expect(result.kvPerToken).toBe(32 * 1024);
    expect(result.kvPerSequence).toBe(GIB / 4);
    expect(result.kvForBatch).toBe(4 * GIB);
    expect(result.statePerSequence).toBe(51_511_296);
    expect(result.maxSequences).toBe(178);
  });

  it("changes growing KV storage independently from weights and DeltaNet state", () => {
    const result = estimateModelMemory({ ...base, kv: "fp8", context: 16_384 });
    expect(result.kvPerToken).toBe(16 * 1024);
    expect(result.kvForBatch).toBe(4 * GIB);
    expect(result.statePerSequence).toBe(51_511_296);
    expect(result.weightBytes).toBe(18e9);
    expect(estimateModelMemory({ ...base, kv: "int4" }).kvPerToken).toBe(8 * 1024);
  });

  it.each([
    ["qwen25", 56, 7.61e9],
    ["qwen3", 256, 32.8e9],
    ["llama8", 128, 8e9],
    ["llama70", 320, 70e9],
  ] as const)("estimates full-attention KV and weights for %s", (model, kibPerToken, parameters) => {
    const result = estimateModelMemory({ ...base, model });
    expect(result.kvPerToken).toBe(kibPerToken * 1024);
    expect(result.statePerSequence).toBe(0);
    expect(result.weightBytes).toBe(2 * parameters);
  });

  it("reports zero capacity when the weights alone exceed the budget", () => {
    const result = estimateModelMemory({ ...base, model: "llama70", gpuGiB: 80, gpuCount: 2 });
    expect(result.maxSequences).toBe(0);
    expect(result.fits).toBe(false);
  });

  it("keeps a full weight copy per GPU and rounds peak batch size up", () => {
    const single = estimateModelMemory({ ...base, gpuGiB: 24, batch: 17 });
    const replicas = estimateModelMemory({ ...base, gpuGiB: 24, batch: 17, gpuCount: 2 });
    expect(single.fits).toBe(false);
    expect(replicas.fits).toBe(true);
    expect(replicas.sequencesPerGpu).toBe(9);
    expect(replicas.maxSequences).toBe(18);
    expect(replicas.weightBytes).toBe(single.weightBytes);
  });

  it.each([["glm47", 368], ["minimax25", 248], ["glm47flash", 52.875], ["kimi26", 68.625]] as const)(
    "matches the BF16 growing KV payload for %s", (model, kibPerToken) => {
      expect(estimateModelMemory({ ...base, model }).kvPerToken).toBe(kibPerToken * 1024);
    },
  );

  it("caps Olmo and Inkling local cache at their respective windows", () => {
    const olmo = estimateModelMemory({ ...base, model: "olmo3" });
    expect(olmo.kvPerSequence).toBe(1.25 * GIB);
    expect(estimateModelMemory({ ...base, model: "olmo3", context: 2_048 }).kvPerSequence).toBe(GIB / 2);
    expect(estimateModelMemory({ ...base, model: "inkling2" }).kvPerSequence).toBe(0.40625 * GIB);
  });

  it("shards GQA by KV head and replicates MLA within a tensor-parallel group", () => {
    const llama = estimateModelMemory({ ...base, model: "llama70", tensorParallel: 16 });
    expect(llama.weightPerGpu).toBe(140e9 / 16);
    expect(llama.kvPerGpuSequence).toBe(llama.kvPerSequence / 8);
    const kimi = estimateModelMemory({ ...base, model: "kimi26", tensorParallel: 8 });
    expect(kimi.kvPerGpuSequence).toBe(kimi.kvPerSequence);
    expect(tensorParallelOptions("glm47flash")).toEqual([1, 2, 4]);
  });

  it("finds the smallest feasible GPU count and balances requests across replicas", () => {
    const workload = { ...base, model: "llama70", batch: 64 } as const;
    const plan = planGpuCapacity(workload, 80);
    expect(plan).toMatchObject({ totalGpus: 8, tensorParallel: 4, replicas: 2 });
    expect(estimateModelMemory({ ...workload, gpuCount: 2, tensorParallel: 4 }).fits).toBe(true);
    expect(estimateModelMemory({ ...workload, gpuCount: 1, tensorParallel: 4 }).fits).toBe(false);
    expect(planGpuCapacity({ ...base, model: "kimi26" }, 24)).toBeUndefined();
  });

  it("includes one byte per block for NVFP4 and MXFP8", () => {
    expect(estimateModelMemory({ ...base, weight: "nvfp4" }).weightBytes).toBe(9e9 * (0.5 + 1 / 16));
    expect(estimateModelMemory({ ...base, weight: "mxfp8" }).weightBytes).toBe(9e9 * (1 + 1 / 32));
  });
});
