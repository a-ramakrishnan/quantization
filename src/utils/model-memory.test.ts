import { describe, expect, it } from "vitest";
import { estimateModelMemory, GIB } from "./model-memory";

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
});
