import { describe, expect, it } from "vitest";
import { estimateQwenMemory, GIB } from "./qwen-memory";

const base = {
  weight: "bf16", kv: "bf16", context: 8_192, batch: 16,
  gpuCount: 1, gpuGiB: 80, utilizationPct: 90, overheadGiB: 2,
} as const;

describe("Qwen3.5-9B memory estimator", () => {
  it("counts full-attention KV and one fixed DeltaNet state per sequence", () => {
    const result = estimateQwenMemory(base);
    expect(result.kvPerToken).toBe(32 * 1024);
    expect(result.kvPerSequence).toBe(GIB / 4);
    expect(result.kvForBatch).toBe(4 * GIB);
    expect(result.statePerSequence).toBe(51_511_296);
    expect(result.maxSequences).toBe(178);
  });

  it("changes growing KV storage independently from weights and DeltaNet state", () => {
    const result = estimateQwenMemory({ ...base, kv: "fp8", context: 16_384 });
    expect(result.kvPerToken).toBe(16 * 1024);
    expect(result.kvForBatch).toBe(4 * GIB);
    expect(result.statePerSequence).toBe(51_511_296);
    expect(result.weightBytes).toBe(18e9);
    expect(estimateQwenMemory({ ...base, kv: "int4" }).kvPerToken).toBe(8 * 1024);
  });

  it("keeps a full weight copy per GPU and rounds peak batch size up", () => {
    const single = estimateQwenMemory({ ...base, gpuGiB: 24, batch: 17 });
    const replicas = estimateQwenMemory({ ...base, gpuGiB: 24, batch: 17, gpuCount: 2 });
    expect(single.fits).toBe(false);
    expect(replicas.fits).toBe(true);
    expect(replicas.sequencesPerGpu).toBe(9);
    expect(replicas.maxSequences).toBe(18);
    expect(replicas.weightBytes).toBe(single.weightBytes);
  });
});
