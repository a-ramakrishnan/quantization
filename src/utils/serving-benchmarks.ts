import type { GpuId, ModelId } from "./model-memory";

type Measurement = { gpu: GpuId; ttftMs: number; outputTokensPerSecond: number; itlMs: number };
type Scenario = { prompt: number; output: number; concurrency: number; measurements: Measurement[] };
type Benchmark = { setup: string; source: string; scenarios: Scenario[] };

// Dell Appendix A reports the paired results in H200, B200 order.
function dellScenario(prompt: number, output: number, concurrency: number,
  h200: [number, number, number], b200: [number, number, number]): Scenario {
  const [hRate, hTtft, hItl] = h200;
  const [bRate, bTtft, bItl] = b200;
  return { prompt, output, concurrency, measurements: [
    { gpu: "h200", ttftMs: hTtft, outputTokensPerSecond: hRate, itlMs: hItl },
    { gpu: "b200", ttftMs: bTtft, outputTokensPerSecond: bRate, itlMs: bItl },
  ] };
}

function b200Scenario(prompt: number, output: number, concurrency: number, ttftMs: number,
  outputTokensPerSecond: number, itlMs: number): Scenario {
  return { prompt, output, concurrency, measurements: [{ gpu: "b200", ttftMs, outputTokensPerSecond, itlMs }] };
}

export const SERVING_BENCHMARKS: Partial<Record<ModelId, Benchmark>> = {
  llama33: {
    setup: "8 GPUs per tested server · TP 8 · vLLM · FP8 weights",
    source: "https://www.dell.com/en-us/infohub/articles/hopper-to-blackwell-dell-poweredge-h200-vs-b200-llm-inference-performance-2#appendix-a-full-benchmark-data",
    scenarios: [
      dellScenario(1_024, 512, 1, [382, 107, 8], [535, 166, 5]),
      dellScenario(1_024, 512, 8, [1_993, 672, 11], [3_608, 454, 6]),
      dellScenario(1_024, 512, 32, [4_094, 2_100, 20], [11_085, 701, 7]),
      dellScenario(4_096, 1_024, 1, [464, 390, 10], [884, 203, 5]),
      dellScenario(4_096, 1_024, 8, [2_305, 2_300, 15], [5_741, 602, 6]),
      dellScenario(4_096, 1_024, 32, [3_829, 6_500, 36], [16_112, 1_500, 9]),
      dellScenario(16_384, 2_048, 1, [688, 1_800, 12], [1_446, 462, 6]),
      dellScenario(16_384, 2_048, 8, [2_341, 10_000, 26], [8_984, 1_700, 7]),
      dellScenario(16_384, 2_048, 32, [3_165, 21_000, 81], [19_237, 3_800, 13]),
    ],
  },
  minimax25: {
    setup: "8 B200 GPUs · TP 8 / EP 8 · SGLang 0.5.8 · FP8 checkpoint · random lengths, nominal token counts shown",
    source: "https://github.com/sgl-project/sgl-docs/blob/main/cookbook/llm/MiniMax/MiniMax-M2.5.md#51-speed-benchmark",
    scenarios: [
      b200Scenario(1_000, 1_000, 16, 305.44, 842.49, 15.65),
      b200Scenario(1_000, 1_000, 100, 391.06, 2_737.02, 32.61),
      b200Scenario(8_000, 1_000, 1, 131.5, 97.04, 10.01),
      b200Scenario(8_000, 1_000, 16, 282.56, 831.38, 16.24),
      b200Scenario(8_000, 1_000, 64, 692.26, 1_810.01, 31.28),
    ],
  },
  glm47flash: {
    setup: "1 B200 GPU · TP 1 · SGLang 0.5.7 · random lengths, nominal token counts shown",
    source: "https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-4.7-Flash.mdx#51-speed-benchmark",
    scenarios: [
      b200Scenario(1_000, 1_000, 1, 131.36, 108.37, 8.93),
      b200Scenario(1_000, 1_000, 16, 299.02, 773.83, 17.52),
      b200Scenario(8_000, 1_000, 1, 296.01, 72.42, 13.13),
      b200Scenario(8_000, 1_000, 16, 1_943.94, 465.10, 27.27),
      b200Scenario(8_000, 1_000, 64, 428.87, 1_017.93, 57.93),
    ],
  },
  kimi26: {
    setup: "8 B200 GPUs · TP 8 · vLLM · mixed INT4/BF16 checkpoint · FP8 KV · EAGLE-3",
    source: "https://lambda.com/inference-models/moonshotai/kimi-k2.6#benchmarking-kimi-k26",
    scenarios: [b200Scenario(8_192, 2_048, 32, 2_264.48, 1_408, 44.5)],
  },
};
