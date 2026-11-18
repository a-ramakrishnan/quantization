import { storageGB } from "./quantization";

export const GIB = 2 ** 30;
export const QWEN = {
  parameters: 9_000_000_000,
  layers: 32,
  fullAttentionEvery: 4,
  kvHeads: 4,
  headDim: 256,
  maxContext: 262_144,
  linearKeyHeads: 16,
  linearValueHeads: 32,
  linearKeyDim: 128,
  linearValueDim: 128,
  convolutionWidth: 4,
} as const;

export const ATTENTION_LAYERS = QWEN.layers / QWEN.fullAttentionEvery;
export const RECURRENT_LAYERS = QWEN.layers - ATTENTION_LAYERS;
export const GPU_PRESETS = {
  l4: { label: "NVIDIA L4", nominalGiB: 24 },
  a10040: { label: "NVIDIA A100 40 GB", nominalGiB: 40 },
  a10080: { label: "NVIDIA A100 80 GB", nominalGiB: 80 },
} as const;
export const WEIGHT_OPTIONS = {
  bf16: { label: "BF16 / FP16", bits: 16 },
  fp8: { label: "FP8 · raw payload", bits: 8 },
  int8: { label: "INT8 · Group-128", bits: 8, groupSize: 128 },
  int4: { label: "INT4 · Group-128", bits: 4, groupSize: 128 },
} as const;
export const KV_OPTIONS = {
  bf16: { label: "BF16 / FP16", bits: 16 },
  fp8: { label: "FP8 / INT8", bits: 8 },
  int4: { label: "Packed INT4", bits: 4 },
} as const;

type MemoryInputs = {
  weight: keyof typeof WEIGHT_OPTIONS;
  kv: keyof typeof KV_OPTIONS;
  context: number;
  batch: number;
  gpuCount: number;
  gpuGiB: number;
  utilizationPct: number;
  overheadGiB: number;
};

export function estimateQwenMemory(input: MemoryInputs) {
  const weight = WEIGHT_OPTIONS[input.weight];
  const weightBytes = storageGB(QWEN.parameters, weight.bits, "groupSize" in weight ? weight.groupSize : undefined) * 1e9;
  const kvPerToken = 2 * ATTENTION_LAYERS * QWEN.kvHeads * QWEN.headDim * KV_OPTIONS[input.kv].bits / 8;
  const kvPerSequence = kvPerToken * input.context;

  // One BF16 convolution window and one FP32 recurrent state per DeltaNet layer.
  const convolutionChannels = 2 * QWEN.linearKeyHeads * QWEN.linearKeyDim + QWEN.linearValueHeads * QWEN.linearValueDim;
  const convolutionBytes = convolutionChannels * (QWEN.convolutionWidth - 1) * 2;
  const recurrentBytes = QWEN.linearValueHeads * QWEN.linearKeyDim * QWEN.linearValueDim * 4;
  const statePerSequence = RECURRENT_LAYERS * (convolutionBytes + recurrentBytes);

  const sequencesPerGpu = Math.ceil(input.batch / input.gpuCount);
  const capacity = input.gpuGiB * GIB;
  const budget = capacity * input.utilizationPct / 100;
  const overhead = input.overheadGiB * GIB;
  const peak = weightBytes + overhead + sequencesPerGpu * (kvPerSequence + statePerSequence);
  const maxSequencesPerGpu = Math.max(0, Math.floor((budget - weightBytes - overhead) / (kvPerSequence + statePerSequence)));

  return {
    weightBytes, kvPerToken, kvPerSequence, statePerSequence,
    sequencesPerGpu, capacity, budget, overhead, peak,
    kvForBatch: kvPerSequence * input.batch,
    maxSequencesPerGpu,
    maxSequences: input.gpuCount * maxSequencesPerGpu,
    fits: peak <= budget,
  };
}
