import { bytesFor } from "./quantization";

export const GIB = 2 ** 30;
export type ModelPreset = {
  label: string;
  parameters: number;
  attentionLayers: number;
  kvHeads: number;
  headDim: number;
  maxContext: number;
  source: string;
  recurrent?: {
    layers: number;
    interval: number;
    keyHeads: number;
    valueHeads: number;
    keyDim: number;
    valueDim: number;
    convolutionWidth: number;
  };
};

export const MODEL_PRESETS = {
  qwen35: {
    label: "Qwen3.5-9B", parameters: 9_000_000_000, attentionLayers: 8,
    kvHeads: 4, headDim: 256, maxContext: 262_144,
    source: "https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json",
    recurrent: { layers: 24, interval: 4, keyHeads: 16, valueHeads: 32, keyDim: 128, valueDim: 128, convolutionWidth: 4 },
  },
  qwen25: {
    label: "Qwen2.5-7B Instruct", parameters: 7_610_000_000, attentionLayers: 28,
    kvHeads: 4, headDim: 128, maxContext: 32_768,
    source: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
  },
  qwen3: {
    label: "Qwen3-32B", parameters: 32_800_000_000, attentionLayers: 64,
    kvHeads: 8, headDim: 128, maxContext: 32_768,
    source: "https://huggingface.co/Qwen/Qwen3-32B",
  },
  llama8: {
    label: "Llama 3.1 8B", parameters: 8_000_000_000, attentionLayers: 32,
    kvHeads: 8, headDim: 128, maxContext: 131_072,
    source: "https://arxiv.org/html/2407.21783#S3.SS2",
  },
  llama70: {
    label: "Llama 3.1 70B", parameters: 70_000_000_000, attentionLayers: 80,
    kvHeads: 8, headDim: 128, maxContext: 131_072,
    source: "https://arxiv.org/html/2407.21783#S3.SS2",
  },
} satisfies Record<string, ModelPreset>;

export type ModelId = keyof typeof MODEL_PRESETS;

export const GPU_PRESETS = {
  l4: { label: "NVIDIA L4", nominalGiB: 24, source: "https://www.nvidia.com/en-us/data-center/l4/" },
  rtx5090: { label: "GeForce RTX 5090", nominalGiB: 32, source: "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/" },
  a10040: { label: "NVIDIA A100", nominalGiB: 40, source: "https://www.nvidia.com/en-us/data-center/a100/" },
  l40s: { label: "NVIDIA L40S", nominalGiB: 48, source: "https://www.nvidia.com/en-us/data-center/l40s/" },
  a10080: { label: "NVIDIA A100", nominalGiB: 80, source: "https://www.nvidia.com/en-us/data-center/a100/" },
  h100: { label: "NVIDIA H100 SXM", nominalGiB: 80, source: "https://www.nvidia.com/en-us/data-center/h100/" },
  h200: { label: "NVIDIA H200", nominalGiB: 141, source: "https://www.nvidia.com/en-us/data-center/h200/" },
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
  model: ModelId;
  weight: keyof typeof WEIGHT_OPTIONS;
  kv: keyof typeof KV_OPTIONS;
  context: number;
  batch: number;
  gpuCount: number;
  gpuGiB: number;
  utilizationPct: number;
  overheadGiB: number;
};

export function attentionKvBytesPerToken(model: ModelPreset, kv: keyof typeof KV_OPTIONS) {
  return 2 * model.attentionLayers * model.kvHeads * model.headDim * KV_OPTIONS[kv].bits / 8;
}

function deltaNetStateBytes(recurrent: ModelPreset["recurrent"]) {
  if (!recurrent) return 0;

  // One BF16 convolution window and one FP32 recurrent state per DeltaNet layer.
  const convolutionChannels = 2 * recurrent.keyHeads * recurrent.keyDim + recurrent.valueHeads * recurrent.valueDim;
  const convolutionBytes = convolutionChannels * (recurrent.convolutionWidth - 1) * 2;
  const recurrentBytes = recurrent.valueHeads * recurrent.keyDim * recurrent.valueDim * 4;
  return recurrent.layers * (convolutionBytes + recurrentBytes);
}

export function estimateModelMemory(input: MemoryInputs) {
  const model: ModelPreset = MODEL_PRESETS[input.model];
  const weight = WEIGHT_OPTIONS[input.weight];
  const scaleBytes = "groupSize" in weight ? Math.ceil(model.parameters / weight.groupSize) * 2 : 0;
  const weightBytes = bytesFor(model.parameters, weight.bits) + scaleBytes;
  const kvPerToken = attentionKvBytesPerToken(model, input.kv);
  const kvPerSequence = kvPerToken * input.context;
  const statePerSequence = deltaNetStateBytes(model.recurrent);

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
