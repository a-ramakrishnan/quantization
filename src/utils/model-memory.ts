import { bytesFor } from "./quantization";

export const GIB = 2 ** 30;
export type ModelPreset = {
  label: string;
  parameters: number;
  attentionLayers: number;
  kvHeads: number;
  headDim: number;
  queryHeads: number;
  maxContext: number;
  source: string;
  card?: string;
  sliding?: { layers: number; window: number };
  latent?: { rank: number; rotaryDim: number };
  deploymentNote?: string;
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
    kvHeads: 4, headDim: 256, queryHeads: 16, maxContext: 262_144,
    source: "https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json",
    recurrent: { layers: 24, interval: 4, keyHeads: 16, valueHeads: 32, keyDim: 128, valueDim: 128, convolutionWidth: 4 },
  },
  qwen25: {
    label: "Qwen2.5-7B Instruct", parameters: 7_610_000_000, attentionLayers: 28,
    kvHeads: 4, headDim: 128, queryHeads: 28, maxContext: 32_768,
    source: "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
  },
  qwen3: {
    label: "Qwen3-32B", parameters: 32_800_000_000, attentionLayers: 64,
    kvHeads: 8, headDim: 128, queryHeads: 64, maxContext: 32_768,
    source: "https://huggingface.co/Qwen/Qwen3-32B",
  },
  llama8: {
    label: "Llama 3.1 8B", parameters: 8_000_000_000, attentionLayers: 32,
    kvHeads: 8, headDim: 128, queryHeads: 32, maxContext: 131_072,
    source: "https://arxiv.org/html/2407.21783#S3.SS2",
  },
  llama70: {
    label: "Llama 3.1 70B", parameters: 70_000_000_000, attentionLayers: 80,
    kvHeads: 8, headDim: 128, queryHeads: 64, maxContext: 131_072,
    source: "https://arxiv.org/html/2407.21783#S3.SS2",
  },
  llama33: {
    label: "Llama 3.3 70B Instruct", parameters: 70_000_000_000, attentionLayers: 80,
    kvHeads: 8, headDim: 128, queryHeads: 64, maxContext: 131_072,
    source: "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
  },
  glm47: {
    label: "Z.ai GLM-4.7", parameters: 358_000_000_000, attentionLayers: 92,
    kvHeads: 8, headDim: 128, queryHeads: 96, maxContext: 202_752,
    source: "https://huggingface.co/zai-org/GLM-4.7/blob/main/config.json",
    card: "https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-4.7.yaml",
  },
  glm47flash: {
    label: "Z.ai GLM-4.7-Flash", parameters: 31_000_000_000, attentionLayers: 47,
    kvHeads: 20, headDim: 256, queryHeads: 20, maxContext: 202_752,
    latent: { rank: 512, rotaryDim: 64 },
    source: "https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json",
    card: "https://huggingface.co/zai-org/GLM-4.7-Flash",
  },
  kimi26: {
    label: "Moonshot Kimi K2.6", parameters: 1_040_000_000_000, attentionLayers: 61,
    kvHeads: 64, headDim: 192, queryHeads: 64, maxContext: 262_144,
    latent: { rank: 512, rotaryDim: 64 },
    source: "https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/config.json",
    card: "https://huggingface.co/moonshotai/Kimi-K2.6",
    deploymentNote: "Text workload using a compressed MLA cache. Vision processing and backend-specific MLA packing are outside this payload estimate.",
  },
  minimax25: {
    label: "MiniMax M2.5", parameters: 230_000_000_000, attentionLayers: 62,
    kvHeads: 8, headDim: 128, queryHeads: 48, maxContext: 196_608,
    source: "https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/config.json",
    card: "https://huggingface.co/MiniMaxAI/MiniMax-M2.5",
  },
  inkling2: {
    label: "Thinking Machines Inkling 2 Small", parameters: 594_000_000_000, attentionLayers: 8,
    kvHeads: 8, headDim: 128, queryHeads: 48, maxContext: 1_048_576,
    sliding: { layers: 40, window: 1_024 },
    source: "https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/main/config.json",
    card: "https://huggingface.co/thinkingmachines/Inkling-2-Small",
    deploymentNote: "Text decoder KV only. Short-convolution state, vision processing and speculative decoding can add memory. The publisher lists 2× B300 or 4× H200 for its native NVFP4 checkpoint.",
  },
  olmo3: {
    label: "Ai2 Olmo 3 32B Think", parameters: 32_000_000_000, attentionLayers: 16,
    kvHeads: 8, headDim: 128, queryHeads: 40, maxContext: 65_536,
    sliding: { layers: 48, window: 4_096 },
    source: "https://huggingface.co/allenai/Olmo-3-32B-Think/blob/main/config.json",
    card: "https://huggingface.co/allenai/Olmo-3-32B-Think",
  },
} satisfies Record<string, ModelPreset>;

export type ModelId = keyof typeof MODEL_PRESETS;

export function tensorParallelOptions(model: ModelId) {
  return [1, 2, 4, 8, 16].filter(size => MODEL_PRESETS[model].queryHeads % size === 0);
}

export const GPU_PRESETS = {
  l4: { label: "NVIDIA L4", nominalGiB: 24, source: "https://www.nvidia.com/en-us/data-center/l4/" },
  rtx5090: { label: "GeForce RTX 5090", nominalGiB: 32, source: "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/" },
  a10040: { label: "NVIDIA A100", nominalGiB: 40, source: "https://www.nvidia.com/en-us/data-center/a100/" },
  l40s: { label: "NVIDIA L40S", nominalGiB: 48, source: "https://www.nvidia.com/en-us/data-center/l40s/" },
  a10080: { label: "NVIDIA A100", nominalGiB: 80, source: "https://www.nvidia.com/en-us/data-center/a100/" },
  h100: { label: "NVIDIA H100 SXM", nominalGiB: 80, source: "https://www.nvidia.com/en-us/data-center/h100/" },
  h200: { label: "NVIDIA H200", nominalGiB: 141, source: "https://www.nvidia.com/en-us/data-center/h200/" },
  b200: { label: "NVIDIA B200", nominalGiB: 180, source: "https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html" },
  b300: { label: "NVIDIA B300", nominalGiB: 288, source: "https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html" },
} as const;
export type GpuId = keyof typeof GPU_PRESETS;
export const WEIGHT_OPTIONS = {
  bf16: { label: "BF16 / FP16", bits: 16 },
  fp8: { label: "FP8 · raw payload", bits: 8 },
  mxfp8: { label: "MXFP8 · Block-32", bits: 8, groupSize: 32, scaleBytes: 1 },
  int8: { label: "INT8 · Group-128", bits: 8, groupSize: 128, scaleBytes: 2 },
  int4: { label: "INT4 · Group-128", bits: 4, groupSize: 128, scaleBytes: 2 },
  nvfp4: { label: "NVFP4 · Block-16", bits: 4, groupSize: 16, scaleBytes: 1 },
} as const;
export const KV_OPTIONS = {
  bf16: { label: "BF16 / FP16", bits: 16 },
  fp8: { label: "FP8 / INT8", bits: 8 },
  int4: { label: "Packed INT4", bits: 4 },
} as const;

export type MemoryInputs = {
  model: ModelId;
  weight: keyof typeof WEIGHT_OPTIONS;
  kv: keyof typeof KV_OPTIONS;
  context: number;
  batch: number;
  gpuCount: number;
  tensorParallel?: number;
  gpuGiB: number;
  utilizationPct: number;
  overheadGiB: number;
};

export function attentionKvBytesPerToken(model: ModelPreset, kv: keyof typeof KV_OPTIONS) {
  const valuesPerLayer = model.latent
    ? model.latent.rank + model.latent.rotaryDim
    : 2 * model.kvHeads * model.headDim;
  return model.attentionLayers * valuesPerLayer * KV_OPTIONS[kv].bits / 8;
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
  const scaleBytes = "groupSize" in weight ? Math.ceil(model.parameters / weight.groupSize) * weight.scaleBytes : 0;
  const weightBytes = bytesFor(model.parameters, weight.bits) + scaleBytes;
  const kvPerToken = attentionKvBytesPerToken(model, input.kv);
  const slidingKvPerSequence = model.sliding
    ? 2 * model.sliding.layers * model.kvHeads * model.headDim * KV_OPTIONS[input.kv].bits / 8
      * Math.min(input.context, model.sliding.window)
    : 0;
  const kvPerSequence = kvPerToken * input.context + slidingKvPerSequence;
  const statePerSequence = deltaNetStateBytes(model.recurrent);
  const tensorParallel = input.tensorParallel ?? 1;
  // GQA shards by KV head. Compressed MLA stays replicated without context parallelism.
  const kvShards = model.latent ? 1 : Math.min(tensorParallel, model.kvHeads);
  const weightPerGpu = Math.ceil(weightBytes / tensorParallel);
  const kvPerGpuSequence = kvPerSequence / kvShards;
  // Keep fixed state replicated as a conservative allowance.
  const bytesPerGpuSequence = kvPerGpuSequence + statePerSequence;

  const sequencesPerGpu = Math.ceil(input.batch / input.gpuCount);
  const capacity = input.gpuGiB * GIB;
  const budget = capacity * input.utilizationPct / 100;
  const overhead = input.overheadGiB * GIB;
  const peak = weightPerGpu + overhead + sequencesPerGpu * bytesPerGpuSequence;
  const maxSequencesPerReplica = Math.max(0, Math.floor((budget - weightPerGpu - overhead) / bytesPerGpuSequence));

  return {
    weightBytes, weightPerGpu, kvPerToken, kvPerSequence, slidingKvPerSequence, kvPerGpuSequence, statePerSequence,
    sequencesPerGpu, capacity, budget, overhead, peak,
    kvForBatch: kvPerSequence * input.batch,
    maxSequencesPerReplica,
    maxSequences: input.gpuCount * maxSequencesPerReplica,
    fits: peak <= budget,
  };
}

export type CapacityInputs = Omit<MemoryInputs, "gpuCount" | "tensorParallel" | "gpuGiB">;

export function planGpuCapacity(input: CapacityInputs, gpuGiB: number) {
  let best: { totalGpus: number; replicas: number; tensorParallel: number; kvPerGpu: number } | undefined;

  for (const tensorParallel of tensorParallelOptions(input.model)) {
    const estimate = estimateModelMemory({ ...input, gpuGiB, gpuCount: 1, tensorParallel });
    if (estimate.maxSequencesPerReplica === 0) continue;

    const replicas = Math.ceil(input.batch / estimate.maxSequencesPerReplica);
    const totalGpus = replicas * tensorParallel;
    if (best && best.totalGpus <= totalGpus) continue;

    const balanced = estimateModelMemory({ ...input, gpuGiB, gpuCount: replicas, tensorParallel });
    best = { totalGpus, replicas, tensorParallel,
      kvPerGpu: balanced.kvPerGpuSequence * balanced.sequencesPerGpu };
  }

  return best;
}
