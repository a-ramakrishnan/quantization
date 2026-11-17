export type QuantScheme = 'symmetric' | 'asymmetric';
export type Granularity = 'per-tensor' | 'per-channel' | 'per-group';
export type PresetKey = 'normal' | 'outliers' | 'biased' | 'uniform';

export interface QuantConfig {
  bits: number;
  scheme: QuantScheme;
  granularity: Granularity;
  clipRatio: number; // 0.5 to 1.0
  manualOverride: boolean;
  manualScale: number;
  manualZeroPoint: number;
}

export interface GroupQuantParams {
  id: string;
  label: string;
  minVal: number;
  maxVal: number;
  clippedMin: number;
  clippedMax: number;
  scale: number;
  zeroPoint: number;
  qMin: number;
  qMax: number;
}

export interface CellResult {
  row: number;
  col: number;
  wFp32: number;
  wq: number;
  rawRoundedQ: number;
  wDequant: number;
  error: number; // wFp32 - wDequant
  absError: number;
  isClipped: boolean;
  groupId: string;
  scale: number;
  zeroPoint: number;
  qMin: number;
  qMax: number;
}

export interface MemoryStats {
  fp32Bits: number;
  quantizedWeightBits: number;
  metadataBits: number;
  totalQuantizedBits: number;
  compressionRatio: number;
  bandwidthSavedPct: number;
}

export interface QuantizationResult {
  cells: CellResult[][];
  groups: GroupQuantParams[];
  yTrue: number[];
  yQuant: number[];
  yError: number[];
  weightMse: number;
  weightMae: number;
  weightMaxErr: number;
  weightSqnrDb: number;
  outputMse: number;
  outputMae: number;
  cosineSimilarity: number;
  qMin: number;
  qMax: number;
  memory: MemoryStats;
}

export type GpuComponentId =
  | 'host'
  | 'hbm'
  | 'l2'
  | 'sram'
  | 'registers'
  | 'tensor_cores';

export interface PipelineStepInfo {
  id: number;
  shortTitle: string;
  title: string;
  gpuTarget: GpuComponentId;
  gpuLabel: string;
  formula: string;
  explanation: string;
  whyItMatters: string;
}
