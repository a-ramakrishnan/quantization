import {
  CellResult,
  GroupQuantParams,
  PresetKey,
  QuantConfig,
  QuantizationResult,
} from '../types/quantization';

export interface PresetData {
  key: PresetKey;
  name: string;
  description: string;
  weights: number[][];
  inputs: number[];
}

export const PRESETS: Record<PresetKey, PresetData> = {
  normal: {
    key: 'normal',
    name: 'Normal Weights',
    description:
      'Typical bell-curve trained weights centered near zero. Symmetric INT4/INT8 works cleanly with low error.',
    weights: [
      [0.42, -1.15, 0.88, -0.34],
      [-0.76, 1.24, -0.19, 0.63],
      [0.15, -0.58, 1.05, -0.92],
      [0.95, 0.27, -1.31, 0.48],
    ],
    inputs: [1.2, -0.5, 0.8, 1.5],
  },
  outliers: {
    key: 'outliers',
    name: 'With Outliers',
    description:
      'Two large outlier weights (9.60, -8.40) stretch the per-tensor scale! Watch how clipping or Per-Channel/Per-Group rescues the small weights.',
    weights: [
      [0.35, -0.42, 9.6, 0.18],
      [-0.51, 0.64, -0.22, 0.47],
      [0.29, -8.4, 0.31, -0.15],
      [0.44, -0.38, 0.52, -0.27],
    ],
    inputs: [1.0, 0.75, -0.6, 1.25],
  },
  biased: {
    key: 'biased',
    name: 'Shifted / Biased',
    description:
      'All weights are positive [1.10 .. 4.80]. Symmetric quantization wastes half its integer bins; Asymmetric (Affine) uses a Zero-Point Z to utilize all bins!',
    weights: [
      [1.2, 2.45, 3.1, 1.85],
      [2.9, 4.8, 1.55, 3.4],
      [1.1, 2.15, 3.95, 2.6],
      [4.2, 1.75, 2.8, 3.65],
    ],
    inputs: [0.8, 1.1, -0.4, 0.9],
  },
  uniform: {
    key: 'uniform',
    name: 'Uniform Spread',
    description:
      'Evenly spaced weights from -1.50 to +1.50, ideal for observing uniform quantization step size Δ = S.',
    weights: [
      [-1.5, -1.3, -1.1, -0.9],
      [-0.7, -0.5, -0.3, -0.1],
      [0.1, 0.3, 0.5, 0.7],
      [0.9, 1.1, 1.3, 1.5],
    ],
    inputs: [1.0, -1.0, 0.5, 0.5],
  },
};

export function getQuantBounds(bits: number, scheme: 'symmetric' | 'asymmetric') {
  const qMax = Math.pow(2, bits - 1) - 1;
  const qMin = scheme === 'symmetric' ? -qMax : -Math.pow(2, bits - 1);
  return { qMin, qMax };
}

function computeGroupParams(
  values: number[],
  id: string,
  label: string,
  config: QuantConfig
): GroupQuantParams {
  const { qMin, qMax } = getQuantBounds(config.bits, config.scheme);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);

  let clippedMin: number;
  let clippedMax: number;
  let scale: number;
  let zeroPoint: number;

  if (config.scheme === 'symmetric') {
    const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal), 1e-6) * config.clipRatio;
    clippedMin = -absMax;
    clippedMax = absMax;
    scale = absMax / qMax;
    zeroPoint = 0;
  } else {
    const center = (minVal + maxVal) / 2;
    const halfSpan = Math.max((maxVal - minVal) / 2, 1e-6) * config.clipRatio;
    clippedMin = center - halfSpan;
    clippedMax = center + halfSpan;
    scale = (clippedMax - clippedMin) / (qMax - qMin);
    if (scale < 1e-8) scale = 1e-6;
    const rawZ = Math.round(qMin - clippedMin / scale);
    zeroPoint = Math.max(qMin, Math.min(qMax, rawZ));
  }

  if (config.manualOverride && config.granularity === 'per-tensor') {
    scale = Math.max(config.manualScale, 1e-5);
    zeroPoint =
      config.scheme === 'symmetric'
        ? 0
        : Math.max(qMin, Math.min(qMax, Math.round(config.manualZeroPoint)));
  }

  return {
    id,
    label,
    minVal,
    maxVal,
    clippedMin,
    clippedMax,
    scale,
    zeroPoint,
    qMin,
    qMax,
  };
}

export function quantizeLinearLayer(
  weights: number[][],
  inputs: number[],
  config: QuantConfig
): QuantizationResult {
  const rows = weights.length;
  const cols = weights[0]?.length ?? 0;
  const { qMin, qMax } = getQuantBounds(config.bits, config.scheme);

  const groups: GroupQuantParams[] = [];
  const groupLookup = new Map<string, GroupQuantParams>();

  if (config.granularity === 'per-tensor') {
    const flat = weights.flat();
    const g = computeGroupParams(flat, 'tensor', 'All Weights (Tensor)', config);
    groups.push(g);
    groupLookup.set('tensor', g);
  } else if (config.granularity === 'per-channel') {
    for (let r = 0; r < rows; r++) {
      const id = `ch-${r}`;
      const g = computeGroupParams(weights[r], id, `Row ${r} (Out Ch ${r})`, config);
      groups.push(g);
      groupLookup.set(id, g);
    }
  } else {
    // per-group (group size = 2 along columns)
    const groupSize = 2;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c += groupSize) {
        const slice = weights[r].slice(c, c + groupSize);
        const id = `g-${r}-${Math.floor(c / groupSize)}`;
        const g = computeGroupParams(
          slice,
          id,
          `R${r} C${c}..${Math.min(c + groupSize - 1, cols - 1)}`,
          config
        );
        groups.push(g);
        groupLookup.set(id, g);
      }
    }
  }

  const getGroupId = (r: number, c: number) => {
    if (config.granularity === 'per-tensor') return 'tensor';
    if (config.granularity === 'per-channel') return `ch-${r}`;
    return `g-${r}-${Math.floor(c / 2)}`;
  };

  let sqErrSum = 0;
  let absErrSum = 0;
  let maxErr = 0;
  let signalSqSum = 0;
  const totalCells = rows * cols;

  const cells: CellResult[][] = weights.map((rowArr, r) =>
    rowArr.map((w, c) => {
      const gid = getGroupId(r, c);
      const gp = groupLookup.get(gid)!;
      const rawRoundedQ = Math.round(w / gp.scale) + gp.zeroPoint;
      const wq = Math.max(gp.qMin, Math.min(gp.qMax, rawRoundedQ));
      const isClipped = rawRoundedQ < gp.qMin || rawRoundedQ > gp.qMax;
      const wDequant = gp.scale * (wq - gp.zeroPoint);
      const error = w - wDequant;
      const absError = Math.abs(error);

      sqErrSum += error * error;
      absErrSum += absError;
      if (absError > maxErr) maxErr = absError;
      signalSqSum += w * w;

      return {
        row: r,
        col: c,
        wFp32: w,
        wq,
        rawRoundedQ,
        wDequant,
        error,
        absError,
        isClipped,
        groupId: gid,
        scale: gp.scale,
        zeroPoint: gp.zeroPoint,
        qMin: gp.qMin,
        qMax: gp.qMax,
      };
    })
  );

  // Forward pass: y = W * x
  const yTrue: number[] = [];
  const yQuant: number[] = [];
  const yError: number[] = [];

  for (let r = 0; r < rows; r++) {
    let sumTrue = 0;
    let sumQuant = 0;
    for (let c = 0; c < cols; c++) {
      sumTrue += weights[r][c] * inputs[c];
      sumQuant += cells[r][c].wDequant * inputs[c];
    }
    yTrue.push(sumTrue);
    yQuant.push(sumQuant);
    yError.push(sumTrue - sumQuant);
  }

  const weightMse = sqErrSum / totalCells;
  const weightMae = absErrSum / totalCells;
  const weightSqnrDb =
    sqErrSum > 1e-12
      ? 10 * Math.log10((signalSqSum + 1e-12) / sqErrSum)
      : 99.9;

  const outSqErr = yError.reduce((acc, e) => acc + e * e, 0);
  const outAbsErr = yError.reduce((acc, e) => acc + Math.abs(e), 0);
  const outputMse = outSqErr / rows;
  const outputMae = outAbsErr / rows;

  const dot = yTrue.reduce((acc, val, i) => acc + val * yQuant[i], 0);
  const normTrue = Math.sqrt(yTrue.reduce((acc, val) => acc + val * val, 0));
  const normQuant = Math.sqrt(yQuant.reduce((acc, val) => acc + val * val, 0));
  const cosineSimilarity =
    normTrue > 1e-9 && normQuant > 1e-9 ? dot / (normTrue * normQuant) : 1;

  // Memory footprint comparison
  const fp32Bits = totalCells * 32;
  const quantizedWeightBits = totalCells * config.bits;
  // Metadata: 16-bit FP16 scale (+ 8-bit zeroPoint if asymmetric) per group
  const bitsPerGroupMeta = config.scheme === 'symmetric' ? 16 : 24;
  const metadataBits = groups.length * bitsPerGroupMeta;
  const totalQuantizedBits = quantizedWeightBits + metadataBits;
  const compressionRatio = fp32Bits / totalQuantizedBits;
  const bandwidthSavedPct = ((fp32Bits - totalQuantizedBits) / fp32Bits) * 100;

  return {
    cells,
    groups,
    yTrue,
    yQuant,
    yError,
    weightMse,
    weightMae,
    weightMaxErr: maxErr,
    weightSqnrDb,
    outputMse,
    outputMae,
    cosineSimilarity,
    qMin,
    qMax,
    memory: {
      fp32Bits,
      quantizedWeightBits,
      metadataBits,
      totalQuantizedBits,
      compressionRatio,
      bandwidthSavedPct,
    },
  };
}

export function generateRandomWeights(): number[][] {
  return Array.from({ length: 4 }, () =>
    Array.from({ length: 4 }, () => Number(((Math.random() - 0.5) * 3.0).toFixed(2)))
  );
}
