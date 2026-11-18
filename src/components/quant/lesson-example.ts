import { basicProjection } from "@/utils/basic-quantization";
import { matVec, quantize } from "@/utils/quantization";

export const WEIGHT_FORMATS = ["INT4", "INT8", "FP8", "FP16"] as const;
export const INFERENCE_MODES = ["W8A16", "W8A8"] as const;
export type LessonMode = typeof WEIGHT_FORMATS[number] | typeof INFERENCE_MODES[number];

export const WEIGHTS = [
  [0.578125, -0.25, 0.75, 0.125],
  [-0.375, 0.625, 0.25, 3 * 2 ** -25],
];
export const INPUT = [1 + 2 ** -12, -0.5, 0.25, 2];
export const MAX_WEIGHT = Math.max(...WEIGHTS.flat().map(Math.abs));

export const PROJECTIONS = {
  INT4: basicProjection(WEIGHTS, INPUT, "INT4"),
  INT8: basicProjection(WEIGHTS, INPUT, "INT8"),
  FP8: basicProjection(WEIGHTS, INPUT, "FP8"),
  FP16: basicProjection(WEIGHTS, INPUT, "FP16"),
};

const config = { symmetric: true, granularity: "tensor", alpha: 1 } as const;
const int8Weights = PROJECTIONS.INT8.result;
const int8Activations = quantize([INPUT], { ...config, format: "INT8" });

export const ACTIVATION_SCALE = int8Activations.params!.scales[0]!;
export const ACTIVATION_CODES = int8Activations.q[0]!;
export const INT8_SUMS = matVec(int8Weights.q, ACTIVATION_CODES);
const weightScale = int8Weights.params!.scales[0]!;

export const FP16_INPUT = quantize([INPUT], { ...config, format: "FP16" }).deq[0]!;
export const FP16_WEIGHTS = quantize(int8Weights.deq, { ...config, format: "FP16" }).deq;

export const INFERENCE_OUTPUTS = {
  W8A16: matVec(FP16_WEIGHTS, FP16_INPUT),
  W8A8: INT8_SUMS.map(value => value * weightScale * ACTIVATION_SCALE),
};
