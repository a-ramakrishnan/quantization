import { quantize } from "./quantization";

export type NextToken = { text: string; logit: number };
export type NextWords = {
  continuation: string;
  wordCount: number;
  candidates: [NextToken, NextToken];
};

export type PredictionMessage =
  | { kind: "progress"; percent?: number }
  | { kind: "generating" }
  | { kind: "result"; prediction: NextWords }
  | { kind: "error"; message: string };

export const MAX_PROMPT_CHARS = 2_000;
export const WEIGHT_START = 30 / 512;
export const WEIGHT_SCALE = 0.125;
export const ACTIVATION = 2.75;

const segmenter = new Intl.Segmenter(undefined, { granularity: "word" });

/** Include the terminating punctuation only when it precedes one of the selected words. */
export function continuationWords(prompt: string, continuation: string) {
  let wordCount = 0;
  for (const item of segmenter.segment(prompt + continuation)) {
    const end = item.index + item.segment.length - prompt.length;
    if (!item.isWordLike || end <= 0) continue;
    wordCount += 1;
    if (wordCount === 2) return { continuation: continuation.slice(0, end), wordCount };
  }
  return { continuation: continuation.trimEnd(), wordCount };
}

export function predictionTrace(weight: number, candidates: NextWords["candidates"], mix: number) {
  const result = quantize([[7 * WEIGHT_SCALE, weight]], {
    format: "INT4", symmetric: true, granularity: "tensor", alpha: 1,
  });
  const code = result.q[0]![1]!;
  const restored = result.deq[0]![1]!;
  const movingWeight = weight + (restored - weight) * mix;
  // Hold every other contribution constant. At the starting slider value these are the model logits.
  const firstLogit = candidates[0].logit + ACTIVATION * (movingWeight - WEIGHT_START);
  const secondLogit = candidates[1].logit;
  const firstProbability = 1 / (1 + Math.exp(secondLogit - firstLogit));
  return { code, restored, movingWeight, firstLogit, secondLogit, firstProbability };
}
