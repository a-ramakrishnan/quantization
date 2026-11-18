import { quantize } from "./quantization";

export type WordPath = {
  word: string;
  completion: string;
  score: number;
  pieces: { text: string; logProbability: number }[];
};
export type NextWords = {
  continuation: string;
  wordCount: number;
  candidates: [WordPath, WordPath] | null;
};

export type PredictionRequest =
  | { kind: "predict"; prompt: string }
  | { kind: "follow"; prompt: string; completion: string };

export type PredictionMessage =
  | { kind: "progress"; percent?: number }
  | { kind: "generating" }
  | { kind: "result"; prediction: NextWords }
  | { kind: "follow-result"; continuation: string }
  | { kind: "error"; message: string };

export const MAX_PROMPT_CHARS = 2_000;
export const WEIGHT_START = 30 / 512;
export const WEIGHT_SCALE = 0.125;
export const ACTIVATION = 2.75;

const segmenter = new Intl.Segmenter(undefined, { granularity: "word" });

export function firstContinuationWord(prompt: string, continuation: string, requireNewWord = false) {
  for (const item of segmenter.segment(prompt + continuation)) {
    const end = item.index + item.segment.length - prompt.length;
    if (item.isWordLike && end > 0) {
      if (requireNewWord && item.index < prompt.length) return;
      return { word: item.segment, start: Math.max(0, item.index - prompt.length), end };
    }
  }
}

export function hasWordBoundary(continuation: string, end: number) {
  // Punctuation or an incomplete UTF-8 token at the end can still extend the current word.
  return end < continuation.length && !/^['’.-]?\uFFFD*$/u.test(continuation.slice(end));
}

/** Preserve the generated spacing and punctuation through the requested word. */
export function continuationWords(prompt: string, continuation: string, limit = 2, requireNewWord = false) {
  let wordCount = 0;
  for (const item of segmenter.segment(prompt + continuation)) {
    const end = item.index + item.segment.length - prompt.length;
    if (!item.isWordLike || end <= 0) continue;
    if (requireNewWord && item.index < prompt.length) return { continuation: "", wordCount: 0 };
    wordCount += 1;
    if (wordCount === limit) return { continuation: continuation.slice(0, end), wordCount };
  }
  return { continuation: continuation.trimEnd(), wordCount };
}

export function tokenLogProbability(scores: Float32Array, token: number) {
  let max = -Infinity;
  for (const score of scores) max = Math.max(max, score);
  let sum = 0;
  for (const score of scores) sum += Math.exp(score - max);
  return scores[token]! - max - Math.log(sum);
}

export function wordPath(prompt: string, prefixes: string[], logProbabilities: number[], ended = false, requireNewWord = false): WordPath | undefined {
  const decoded = prefixes.at(-1) ?? "";
  const word = firstContinuationWord(prompt, decoded, requireNewWord);
  if (!word) return;

  // The last decoded word may be a fragment when generation hits its token limit.
  if (!ended && !hasWordBoundary(decoded, word.end)) return;
  const last = prefixes.findIndex(prefix => prefix.length >= word.end && decoded.startsWith(prefix));
  if (last < 0) return;

  const pieces = prefixes.slice(0, last + 1).map((prefix, index) => {
    const previous = prefixes[index - 1] ?? "";
    const text = !prefix.includes("\uFFFD") && decoded.startsWith(prefix) && prefix.startsWith(previous)
      ? prefix.slice(previous.length).replaceAll(" ", "␠").replaceAll("\n", "\\n").replaceAll("\t", "\\t")
      : "byte fragment";
    return { text: text || "empty", logProbability: logProbabilities[index]! };
  });
  return {
    word: word.word, completion: decoded.slice(0, word.end),
    score: pieces.reduce((total, piece) => total + piece.logProbability, 0), pieces,
  };
}

export function predictionTrace(weight: number, candidates: [WordPath, WordPath], mix: number) {
  const result = quantize([[7 * WEIGHT_SCALE, weight]], {
    format: "INT4", symmetric: true, granularity: "tensor", alpha: 1,
  });
  const code = result.q[0]![1]!;
  const restored = result.deq[0]![1]!;
  const movingWeight = weight + (restored - weight) * mix;
  const firstScore = candidates[0].score + ACTIVATION * (movingWeight - WEIGHT_START);
  const secondScore = candidates[1].score;
  const firstProbability = 1 / (1 + Math.exp(secondScore - firstScore));
  return { code, restored, movingWeight, firstScore, secondScore, firstProbability };
}
