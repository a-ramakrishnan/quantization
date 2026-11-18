import { describe, expect, it } from "vitest";
import { continuationWords, predictionTrace, tokenLogProbability, WEIGHT_START, wordPath } from "./next-word";
import type { WordPath } from "./next-word";

describe("next-word demo", () => {
  it("keeps the first two whole words and the original spacing", () => {
    expect(continuationWords("The sky is", " blue today, with clouds")).toEqual({
      continuation: " blue today", wordCount: 2,
    });
    expect(continuationWords("Say hel", "lo world again")).toEqual({ continuation: "lo world", wordCount: 2 });
    expect(continuationWords("Once", " ...")).toEqual({ continuation: " ...", wordCount: 0 });
  });

  it("uses the reference scores before rounding and flips the sample at INT4", () => {
    const candidates: [WordPath, WordPath] = [
      { word: "blue", completion: " blue", score: -0.62, pieces: [] },
      { word: "clear", completion: " clear", score: -0.7, pieces: [] },
    ];
    const before = predictionTrace(WEIGHT_START, candidates, 0);
    const after = predictionTrace(WEIGHT_START, candidates, 1);
    expect(before.firstScore).toBe(-0.62);
    expect(before.firstProbability).toBeCloseTo(0.52, 3);
    expect(after.restored).toBe(0);
    expect(after.firstProbability).toBeCloseTo(0.48, 3);
    expect(predictionTrace(0.0625, candidates, 1).restored).toBe(0.125);
  });

  it("combines fragments into a whole word and excludes the lookahead token", () => {
    expect(wordPath("The next word is", [" ", " un", " unbelievable", " unbelievable next"], [-0.2, -0.4, -0.7, -0.9])).toEqual({
      word: "unbelievable",
      completion: " unbelievable",
      score: -1.3,
      pieces: [
        { text: "␠", logProbability: -0.2 }, { text: "un", logProbability: -0.4 },
        { text: "believable", logProbability: -0.7 },
      ],
    });
    expect(wordPath("Say hel", ["lo", "lo next"], [-0.5, -0.8])?.word).toBe("hello");
    expect(wordPath("Say hel", ["lo", "lo next"], [-0.5, -0.8])?.completion).toBe("lo");
    expect(wordPath("Start", [" ", " \uFFFD"], [-0.1, -1])).toBeUndefined();
    expect(wordPath("Start", [" end"], [-1])).toBeUndefined();
    expect(wordPath("Start", [" it", " it'"], [-1, -2])).toBeUndefined();
    expect(wordPath("Start", [" caf", " caf\uFFFD"], [-1, -2])).toBeUndefined();
    expect(wordPath("Start", [" end", " end"], [-1, -2], true)?.score).toBe(-1);
  });

  it("follows a chosen word without changing it into a different word", () => {
    expect(wordPath("The kitchen", ["s", "s floor", "s floor next"], [-1, -2, -3], false, true)).toBeUndefined();
    expect(wordPath("The kitchen", [" floor", " floor was"], [-1, -2], false, true)?.completion).toBe(" floor");
    expect(continuationWords("The kitchen", "s were spotless", 2, true).wordCount).toBe(0);
    expect(continuationWords("The kitchen", " floor was spotless", 2, true).continuation).toBe(" floor was");
  });

  it("normalizes token log probabilities across the vocabulary", () => {
    const scores = Float32Array.from([1_000, 1_000, 1_000 - Math.log(2)]);
    expect(Math.exp(tokenLogProbability(scores, 0))).toBeCloseTo(0.4, 4);
    expect(Math.exp(tokenLogProbability(scores, 2))).toBeCloseTo(0.2, 4);
  });
});
