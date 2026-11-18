import { describe, expect, it } from "vitest";
import { continuationWords, predictionTrace, WEIGHT_START } from "./next-word";
import type { NextWords } from "./next-word";

describe("next-word demo", () => {
  it("keeps the first two whole words and the original spacing", () => {
    expect(continuationWords("The sky is", " blue today, with clouds")).toEqual({
      continuation: " blue today", wordCount: 2,
    });
    expect(continuationWords("Say hel", "lo world again")).toEqual({ continuation: "lo world", wordCount: 2 });
    expect(continuationWords("Once", " ...")).toEqual({ continuation: " ...", wordCount: 0 });
  });

  it("uses the reference scores before rounding and flips the sample at INT4", () => {
    const candidates: NextWords["candidates"] = [{ text: "blue", logit: 0.08 }, { text: "clear", logit: 0 }];
    const before = predictionTrace(WEIGHT_START, candidates, 0);
    const after = predictionTrace(WEIGHT_START, candidates, 1);
    expect(before.firstLogit).toBe(0.08);
    expect(before.firstProbability).toBeCloseTo(0.52, 3);
    expect(after.restored).toBe(0);
    expect(after.firstProbability).toBeCloseTo(0.48, 3);
    expect(predictionTrace(0.0625, candidates, 1).restored).toBe(0.125);
  });
});
