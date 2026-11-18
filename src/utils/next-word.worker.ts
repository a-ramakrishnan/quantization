import {
  AutoModelForCausalLM, AutoTokenizer, env, LogitsProcessor, LogitsProcessorList,
  StoppingCriteria, StoppingCriteriaList, Tensor,
} from "@huggingface/transformers";
import { continuationWords, firstContinuationWord, hasWordBoundary, MAX_PROMPT_CHARS, tokenLogProbability, wordPath } from "./next-word";
import type { NextWords, PredictionMessage, PredictionRequest } from "./next-word";

const modelId = "Xenova/distilgpt2";
const revision = "a41c10485c18a64b6606729b6a082330cbd8f49e";
const maxInputTokens = 512;
const maxNewTokens = 24;

env.allowLocalModels = false;
if (env.backends.onnx?.wasm) env.backends.onnx.wasm.numThreads = 1;

function send(message: PredictionMessage) {
  globalThis.postMessage(message);
}

async function loadModel() {
  send({ kind: "progress" });
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(modelId, { revision }),
    AutoModelForCausalLM.from_pretrained(modelId, {
      revision, device: "wasm", dtype: "q8", model_file_name: "decoder_model_merged",
      progress_callback: progress => {
        if (progress.status === "progress" && progress.file.endsWith(".onnx")) {
          send({ kind: "progress", percent: Math.min(100, Math.floor(progress.progress)) });
        }
      },
    }),
  ]);
  return { tokenizer, model };
}

let engine: ReturnType<typeof loadModel> | undefined;

self.addEventListener("message", async (event: MessageEvent<PredictionRequest>) => {
  const request = event.data;
  const following = request.kind === "follow";
  const prompt = following ? request.prompt + request.completion : request.prompt;
  try {
    if (!request.prompt.trim() || request.prompt.length > MAX_PROMPT_CHARS) throw new Error("Enter a shorter prompt to continue.");
    const { tokenizer, model } = await (engine ??= loadModel().catch(error => {
      engine = undefined;
      throw error;
    }));
    const inputs = tokenizer(prompt, { add_special_tokens: false });
    const inputLength = inputs.input_ids.dims.at(-1)!;
    if (inputLength > maxInputTokens + (following ? maxNewTokens : 0)) {
      throw new Error("Shorten your original prompt so it fits the model’s input limit.");
    }
    send({ kind: "generating" });

    const decode = (ids: number[] | bigint[] | Tensor) =>
      tokenizer.decode(ids, { skip_special_tokens: true, clean_up_tokenization_spaces: false });

    async function generate(wordLimit: 1 | 2, forcedPrefix: number[] = []) {
      const scores: Float32Array[] = [];
      class CaptureScores extends LogitsProcessor {
        _call(ids: bigint[][], logits: Tensor) {
          const values = logits.data as Float32Array;
          scores.push(new Float32Array(values));
          const forcedToken = forcedPrefix[ids[0]!.length - inputLength];
          if (forcedToken !== undefined) {
            values.fill(-Infinity);
            values[forcedToken] = 0;
          }
          return logits;
        }
      }
      class StopAtWordBoundary extends StoppingCriteria {
        _call(ids: (number[] | bigint[])[]) {
          return ids.map(row => {
            const text = decode(row.slice(inputLength));
            const words = continuationWords(prompt, text, wordLimit, following);
            return words.wordCount === wordLimit && hasWordBoundary(text, words.continuation.length);
          });
        }
      }

      const processors = new LogitsProcessorList();
      processors.push(new CaptureScores());
      const stopping = new StoppingCriteriaList();
      stopping.push(new StopAtWordBoundary());
      const output = await model.generate({
        ...inputs, do_sample: false, max_new_tokens: maxNewTokens, logits_processor: processors, stopping_criteria: stopping,
      });
      if (!(output instanceof Tensor)) throw new Error("Unable to read the local prediction.");
      const ids = Array.from(output.data as BigInt64Array, Number).slice(inputLength);
      const prefixes = ids.map((_, index) => decode(ids.slice(0, index + 1)));
      const text = prefixes.at(-1) ?? "";
      const logProbabilities = ids.map((id, index) => tokenLogProbability(scores[index]!, id));
      const ended = ids.at(-1) === tokenizer.eos_token_id;
      const path = wordPath(prompt, prefixes, logProbabilities, ended, following);
      return { ids, scores, prefixes, text, path, ended };
    }

    async function findWordBranch(source: Awaited<ReturnType<typeof generate>>, at: number, wordLimit: 1 | 2) {
      const scores = source.scores[at];
      if (!scores) return;
      const prefix = source.ids.slice(0, at);
      const sameWord = (word: string) => word.toLowerCase() === source.path?.word.toLowerCase();
      const byScore = Array.from(scores.keys()).sort((a, b) => scores[b]! - scores[a]!);
      let attempts = 0;
      for (const token of byScore.slice(0, 128)) {
        if (token === source.ids[at] || !Number.isFinite(scores[token])) continue;
        const branch = [...prefix, token];
        const preview = firstContinuationWord(prompt, decode(branch), following);
        if (!preview || sameWord(preview.word)) continue;
        const alternate = await generate(wordLimit, branch);
        if (alternate.path && !sameWord(alternate.path.word)) return alternate;
        if (++attempts === 3) break;
      }
    }

    if (following) {
      const greedy = await generate(2);
      const next = greedy.path ? greedy : await findWordBranch(greedy, 0, 2);
      let continuation = next?.path?.completion ?? "";
      if (next) {
        const words = continuationWords(prompt, next.text, 2, true);
        if (words.wordCount === 2 && (next.ended || hasWordBoundary(next.text, words.continuation.length))) {
          continuation = words.continuation;
        }
      }
      send({ kind: "follow-result", continuation });
      return;
    }

    const greedy = await generate(2);
    // If greedy decoding stops at formatting, try another model-scored path that starts a word.
    const main = greedy.path ? greedy : (await findWordBranch(greedy, 0, 2)) ?? greedy;
    const first = firstContinuationWord(prompt, main.text);
    let candidates: NextWords["candidates"] = null;
    if (main.path && first) {
      const branchAt = main.prefixes.findIndex(prefix => prefix.length > first.start && main.text.startsWith(prefix));
      const alternate = await findWordBranch(main, branchAt, 1);
      if (alternate?.path) candidates = [main.path, alternate.path];
    }
    send({ kind: "result", prediction: { ...continuationWords(prompt, main.text), candidates } });
  } catch (error) {
    send({ kind: "error", message: error instanceof Error ? error.message : "Local inference failed." });
  }
});
