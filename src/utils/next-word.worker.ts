import {
  AutoModelForCausalLM, AutoTokenizer, env, LogitsProcessor, LogitsProcessorList,
  StoppingCriteria, StoppingCriteriaList, Tensor,
} from "@huggingface/transformers";
import { continuationWords, MAX_PROMPT_CHARS } from "./next-word";
import type { NextWords, PredictionMessage } from "./next-word";

const modelId = "Xenova/distilgpt2";
const revision = "a41c10485c18a64b6606729b6a082330cbd8f49e";
const maxInputTokens = 512;

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

self.addEventListener("message", async (event: MessageEvent<string>) => {
  const prompt = event.data;
  try {
    if (!prompt.trim() || prompt.length > MAX_PROMPT_CHARS) throw new Error("Enter a shorter prompt to continue.");
    const { tokenizer, model } = await (engine ??= loadModel().catch(error => {
      engine = undefined;
      throw error;
    }));
    const inputs = tokenizer(prompt, { add_special_tokens: false });
    const inputLength = inputs.input_ids.dims.at(-1)!;
    if (inputLength > maxInputTokens) throw new Error("This prompt exceeds 512 model tokens. Shorten it and try again.");
    send({ kind: "generating" });

    const decode = (ids: number[] | bigint[] | Tensor) =>
      tokenizer.decode(ids, { skip_special_tokens: true, clean_up_tokenization_spaces: false });
    let candidates: NextWords["candidates"] | undefined;
    class CaptureFirstScores extends LogitsProcessor {
      _call(_ids: bigint[][], logits: Tensor) {
        if (candidates) return logits;
        let best = -1;
        let second = -1;
        const scores = logits.data;
        for (let id = 0; id < scores.length; id++) {
          const score = Number(scores[id]);
          if (!Number.isFinite(score)) continue;
          if (best === -1 || score > Number(scores[best])) {
            second = best;
            best = id;
          } else if (second === -1 || score > Number(scores[second])) {
            second = id;
          }
        }
        if (second === -1) throw new Error("The model returned fewer than two candidate tokens.");
        const token = (id: number) => ({
          text: tokenizer.decode([id], { clean_up_tokenization_spaces: false }),
          logit: Number(scores[id]),
        });
        candidates = [token(best), token(second)];
        return logits;
      }
    }
    class StopAfterWords extends StoppingCriteria {
      _call(ids: number[][]) {
        return ids.map(row => {
          const text = decode(row.slice(inputLength));
          const words = continuationWords(prompt, text);
          // Wait for a boundary beyond the second word so its last subword is included.
          return words.wordCount === 2 && words.continuation.length < text.length;
        });
      }
    }
    const processors = new LogitsProcessorList();
    processors.push(new CaptureFirstScores());
    const stopping = new StoppingCriteriaList();
    stopping.push(new StopAfterWords());
    const output = await model.generate({
      ...inputs, do_sample: false, max_new_tokens: 24, logits_processor: processors, stopping_criteria: stopping,
    });
    if (!(output instanceof Tensor) || !candidates) throw new Error("Unable to read the local prediction.");
    const text = decode(Array.from(output.data as BigInt64Array).slice(inputLength));
    send({ kind: "result", prediction: { ...continuationWords(prompt, text), candidates } });
  } catch (error) {
    send({ kind: "error", message: error instanceof Error ? error.message : "Local inference failed." });
  }
});
