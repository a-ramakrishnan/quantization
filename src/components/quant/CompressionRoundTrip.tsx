import { useMemo, useState } from "react";
import type { basicProjection } from "@/utils/basic-quantization";
import { decodeInt4Prefix, encodeInt4Prefix } from "@/utils/prefix-compression";
import { matVec } from "@/utils/quantization";
import { traceNumber } from "./gpu-weight-trace";
import { INPUT } from "./lesson-example";
import { TraceReadout } from "./TraceReadout";
import { useLessonPlayback } from "./useLessonPlayback";

const phases = ["Count codes", "Build tree", "Encode", "Decode", "Check output"];
const phaseSeconds = 4;

function PrefixBranches({ symbols, prefix = "", activePath }: {
  symbols: ReturnType<typeof encodeInt4Prefix>["symbols"];
  prefix?: string;
  activePath?: string;
}) {
  const leaf = symbols.find(symbol => symbol.path === prefix);
  if (leaf) return <span className="compression-leaf" data-active={prefix === activePath}>{leaf.code}</span>;
  return (
    <ul>
      {["0", "1"].map(bit => <li key={bit}>
        <span className="compression-edge" data-active={activePath?.startsWith(prefix + bit)}>{bit}</span>
        <PrefixBranches symbols={symbols} prefix={prefix + bit} activePath={activePath} />
      </li>)}
    </ul>
  );
}

export function CompressionRoundTrip({ projection, mask, sparseOutput }: {
  projection: ReturnType<typeof basicProjection>;
  mask: boolean[];
  sparseOutput: number;
}) {
  const [copies, setCopies] = useState(1);
  const playback = useLessonPlayback(phases.length, true, 1, phaseSeconds);
  const { chapter, chapterTime, duration, time, playing, reducedMotion, seek } = playback;
  const scale = projection.result.params!.scales[0]!;
  const { codes, encoded, decoded, decodedOutput, exact } = useMemo(() => {
    const codes = projection.result.q.flat().filter((_, index) => mask[index]);
    const stream = Array.from({ length: copies }, () => codes).flat();
    const encoded = encodeInt4Prefix(stream);
    const decoded = decodeInt4Prefix(encoded.treeBytes, encoded.payloadBytes, stream.length);
    let offset = 0;
    const row = mask.slice(0, INPUT.length).map(keep => keep ? decoded[offset++]! * scale : 0);
    return { codes, encoded, decoded, decodedOutput: matVec([row], INPUT)[0]!,
      exact: stream.every((code, index) => code === decoded[index]) };
  }, [copies, mask, projection, scale]);

  const cursor = Math.min(codes.length - 1, Math.floor(chapterTime / phaseSeconds * codes.length));
  const code = chapter === 3 ? decoded[cursor] : codes[cursor];
  const path = encoded.words[cursor] || "∅";
  const maskBytes = Math.ceil(mask.length * copies / 8);
  const treeBytes = encoded.treeBytes.length;
  const payloadBytes = encoded.payloadBytes.length;
  const loadedBytes = Math.ceil(codes.length * copies / 2) + maskBytes + projection.scaleBytes;
  const fileBytes = treeBytes + payloadBytes + maskBytes + projection.scaleBytes;
  const readouts = [
    { value: `${codes.length * copies}`, name: "kept INT4 codes", why: "The mask omits pruned positions. The histogram counts the remaining codes.", next: "Merge the two least frequent groups until they form one tree." },
    { value: `${encoded.treeBits.length} bits`, name: "serialized tree", why: "Split each branch into a 0 edge and a 1 edge. Each leaf gives one complete codeword.", next: "Replace each kept INT4 code with the bits on its path." },
    { value: `${code} → ${path}`, name: "append a codeword", why: "The leaf's path determines the bits. A one-symbol tree needs no payload bits.", next: "Read the tree bytes, then follow the payload bits from the root." },
    { value: `${path} → ${code}`, name: "recover one INT4 code", why: "A leaf identifies the code and ends the current path. The decoder returns to the root.", next: "Use the mask to restore pruned positions and multiply the codes by the scale." },
    { value: traceNumber(decodedOutput), name: "decoded row 1 output", why: "The recovered codes, original mask, and original scale produce the same pruned INT4 weights.", next: "Compare the two sizes below and try repeating the block to share the tree." },
  ];
  const current = readouts[chapter]!;

  return (
    <div className="compression-round-trip">
      <h4>Can we compress the same weights twice?</h4>
      <p>Pack the kept codes with a Huffman tree, then decode the bytes and run the same dot product.</p>
      <div className="experiment-buttons" role="group" aria-label="Weight blocks sharing one coding tree">
        {[1, 16].map(count => <button type="button" key={count} aria-pressed={copies === count}
          onClick={() => { setCopies(count); seek(0); }}>{count === 1 ? "One weight block" : "Repeat the block 16 times"}</button>)}
      </div>
      <div className="compression-stage">
        <div className="experiment-buttons" role="group" aria-label="Compression stages">
          {phases.map((phase, index) => <button key={phase} type="button" aria-pressed={chapter === index}
            onClick={() => seek(index * phaseSeconds)}>{phase}</button>)}
        </div>
        {chapter === 0 ? (
          <ul className="compression-histogram" aria-label="Kept code frequencies">
            {encoded.symbols.map(({ code, count }) => (
              <li key={code}><strong>{code}</strong><span className="compression-frequency" aria-hidden="true"
                style={{ width: `${100 * count / encoded.symbols[0]!.count}%` }} />
                <span>×{count}</span>
              </li>
            ))}
          </ul>
        ) : null}
        {chapter >= 1 && chapter <= 3 && (
          <div className="compression-tree" role="group" aria-label="Huffman tree with bit-labeled edges">
            <span>Root</span>
            <PrefixBranches symbols={encoded.symbols} activePath={chapter === 1 ? undefined : encoded.words[cursor]} />
          </div>
        )}
        {chapter >= 2 && (
          <ol className="compression-stream" aria-label={chapter === 2 ? "Encoded codewords for the first block" : "Decoded codes for the first block"}>
            {codes.map((_, index) => (
              <li key={index} data-active={chapter !== 4 && index === cursor} data-pending={chapter !== 4 && index > cursor}>
                <strong>{chapter === 2 ? encoded.words[index] || "∅" : decoded[index]}</strong>
                <small>{chapter === 2 ? `code ${codes[index]}` : encoded.words[index] || "∅"}</small>
              </li>
            ))}
          </ol>
        )}
        <TraceReadout {...current} playing={playing} />
        <div className="experiment-buttons" role="group" aria-label="Compression playback">
          <button type="button" disabled={reducedMotion} onClick={playback.togglePlayback}>
            {playing ? "Pause compression" : "Play compression"}
          </button>
          <button type="button" disabled={chapter === 0} onClick={() => seek((chapter - 1) * phaseSeconds)}>Previous step</button>
          <button type="button" disabled={chapter === phases.length - 1} onClick={() => seek((chapter + 1) * phaseSeconds)}>Next step</button>
        </div>
        <label className="experiment-range">
          <span>Compression progress</span>
          <input type="range" min={0} max={duration} step={playback.scrubStep} value={time}
            aria-valuetext={phases[chapter]} onChange={event => seek(Number(event.target.value))} />
        </label>
      </div>
      <dl className="compression-sizes">
        <div><dt>Entropy-coded file payload</dt><dd>{fileBytes} B</dd></div>
        <div><dt>Loaded as packed sparse INT4</dt><dd>{loadedBytes} B</dd></div>
      </dl>
      <p className="experiment-equation">{encoded.payloadBits.length} payload bits → {payloadBytes} B + {treeBytes} B tree + {maskBytes} B mask + {projection.scaleBytes} B scale</p>
      <p>The payload and tree are padded separately. The example assumes a known block length and layout. The mask tells the decoder how many codes to read.</p>
      {chapter === 4 && <>
        <dl className="compression-sizes" aria-label="Row 1 output comparison">
          <div><dt>Original BF16</dt><dd>{traceNumber(projection.baseline[0]!)}</dd></div>
          <div><dt>Pruned INT4</dt><dd>{traceNumber(sparseOutput)}</dd></div>
          <div><dt>Decoded pruned INT4</dt><dd>{traceNumber(decodedOutput)}</dd></div>
        </dl>
        <p>{exact && sparseOutput === decodedOutput ? "All kept codes and the pruned INT4 output match exactly." : "The round trip did not match."}</p>
      </>}
      <p>GPU memory also includes the KV cache and runtime allocations. <a href="?view=kv-cache">Estimate total GPU memory</a>.</p>
    </div>
  );
}
