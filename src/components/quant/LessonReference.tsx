import type { FormatId } from "@/types/quantization";
import { getFormat, storageGB } from "@/utils/quantization";

const MODEL_WEIGHTS = 9_000_000_000;
const GROUP_SIZE = 128;

export function ModelStorage({ initial, formats, selected }: {
  initial: "FP32" | "BF16";
  formats: readonly FormatId[];
  selected: FormatId;
}) {
  const baseline = storageGB(MODEL_WEIGHTS, initial === "FP32" ? 32 : 16);
  const rows = [
    { label: `Initial · ${initial}`, size: baseline, initial: true, selected: false },
    ...formats.map(format => {
      const spec = getFormat(format);
      return {
        label: `Final · ${spec.label}`,
        size: storageGB(MODEL_WEIGHTS, spec.bits, spec.kind === "int" ? GROUP_SIZE : undefined),
        initial: false,
        selected: format === selected,
      };
    }),
  ];

  return (
    <section className="lesson-reference" aria-label="Model weight storage">
      <h3>Model weight storage</h3>
      <p>Weights and scales for a 9B-parameter model. Integer estimates use one FP16 scale per 128 weights and an implicit zero-point.</p>
      <ul className="lesson-size-list">
        {rows.map(row => (
          <li key={row.label} data-initial={row.initial} data-selected={row.selected}>
            <div className="lesson-size-label">
              <span>{row.label}{row.selected && formats.length > 1 && <span className="lesson-size-selected"> · selected</span>}</span>
              <strong>{row.size.toFixed(2)} GB</strong>
            </div>
            <div className="lesson-size-track" aria-hidden="true">
              <span style={{ width: `${row.size / baseline * 100}%` }} />
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

const SYMBOLS = [
  ["W, w", "Original weight matrix and one weight", "original"],
  ["S", "Scale, or distance between grid values", ""],
  ["Z", "Zero-point, the integer code for zero", ""],
  ["q", "Stored integer code for a weight", "code"],
  ["ŵ", "Reconstructed weight", "reconstructed"],
  ["x", "Input activation vector", ""],
  ["Sₓ", "Scale for converting input activation codes", ""],
  ["y, ŷ", "Original and reconstructed outputs", ""],
  ["Σⱼ", "Add the contributions from each column j", ""],
  ["Δ", "Original minus reconstructed value", "error"],
  ["≈", "Approximately equal", ""],
];

export function SymbolLegend() {
  return (
    <section className="lesson-reference" aria-label="Symbol legend">
      <h3>Symbol legend</h3>
      <dl className="lesson-symbols">
        {SYMBOLS.map(([symbol, meaning, color]) => (
          <div key={symbol}><dt data-role={color}>{symbol}</dt><dd>{meaning}</dd></div>
        ))}
      </dl>
    </section>
  );
}
