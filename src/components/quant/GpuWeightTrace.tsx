import type { FormatId, QuantResult } from "@/types/quantization";
import { floatBitsFor, getFormat } from "@/utils/quantization";
import { getGpuWeightTrace, reconstructionFormula, traceNumber } from "./gpu-weight-trace";
import { TraceReadout } from "./TraceReadout";

type TraceStep = {
  label: string;
  value: string;
  color: "stored" | "reconstructed";
  why: string;
  action: string;
};

export function GpuWeightTrace({ chapter, progress, format, result, selected, original, grid, playing }: {
  chapter: number;
  progress: number;
  format: FormatId;
  result: QuantResult;
  selected: [number, number];
  original: number;
  grid: string;
  playing: boolean;
}) {
  const [row, column] = selected;
  const spec = getFormat(format);
  const trace = getGpuWeightTrace(format, result, selected);
  const { code, reconstructed, scale, zero, stored } = trace;
  const storedDetail = format === "INT4"
    ? `Packed byte ${stored} · tracked code ${code} is the ${column % 2 === 0 ? "high" : "low"} nibble`
    : spec.kind === "int"
      ? `${spec.label} bits ${stored} · decoded code ${code}`
      : `${spec.label} weight bits · ${stored}`;

  const hardware = ["CPU", "HBM", "L2", "Shared memory", "Registers"];
  const steps: TraceStep[] = chapter < 2
    ? hardware.map((label, i) => ({
      label, value: stored, color: "stored",
      why: i === 0 ? "The CPU has already encoded this weight for storage." : `The ${label} receives the same bits from ${hardware[i - 1]}.`,
      action: `Copy the same bits to ${label}`,
    }))
    : scale !== undefined
      ? [
        { label: format === "INT4" ? "Packed byte" : "Stored bits", value: stored, color: "stored",
          why: "The registers received the stored bits intact.", action: "Read the stored bits" },
        { label: "Read code q", value: String(code), color: "stored",
          why: storedDetail, action: "Read the integer code" },
        { label: `Subtract Z = ${zero}`, value: String(code - zero), color: "stored",
          why: `Subtract the zero-point ${zero} from code ${code}.`, action: `Subtract the zero-point ${zero}` },
        { label: "Multiply by S", value: traceNumber(reconstructed), color: "reconstructed",
          why: reconstructionFormula(trace), action: `Multiply by scale S ≈ ${traceNumber(scale)}` },
      ]
      : [
        { label: "Stored bits", value: stored, color: "stored",
          why: "The registers received the stored bits intact.", action: "Read the stored bits" },
        { label: "Read sign, exponent, fraction", value: floatBitsFor(format, code).binary, color: "stored",
          why: "Read the fields from the same encoded bits.", action: "Read the sign, exponent, and fraction" },
        { label: "Decode rounded weight", value: traceNumber(reconstructed), color: "reconstructed",
          why: reconstructionFormula(trace), action: "Decode the rounded floating-point value" },
      ];

  const last = steps.length - 1;
  const start = chapter === 1 ? 1 : 0;
  const end = chapter === 0 ? 1 : last;
  const position = start + (end - start) * progress;
  const from = Math.floor(position);
  const to = Math.min(end, from + 1);
  // Leave time after each arrival to read the new value before it moves again.
  const travel = Math.min(1, (position - from) * 1.5);
  const active = travel === 1 ? to : from;
  const x = (index: number) => 108 + 744 * index / last;
  const tokenX = x(from) + (x(to) - x(from)) * travel;
  const token = steps[active]!;
  const tokenColor = token.color === "stored" ? "var(--theater-gold)" : "var(--theater-green)";

  const next = active < end
    ? steps[active + 1]!.action
    : ["Fetch the same bits from HBM through L2.", "Decode the bits in the registers.", "Multiply by the matching activation."][chapter]!;

  return (
    <>
      <div className="stage-math-viewport mt-4">
        <svg viewBox="0 0 960 400" className="h-full w-full" role="img" aria-label="Selected weight moving through the GPU">
          <defs><pattern id={grid} width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" className="stroke-theater-grid" strokeWidth="0.5" /></pattern></defs>
          <rect width="960" height="400" fill={`url(#${grid})`} />
          <g>
            <text x="480" y="49" textAnchor="middle" className="fill-theater-ink font-serif" fontSize="30">
              {chapter === 0 ? "Copy the stored weight to HBM" : chapter === 1 ? "Follow the same bits to the SM" : "Reconstruct the weight in registers"}
            </text>
            <text x="480" y="89" textAnchor="middle" className="fill-theater-blue font-mono" fontSize="21">
              Original W[{row},{column}] = {traceNumber(original)}
            </text>
            <line x1={x(0)} x2={x(last)} y1="199" y2="199" className="stroke-theater-grid" strokeWidth="3" />
            <line x1={x(start)} x2={tokenX} y1="199" y2="199" stroke={tokenColor} strokeWidth="3" opacity="0.75" />
            {steps.map(({ label, value }, index) => (
              <g key={label} opacity={index < start || index > end ? 0.55 : 1}>
                <circle cx={x(index)} cy="199" r="7" fill={index === active ? tokenColor : "var(--theater-muted)"} />
                <text x={x(index)} y="269" textAnchor="middle" fill={index === active ? tokenColor : "var(--theater-ink)"} className="font-mono" fontSize="18">
                  {label}
                </text>
                {chapter === 2 && (
                  <text x={x(index)} y="301" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="16">{value}</text>
                )}
              </g>
            ))}
            <g transform={`translate(${tokenX} 161)`} data-trace-step={token.label}>
              <path d="M-6 23 L0 34 L6 23" fill={tokenColor} />
              <rect x="-95" y="-25" width="190" height="49" rx="8" fill="var(--theater)" stroke={tokenColor} strokeWidth="2" />
              <text y="7" textAnchor="middle" fill={tokenColor} className="font-mono" fontSize={token.value.length > 12 ? 17 : 23}>
                {token.value}
              </text>
            </g>
            <text x="480" y="357" textAnchor="middle" className="fill-theater-muted font-mono" fontSize="18">
              {chapter < 2
                ? storedDetail
                : reconstructionFormula(trace)}
            </text>
          </g>
        </svg>
      </div>
      <TraceReadout value={token.value} name={token.label} why={token.why} next={next} playing={playing} />
    </>
  );
}
