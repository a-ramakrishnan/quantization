import type { FormatId, QuantResult } from "@/types/quantization";
import { floatBitsFor, getFormat, packNibbles, twosComplement } from "@/utils/quantization";

export const traceNumber = (value: number) =>
  Object.is(value, -0) ? "-0" : Number(value.toPrecision(7)).toString();

function describeStoredRow(format: FormatId, values: number[]) {
  const spec = getFormat(format);
  if (format === "INT4") {
    return { valuesPerItem: 2, items: packNibbles(values).map(byte => byte.hex) };
  }

  return {
    valuesPerItem: 1,
    items: values.map(value =>
      spec.kind === "int" ? twosComplement(value, spec.bits) : floatBitsFor(format, value).hex,
    ),
  };
}

export function getGpuWeightTrace(format: FormatId, result: QuantResult, [row, column]: [number, number]) {
  const qRow = result.q[row]!;
  const code = qRow[column]!;
  const block = result.blockOf(row, column);
  const scale = result.params?.scales[block];
  const zero = result.params?.zeros[block] ?? 0;
  const payload = describeStoredRow(format, qRow);
  const stored = payload.items[Math.floor(column / payload.valuesPerItem)]!;
  const reconstructed = result.deq[row]![column]!;
  return { code, scale, zero, payload, stored, reconstructed };
}

export function reconstructionFormula({ code, scale, zero, stored, reconstructed }: ReturnType<typeof getGpuWeightTrace>) {
  return scale === undefined
    ? `${stored} → ${traceNumber(reconstructed)}`
    : `(${code} − ${zero}) × ${traceNumber(scale)} ≈ ${traceNumber(reconstructed)}`;
}
