import { cn } from "@/lib/utils";
import type { Matrix } from "@/types/quantization";

export function heatStyle(v: number, maxAbs: number) {
  const t = Math.min(1, Math.abs(v) / (maxAbs || 1));
  const c = v >= 0 ? "var(--heat-pos)" : "var(--heat-neg)";
  return { backgroundColor: `color-mix(in oklab, ${c} ${Math.round(t * 38)}%, transparent)` };
}

export function MatrixGrid({
  m,
  editable,
  onChange,
  selected,
  onSelect,
  format = (v) => v.toFixed(3),
  label,
}: {
  m: Matrix;
  editable?: boolean;
  onChange?: (r: number, c: number, v: number) => void;
  selected?: [number, number];
  onSelect?: (r: number, c: number) => void;
  format?: (v: number) => string;
  label?: string;
}) {
  const maxAbs = Math.max(...m.flat().map(Math.abs));
  return (
    <div>
      {label && <div className="mb-2 font-mono text-xs text-muted-foreground">{label}</div>}
      <div className="grid grid-cols-4 gap-1.5 rounded-lg border bg-card p-2">
        {m.map((row, r) =>
          row.map((v, c) => {
            const isSel = selected?.[0] === r && selected?.[1] === c;
            return (
              <div
                key={`${r}-${c}`}
                style={heatStyle(v, maxAbs)}
                onClick={() => onSelect?.(r, c)}
                className={cn(
                  "rounded-md transition-all",
                  onSelect && "cursor-pointer",
                  isSel && "ring-2 ring-primary ring-offset-1 ring-offset-card",
                )}
              >
                {editable ? (
                  <input
                    type="number"
                    step="0.1"
                    value={v}
                    onFocus={() => onSelect?.(r, c)}
                    onChange={(e) => onChange?.(r, c, parseFloat(e.target.value) || 0)}
                    className="w-full bg-transparent py-2 text-center font-mono text-sm outline-none focus:ring-2 focus:ring-ring rounded-md [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none"
                    aria-label={`W[${r}][${c}]`}
                  />
                ) : (
                  <div className={cn("py-2 text-center font-mono text-sm")}>{format(v)}</div>
                )}
              </div>
            );
          }),
        )}
      </div>
    </div>
  );
}
