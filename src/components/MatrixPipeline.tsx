import React from 'react';
import { CellResult, Granularity, QuantConfig } from '../types/quantization';

interface MatrixPipelineProps {
  weights: number[][];
  result: QuantizationResultCompat;
  config: QuantConfig;
  selectedPos: { row: number; col: number };
  onSelectCell: (row: number, col: number) => void;
  onUpdateWeight: (row: number, col: number, value: number) => void;
  activeStep: number;
}

// Local structural type to avoid circular/unused issues
type QuantizationResultCompat = {
  cells: CellResult[][];
  qMin: number;
  qMax: number;
};

function isSameGroup(
  r: number,
  c: number,
  sel: { row: number; col: number },
  granularity: Granularity
) {
  if (granularity === 'per-tensor') return true;
  if (granularity === 'per-channel') return r === sel.row;
  return r === sel.row && Math.floor(c / 2) === Math.floor(sel.col / 2);
}

export const MatrixPipeline: React.FC<MatrixPipelineProps> = ({
  weights,
  result,
  config,
  selectedPos,
  onSelectCell,
  onUpdateWeight,
  activeStep,
}) => {
  const flatWeights = weights.flat();
  const maxAbsW = Math.max(...flatWeights.map((v) => Math.abs(v)), 0.01);
  const maxAbsErr = Math.max(
    ...result.cells.flat().map((c) => c.absError),
    0.001
  );

  const valueBg = (val: number, maxAbs: number) => {
    const ratio = Math.min(Math.abs(val) / maxAbs, 1);
    const alpha = (ratio * 0.28).toFixed(2);
    if (val >= 0) return `rgba(59, 130, 246, ${alpha})`; // blue
    return `rgba(244, 63, 94, ${alpha})`; // rose
  };

  const errorBg = (err: number) => {
    const ratio = Math.min(Math.abs(err) / maxAbsErr, 1);
    const alpha = (ratio * 0.42).toFixed(2);
    return `rgba(245, 158, 11, ${alpha})`; // amber for error intensity
  };

  const cellRing = (r: number, c: number) => {
    const isSelected = r === selectedPos.row && c === selectedPos.col;
    const inGroup = isSameGroup(r, c, selectedPos, config.granularity);
    if (isSelected) {
      return 'ring-2 ring-indigo-500 dark:ring-indigo-400 z-10';
    }
    if (inGroup && config.granularity !== 'per-tensor') {
      return 'ring-1 ring-indigo-400/50 dark:ring-indigo-500/40';
    }
    return '';
  };

  return (
    <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 lg:p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Side-by-Side Weight Transformation Pipeline (4×4 Linear Layer)
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Edit any FP32 weight in <strong className="text-slate-700 dark:text-slate-200">W</strong> or click any cell to inspect its exact math &amp; GPU register values.
          </p>
        </div>
        <span className="text-xs font-mono px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
          Selected: [{selectedPos.row}, {selectedPos.col}]
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {/* 1. FP32 Weights W */}
        <div
          className={`rounded-xl border p-3 transition ${
            activeStep === 0
              ? 'border-indigo-500/70 bg-indigo-50/30 dark:bg-indigo-950/20'
              : 'border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/40'
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-800 dark:text-slate-200">
              1. FP32 Weights (W)
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-200/80 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
              Editable
            </span>
          </div>
          <div className="grid grid-cols-4 gap-1.5">
            {weights.map((rowArr, r) =>
              rowArr.map((w, c) => (
                <div
                  key={`fp32-${r}-${c}`}
                  onClick={() => onSelectCell(r, c)}
                  style={{ backgroundColor: valueBg(w, maxAbsW) }}
                  className={`relative rounded-lg border border-slate-200 dark:border-slate-700/80 p-1 cursor-pointer transition ${cellRing(
                    r,
                    c
                  )}`}
                >
                  <input
                    type="number"
                    step="0.05"
                    value={w}
                    onFocus={() => onSelectCell(r, c)}
                    onChange={(e) => {
                      const parsed = parseFloat(e.target.value);
                      if (!Number.isNaN(parsed)) {
                        onUpdateWeight(r, c, parsed);
                      }
                    }}
                    aria-label={`Weight row ${r} col ${c}`}
                    className="w-full bg-transparent text-center font-mono text-xs font-medium text-slate-900 dark:text-slate-100 focus:outline-none"
                  />
                </div>
              ))
            )}
          </div>
          <p className="mt-2.5 text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
            Original 32-bit floats. Positive values shown in blue, negative in rose.
          </p>
        </div>

        {/* 2. Quantized Integers W_q */}
        <div
          className={`rounded-xl border p-3 transition ${
            activeStep === 1 || activeStep === 2 || activeStep === 3
              ? 'border-indigo-500/70 bg-indigo-50/30 dark:bg-indigo-950/20'
              : 'border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/40'
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-800 dark:text-slate-200">
              2. Quantized (W_q)
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-700 dark:text-emerald-300">
              INT{config.bits} [{result.qMin}..{result.qMax}]
            </span>
          </div>
          <div className="grid grid-cols-4 gap-1.5">
            {result.cells.map((rowArr, r) =>
              rowArr.map((cell, c) => (
                <button
                  type="button"
                  key={`wq-${r}-${c}`}
                  onClick={() => onSelectCell(r, c)}
                  style={{
                    backgroundColor: valueBg(
                      cell.wq - cell.zeroPoint,
                      Math.max(Math.abs(result.qMax), Math.abs(result.qMin))
                    ),
                  }}
                  className={`relative rounded-lg border border-slate-200 dark:border-slate-700/80 py-1.5 px-1 text-center font-mono text-xs font-semibold text-slate-900 dark:text-slate-100 transition ${cellRing(
                    r,
                    c
                  )}`}
                >
                  {cell.wq}
                  {cell.isClipped && (
                    <span
                      title="Clipped to range!"
                      className="absolute top-0.5 right-0.5 w-1.5 h-1.5 rounded-full bg-amber-500"
                    />
                  )}
                </button>
              ))
            )}
          </div>
          <p className="mt-2.5 text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
            Stored in GPU HBM. Amber dot indicates the weight was clipped to [{result.qMin}, {result.qMax}].
          </p>
        </div>

        {/* 3. Dequantized Weights W_hat */}
        <div
          className={`rounded-xl border p-3 transition ${
            activeStep === 4
              ? 'border-indigo-500/70 bg-indigo-50/30 dark:bg-indigo-950/20'
              : 'border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/40'
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-800 dark:text-slate-200">
              3. Dequantized (Ŵ)
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-300">
              In Registers
            </span>
          </div>
          <div className="grid grid-cols-4 gap-1.5">
            {result.cells.map((rowArr, r) =>
              rowArr.map((cell, c) => (
                <button
                  type="button"
                  key={`wd-${r}-${c}`}
                  onClick={() => onSelectCell(r, c)}
                  style={{ backgroundColor: valueBg(cell.wDequant, maxAbsW) }}
                  className={`rounded-lg border border-slate-200 dark:border-slate-700/80 py-1.5 px-1 text-center font-mono text-xs font-medium text-slate-900 dark:text-slate-100 transition ${cellRing(
                    r,
                    c
                  )}`}
                >
                  {cell.wDequant.toFixed(2)}
                </button>
              ))
            )}
          </div>
          <p className="mt-2.5 text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
            Reconstructed via Ŵ = S·(W_q − Z) in SM registers just before Tensor Core matmul.
          </p>
        </div>

        {/* 4. Quantization Error W - W_hat */}
        <div
          className={`rounded-xl border p-3 transition ${
            activeStep === 5
              ? 'border-indigo-500/70 bg-indigo-50/30 dark:bg-indigo-950/20'
              : 'border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/40'
          }`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-800 dark:text-slate-200">
              4. Error (W − Ŵ)
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-700 dark:text-rose-300">
              Loss / Drift
            </span>
          </div>
          <div className="grid grid-cols-4 gap-1.5">
            {result.cells.map((rowArr, r) =>
              rowArr.map((cell, c) => (
                <button
                  type="button"
                  key={`err-${r}-${c}`}
                  onClick={() => onSelectCell(r, c)}
                  style={{ backgroundColor: errorBg(cell.error) }}
                  className={`rounded-lg border border-slate-200 dark:border-slate-700/80 py-1.5 px-1 text-center font-mono text-xs font-medium text-slate-900 dark:text-slate-100 transition ${cellRing(
                    r,
                    c
                  )}`}
                >
                  {cell.error >= 0 ? '+' : ''}
                  {cell.error.toFixed(2)}
                </button>
              ))
            )}
          </div>
          <p className="mt-2.5 text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
            Difference between true W and dequantized Ŵ. Brighter amber = larger error.
          </p>
        </div>
      </div>
    </section>
  );
};
