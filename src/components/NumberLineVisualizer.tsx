import React from 'react';
import { CellResult, GroupQuantParams } from '../types/quantization';

interface NumberLineVisualizerProps {
  selectedCell: CellResult;
  groupParams: GroupQuantParams;
  allCellsInGroup: CellResult[];
}

export const NumberLineVisualizer: React.FC<NumberLineVisualizerProps> = ({
  selectedCell,
  groupParams,
  allCellsInGroup,
}) => {
  const { qMin, qMax, scale, zeroPoint, clippedMin, clippedMax } = groupParams;
  const minDequant = scale * (qMin - zeroPoint);
  const maxDequant = scale * (qMax - zeroPoint);

  const allFloats = [
    minDequant,
    maxDequant,
    clippedMin,
    clippedMax,
    ...allCellsInGroup.map((c) => c.wFp32),
  ];
  const rawMin = Math.min(...allFloats);
  const rawMax = Math.max(...allFloats);
  const span = Math.max(rawMax - rawMin, 0.2);
  const domainMin = rawMin - span * 0.08;
  const domainMax = rawMax + span * 0.08;

  const width = 680;
  const height = 165;
  const padX = 42;
  const innerW = width - padX * 2;
  const axisY = 92;

  const toX = (val: number) =>
    padX + ((val - domainMin) / (domainMax - domainMin)) * innerW;

  // Build integer grid ticks; if too many (> 33), sample nicely while keeping qMin, qMax, zeroPoint, selectedCell.wq
  const totalLevels = qMax - qMin + 1;
  const tickIntegers: number[] = [];
  if (totalLevels <= 33) {
    for (let q = qMin; q <= qMax; q++) tickIntegers.push(q);
  } else {
    const step = Math.ceil(totalLevels / 16);
    const set = new Set<number>([qMin, qMax, zeroPoint, selectedCell.wq]);
    for (let q = qMin; q <= qMax; q += step) set.add(q);
    tickIntegers.push(...Array.from(set).sort((a, b) => a - b));
  }

  const selFp32X = toX(selectedCell.wFp32);
  const selDequantX = toX(selectedCell.wDequant);
  const zeroFloatX = toX(scale * (0 - zeroPoint));

  return (
    <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 lg:p-5 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Number-Line Bin Mapping &amp; Snapping ({groupParams.label})
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Shows how continuous FP32 weights snap to discrete integer levels separated by step size Δ = S ={' '}
            <span className="font-mono text-slate-700 dark:text-slate-200">
              {scale.toFixed(4)}
            </span>
            .
          </p>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono">
          <span className="inline-flex items-center gap-1.5 text-blue-600 dark:text-blue-400">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block" />
            FP32 (W)
          </span>
          <span className="inline-flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
            <span className="w-2.5 h-2.5 rounded-sm bg-amber-500 inline-block" />
            Dequantized (Ŵ)
          </span>
        </div>
      </div>

      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full min-w-[540px] h-auto select-none"
        >
          {/* Clipped Range Highlight */}
          <rect
            x={Math.min(toX(clippedMin), toX(clippedMax))}
            y={axisY - 12}
            width={Math.abs(toX(clippedMax) - toX(clippedMin))}
            height={24}
            rx={6}
            className="fill-indigo-500/10 stroke-indigo-500/25"
            strokeWidth="1"
          />

          {/* Main continuous axis */}
          <line
            x1={padX - 12}
            y1={axisY}
            x2={width - padX + 12}
            y2={axisY}
            className="stroke-slate-300 dark:stroke-slate-700"
            strokeWidth="2"
          />

          {/* Zero-point / 0.0 marker */}
          {zeroFloatX >= padX - 10 && zeroFloatX <= width - padX + 10 && (
            <g>
              <line
                x1={zeroFloatX}
                y1={axisY - 18}
                x2={zeroFloatX}
                y2={axisY + 18}
                className="stroke-emerald-500/70"
                strokeDasharray="3 3"
                strokeWidth="1.5"
              />
              <text
                x={zeroFloatX}
                y={axisY + 32}
                textAnchor="middle"
                className="fill-emerald-600 dark:fill-emerald-400 text-[9px] font-mono"
              >
                0.0 (Z={zeroPoint})
              </text>
            </g>
          )}

          {/* Discrete integer ticks */}
          {tickIntegers.map((q) => {
            const x = toX(scale * (q - zeroPoint));
            const isSelQ = q === selectedCell.wq;
            return (
              <g key={q}>
                <line
                  x1={x}
                  y1={axisY - (isSelQ ? 12 : 7)}
                  x2={x}
                  y2={axisY + (isSelQ ? 12 : 7)}
                  className={
                    isSelQ
                      ? 'stroke-amber-500'
                      : 'stroke-slate-400 dark:stroke-slate-600'
                  }
                  strokeWidth={isSelQ ? 2.5 : 1.2}
                />
                {(totalLevels <= 17 || isSelQ || q === qMin || q === qMax) && (
                  <text
                    x={x}
                    y={axisY + 22}
                    textAnchor="middle"
                    className={`text-[8.5px] font-mono ${
                      isSelQ
                        ? 'fill-amber-600 dark:fill-amber-400 font-bold'
                        : 'fill-slate-400 dark:fill-slate-500'
                    }`}
                  >
                    {q}
                  </text>
                )}
              </g>
            );
          })}

          {/* Other group weights as subtle dots */}
          {allCellsInGroup.map((c) => {
            if (c.row === selectedCell.row && c.col === selectedCell.col) {
              return null;
            }
            return (
              <circle
                key={`dot-${c.row}-${c.col}`}
                cx={toX(c.wFp32)}
                cy={axisY - 22}
                r={3.2}
                className="fill-slate-400/70 dark:fill-slate-500/70"
              />
            );
          })}

          {/* Selected weight snap arc: FP32 -> Dequantized */}
          <path
            d={`M ${selFp32X} ${axisY - 34} Q ${(selFp32X + selDequantX) / 2} ${
              axisY - 68
            } ${selDequantX} ${axisY - 6}`}
            fill="none"
            className="stroke-indigo-500 dark:stroke-indigo-400"
            strokeWidth="2"
            strokeDasharray="4 3"
          />

          {/* Selected FP32 dot */}
          <circle
            cx={selFp32X}
            cy={axisY - 34}
            r={6}
            className="fill-blue-500 stroke-white dark:stroke-slate-900"
            strokeWidth="2"
          />
          <text
            x={selFp32X}
            y={axisY - 46}
            textAnchor="middle"
            className="fill-blue-600 dark:fill-blue-400 text-[10px] font-mono font-bold"
          >
            W={selectedCell.wFp32.toFixed(2)}
          </text>

          {/* Selected Dequantized square */}
          <rect
            x={selDequantX - 5}
            y={axisY - 5}
            width={10}
            height={10}
            rx={2}
            className="fill-amber-500 stroke-white dark:stroke-slate-900"
            strokeWidth="1.5"
          />
          <text
            x={selDequantX}
            y={height - 12}
            textAnchor="middle"
            className="fill-amber-600 dark:fill-amber-400 text-[10px] font-mono font-bold"
          >
            Ŵ={selectedCell.wDequant.toFixed(3)} (q={selectedCell.wq})
          </text>
        </svg>
      </div>

      <p className="text-xs text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-800/50 rounded-lg px-3 py-2 border border-slate-200/80 dark:border-slate-800">
        <strong>How to read this:</strong> The shaded band is the calibrated range{' '}
        <span className="font-mono">
          [{clippedMin.toFixed(2)}, {clippedMax.toFixed(2)}]
        </span>
        . Every tick mark is a representable integer level{' '}
        <span className="font-mono">q</span>. Your selected weight{' '}
        <span className="font-mono text-blue-600 dark:text-blue-400">
          {selectedCell.wFp32.toFixed(2)}
        </span>{' '}
        snaps to{' '}
        <span className="font-mono text-amber-600 dark:text-amber-400">
          q = {selectedCell.wq}
        </span>
        , dequantizing to{' '}
        <span className="font-mono">{selectedCell.wDequant.toFixed(3)}</span>{' '}
        (error ={' '}
        <span className="font-mono">{selectedCell.error.toFixed(4)}</span>).
      </p>
    </section>
  );
};
