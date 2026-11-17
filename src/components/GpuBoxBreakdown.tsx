import React, { useState } from 'react';
import { ArrowRight, Box, CheckCircle2, Layers } from 'lucide-react';
import {
  CellResult,
  QuantConfig,
  QuantizationResult,
} from '../types/quantization';

interface GpuBoxBreakdownProps {
  config: QuantConfig;
  result: QuantizationResult;
  selectedCell: CellResult;
  inputs: number[];
  onReconstructAll: () => void;
}

export const GpuBoxBreakdown: React.FC<GpuBoxBreakdownProps> = ({
  config,
  result,
  selectedCell,
  inputs,
  onReconstructAll,
}) => {
  const [activeBox, setActiveBox] = useState<number>(0);
  const rowIdx = selectedCell.row;
  const rowCells = result.cells[rowIdx];

  const boxes = [
    {
      id: 'host',
      badge: 'Box 1 of 5',
      title: 'Host CPU / Calibrator (Independent Step)',
      whatInside: 'Takes FP32 Weight Row W and computes Scale (S), Zero-Point (Z), and Integer W_q.',
      tensorBefore: `W[${rowIdx},:] = [${rowCells.map((c) => c.wFp32.toFixed(2)).join(', ')}] (FP32, 128 bits)`,
      tensorAfter: `W_q[${rowIdx},:] = [${rowCells.map((c) => c.wq).join(', ')}] (INT${config.bits}, ${4 * config.bits} bits)`,
      detail: `Scale S = ${selectedCell.scale.toFixed(4)}, Zero-Point Z = ${selectedCell.zeroPoint}. Only the integers and tiny metadata will be sent over PCIe.`,
    },
    {
      id: 'hbm',
      badge: 'Box 2 of 5',
      title: 'GPU HBM / VRAM (Global Memory)',
      whatInside: 'Stores packed integer weights W_q + (S, Z) and input activations X.',
      tensorBefore: `FP32 would need ${result.memory.fp32Bits} bits in HBM`,
      tensorAfter: `Stored: ${result.memory.quantizedWeightBits}b (W_q) + ${result.memory.metadataBits}b (S,Z) = ${result.memory.totalQuantizedBits} bits`,
      detail: `HBM bandwidth (~2–3 TB/s) is the main bottleneck in linear layers. Storing INT${config.bits} achieves ${result.memory.compressionRatio.toFixed(2)}× compression.`,
    },
    {
      id: 'sram',
      badge: 'Box 3 of 5',
      title: 'L2 Cache → SM Shared Memory (SRAM)',
      whatInside: 'Streams compressed tiles of W_q, (S, Z), and X onto the Streaming Multiprocessor.',
      tensorBefore: `HBM Tile: Packed INT${config.bits} bytes`,
      tensorAfter: `SRAM Tile: W_q[${rowIdx},:] = [${rowCells.map((c) => c.wq).join(', ')}], X = [${inputs.map((v) => v.toFixed(1)).join(', ')}]`,
      detail: 'On-chip SRAM runs at ~20+ TB/s. Packing weights lets each HBM fetch bring in more weights per byte.',
    },
    {
      id: 'registers',
      badge: 'Box 4 of 5',
      title: 'SM Register File (On-the-Fly Dequantizer)',
      whatInside: 'Unpacks W_q into registers and reconstructs Ŵ = S · (W_q − Z) right before math.',
      tensorBefore: `W_q[${rowIdx},:] = [${rowCells.map((c) => c.wq).join(', ')}] (INT${config.bits})`,
      tensorAfter: `Ŵ[${rowIdx},:] = [${rowCells.map((c) => c.wDequant.toFixed(2)).join(', ')}] (Float in Registers)`,
      detail: 'Crucial: Ŵ lives only in registers and is NEVER written back to HBM!',
    },
    {
      id: 'tensor_cores',
      badge: 'Box 5 of 5',
      title: 'Tensor Cores & FP32 Accumulators',
      whatInside: 'Computes dot product ŷ_i = Σ Ŵ_{i,j} · x_j in FP32 and writes ŷ to HBM.',
      tensorBefore: `Ŵ[${rowIdx},:] · X = ${rowCells.map((c, i) => `(${c.wDequant.toFixed(2)}×${inputs[i].toFixed(1)})`).join(' + ')}`,
      tensorAfter: `ŷ[${rowIdx}] = ${result.yQuant[rowIdx].toFixed(3)} (True y[${rowIdx}] = ${result.yTrue[rowIdx].toFixed(3)})`,
      detail: `Quantization error on this output row is ${(result.yTrue[rowIdx] - result.yQuant[rowIdx]).toFixed(4)}.`,
    },
  ];

  const current = boxes[activeBox];

  return (
    <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 sm:p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">
            <Box className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Independent GPU Box Breakdown (Step-by-Step)
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Inspect how your row <span className="font-mono">[{rowIdx},:]</span> transforms inside each box independently, then reconstruct the full GPU diagram.
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onReconstructAll}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-xs transition"
        >
          <Layers className="w-3.5 h-3.5" />
          Reconstruct All Boxes Together
        </button>
      </div>

      {/* Box Selector Pills */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
        {boxes.map((b, idx) => (
          <button
            key={b.id}
            type="button"
            onClick={() => setActiveBox(idx)}
            className={`p-2.5 rounded-xl border text-left transition ${
              activeBox === idx
                ? 'border-indigo-500 bg-indigo-500/10 dark:bg-indigo-950/50'
                : idx < activeBox
                ? 'border-emerald-500/30 bg-emerald-500/5'
                : 'border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/40'
            }`}
          >
            <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 dark:text-slate-400">
              <span>{b.badge}</span>
              {idx < activeBox && <CheckCircle2 className="w-3 h-3 text-emerald-500" />}
            </div>
            <div className="text-xs font-semibold text-slate-900 dark:text-slate-100 mt-0.5 truncate">
              {b.title.split('(')[0].trim()}
            </div>
          </button>
        ))}
      </div>

      {/* Active Box Card */}
      <div className="rounded-xl border border-indigo-500/30 bg-slate-50 dark:bg-slate-950/60 p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
            {current.title}
          </h3>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">
            {current.badge}
          </span>
        </div>
        <p className="text-xs text-slate-600 dark:text-slate-300">{current.whatInside}</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3 space-y-1">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
              Input to This Box
            </div>
            <div className="font-mono text-xs text-slate-800 dark:text-slate-200 break-all">
              {current.tensorBefore}
            </div>
          </div>
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 space-y-1">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
              Output from This Box
            </div>
            <div className="font-mono text-xs font-semibold text-emerald-700 dark:text-emerald-300 break-all">
              {current.tensorAfter}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <p className="text-xs text-slate-500 dark:text-slate-400">{current.detail}</p>
          {activeBox < boxes.length - 1 ? (
            <button
              type="button"
              onClick={() => setActiveBox((b) => b + 1)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition"
            >
              Next Box
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          ) : (
            <button
              type="button"
              onClick={onReconstructAll}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition"
            >
              All Boxes Done — Reconstruct Together
              <Layers className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </section>
  );
};
