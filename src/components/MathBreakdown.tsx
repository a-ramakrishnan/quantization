import React from 'react';
import { Calculator } from 'lucide-react';
import { CellResult, GroupQuantParams, QuantConfig } from '../types/quantization';

interface MathBreakdownProps {
  selectedCell: CellResult;
  groupParams: GroupQuantParams;
  config: QuantConfig;
}

export const MathBreakdown: React.FC<MathBreakdownProps> = ({
  selectedCell,
  groupParams,
  config,
}) => {
  const { scale, zeroPoint, qMin, qMax, clippedMin, clippedMax } = groupParams;
  const divided = selectedCell.wFp32 / scale;
  const rounded = Math.round(divided);

  return (
    <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 lg:p-5 space-y-3.5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400">
            <Calculator className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Live Step-by-Step Math for Cell [{selectedCell.row}, {selectedCell.col}]
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Group: <span className="font-medium">{groupParams.label}</span> ·{' '}
              {config.scheme === 'symmetric' ? 'Symmetric' : 'Asymmetric (Affine)'} · INT
              {config.bits}
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-2.5 text-xs">
        {/* 1. Scale & Zero-Point */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3 space-y-1">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-slate-800 dark:text-slate-200">
              1. Calibration: Scale (S) &amp; Zero-Point (Z)
            </span>
            <span className="font-mono text-[11px] text-indigo-600 dark:text-indigo-400">
              [{qMin}, {qMax}]
            </span>
          </div>
          {config.scheme === 'symmetric' ? (
            <div className="font-mono text-[11.5px] text-slate-700 dark:text-slate-300">
              S = max(|α|) / q_max = {Math.max(Math.abs(clippedMin), Math.abs(clippedMax)).toFixed(4)} / {qMax} ={' '}
              <strong className="text-indigo-600 dark:text-indigo-400">
                {scale.toFixed(5)}
              </strong>
              , &nbsp; Z = <strong>0</strong>
            </div>
          ) : (
            <div className="font-mono text-[11.5px] text-slate-700 dark:text-slate-300">
              S = ({clippedMax.toFixed(3)} − {clippedMin.toFixed(3)}) / ({qMax} − ({qMin})) ={' '}
              <strong className="text-indigo-600 dark:text-indigo-400">
                {scale.toFixed(5)}
              </strong>
              , &nbsp; Z = <strong>{zeroPoint}</strong>
            </div>
          )}
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            Scale S is the float distance between two adjacent integers; Zero-Point Z aligns real 0.0 to an integer.
          </p>
        </div>

        {/* 2. Quantization */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3 space-y-1">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-slate-800 dark:text-slate-200">
              2. Quantize: FP32 → INT{config.bits}
            </span>
            {selectedCell.isClipped && (
              <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-500/15 text-amber-700 dark:text-amber-300">
                Clipped!
              </span>
            )}
          </div>
          <div className="font-mono text-[11.5px] text-slate-700 dark:text-slate-300 leading-relaxed">
            W_q = clamp(round({selectedCell.wFp32.toFixed(3)} / {scale.toFixed(4)}) + {zeroPoint}, {qMin}, {qMax})
            <br />
            &nbsp;&nbsp;&nbsp;&nbsp;= clamp(round({divided.toFixed(3)}) + {zeroPoint}, {qMin}, {qMax})
            <br />
            &nbsp;&nbsp;&nbsp;&nbsp;= clamp({rounded} + {zeroPoint}, {qMin}, {qMax}) ={' '}
            <strong className="text-emerald-600 dark:text-emerald-400">
              {selectedCell.wq}
            </strong>
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            This integer <span className="font-mono">{selectedCell.wq}</span> is what gets stored in GPU HBM.
          </p>
        </div>

        {/* 3. Dequantization */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3 space-y-1">
          <span className="font-semibold text-slate-800 dark:text-slate-200 block">
            3. Dequantize (in GPU SM Registers): INT{config.bits} → Float
          </span>
          <div className="font-mono text-[11.5px] text-slate-700 dark:text-slate-300">
            Ŵ = S · (W_q − Z) = {scale.toFixed(4)} · ({selectedCell.wq} − {zeroPoint}) ={' '}
            <strong className="text-amber-600 dark:text-amber-400">
              {selectedCell.wDequant.toFixed(5)}
            </strong>
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            Reconstructed weight used in Tensor Core dot products.
          </p>
        </div>

        {/* 4. Error */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3 space-y-1">
          <span className="font-semibold text-slate-800 dark:text-slate-200 block">
            4. Quantization Error
          </span>
          <div className="font-mono text-[11.5px] text-slate-700 dark:text-slate-300">
            e = W − Ŵ = {selectedCell.wFp32.toFixed(4)} − {selectedCell.wDequant.toFixed(4)} ={' '}
            <strong className="text-rose-600 dark:text-rose-400">
              {selectedCell.error >= 0 ? '+' : ''}
              {selectedCell.error.toFixed(5)}
            </strong>
          </div>
        </div>
      </div>
    </section>
  );
};
