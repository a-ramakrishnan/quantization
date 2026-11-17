import React from 'react';
import { Activity } from 'lucide-react';
import { QuantizationResult } from '../types/quantization';

interface ForwardPassPanelProps {
  inputs: number[];
  onUpdateInput: (idx: number, val: number) => void;
  result: QuantizationResult;
}

export const ForwardPassPanel: React.FC<ForwardPassPanelProps> = ({
  inputs,
  onUpdateInput,
  result,
}) => {
  const maxAbsY = Math.max(
    ...result.yTrue.map(Math.abs),
    ...result.yQuant.map(Math.abs),
    0.5
  );

  return (
    <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 lg:p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Linear Layer Forward Pass (y = Wx vs. ŷ = Ŵx) &amp; Error Metrics
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Edit the input activation vector <span className="font-mono">x</span> to see how weight quantization affects the final layer output.
            </p>
          </div>
        </div>
      </div>

      {/* Editable Input Vector x */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3 flex flex-wrap items-center gap-3">
        <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
          Input Vector x (FP32):
        </span>
        <div className="grid grid-cols-4 gap-2 flex-1 min-w-[240px] max-w-md">
          {inputs.map((val, idx) => (
            <div
              key={`x-${idx}`}
              className="flex items-center gap-1 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1"
            >
              <span className="text-[10px] font-mono text-slate-400">x{idx}</span>
              <input
                type="number"
                step="0.1"
                value={val}
                onChange={(e) => {
                  const parsed = parseFloat(e.target.value);
                  if (!Number.isNaN(parsed)) onUpdateInput(idx, parsed);
                }}
                aria-label={`Input x${idx}`}
                className="w-full bg-transparent text-right font-mono text-xs font-medium text-slate-900 dark:text-slate-100 focus:outline-none"
              />
            </div>
          ))}
        </div>
      </div>

      {/* Output Comparison Bars */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="space-y-2.5">
          <div className="flex items-center justify-between text-xs font-medium text-slate-600 dark:text-slate-300">
            <span>Output Vector Comparison</span>
            <div className="flex items-center gap-3 text-[11px]">
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-sm bg-blue-500 inline-block" />
                True y = Wx
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-sm bg-amber-500 inline-block" />
                Quantized ŷ = Ŵx
              </span>
            </div>
          </div>

          <div className="space-y-2">
            {result.yTrue.map((yt, i) => {
              const yq = result.yQuant[i];
              const err = result.yError[i];
              const wTrue = Math.min((Math.abs(yt) / maxAbsY) * 100, 100);
              const wQuant = Math.min((Math.abs(yq) / maxAbsY) * 100, 100);
              return (
                <div
                  key={`out-${i}`}
                  className="rounded-lg border border-slate-200 dark:border-slate-800 p-2.5 bg-slate-50/70 dark:bg-slate-950/40 space-y-1.5"
                >
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="font-semibold text-slate-700 dark:text-slate-200">
                      Out[{i}]
                    </span>
                    <span className="text-[11px] text-slate-500 dark:text-slate-400">
                      y={yt.toFixed(3)} · ŷ={yq.toFixed(3)} ·{' '}
                      <span className="text-rose-600 dark:text-rose-400">
                        Δ={err >= 0 ? '+' : ''}
                        {err.toFixed(3)}
                      </span>
                    </span>
                  </div>
                  <div className="space-y-1">
                    <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div
                        className="h-full bg-blue-500 rounded-full transition-all"
                        style={{ width: `${wTrue}%` }}
                      />
                    </div>
                    <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div
                        className="h-full bg-amber-500 rounded-full transition-all"
                        style={{ width: `${wQuant}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Summary Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 content-start">
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3.5">
            <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
              Weight MSE
            </div>
            <div className="mt-1 text-lg font-mono font-bold text-slate-900 dark:text-slate-100">
              {result.weightMse.toFixed(5)}
            </div>
            <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
              Mean squared error between W and Ŵ. Lower is better.
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3.5">
            <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
              Weight SQNR
            </div>
            <div className="mt-1 text-lg font-mono font-bold text-indigo-600 dark:text-indigo-400">
              {result.weightSqnrDb.toFixed(1)} dB
            </div>
            <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
              Signal-to-Quantization-Noise Ratio. ~6 dB gained per extra bit!
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3.5">
            <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
              Output MSE &amp; Cosine Sim
            </div>
            <div className="mt-1 text-lg font-mono font-bold text-slate-900 dark:text-slate-100">
              {result.outputMse.toFixed(4)}
            </div>
            <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
              Cosine similarity:{' '}
              <span className="font-mono font-semibold text-emerald-600 dark:text-emerald-400">
                {(result.cosineSimilarity * 100).toFixed(2)}%
              </span>
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3.5">
            <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
              HBM Memory Savings
            </div>
            <div className="mt-1 text-lg font-mono font-bold text-emerald-600 dark:text-emerald-400">
              {result.memory.compressionRatio.toFixed(2)}×
            </div>
            <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
              {result.memory.totalQuantizedBits}b vs {result.memory.fp32Bits}b FP32 (incl. S,Z metadata).
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};
