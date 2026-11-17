import React, { useMemo, useState } from 'react';
import { ArrowRight, Cpu, Layers, Sparkles } from 'lucide-react';
import {
  packInt4PairToHex,
  toBfloat16Approx,
  toFp16Approx,
  toFp8E4M3,
} from '../utils/quantization';

type BasicTarget = 'fp16' | 'fp8' | 'int8' | 'int4';

interface MatrixBlock {
  key: 'Q' | 'K' | 'V';
  title: string;
  role: string;
  weights: number[][];
}

const INITIAL_VECTOR_X = [0.8, -0.4, 1.2, 0.5];

const INITIAL_QKV: Record<'Q' | 'K' | 'V', number[][]> = {
  Q: [
    [0.62, -0.35, 0.91, -0.18],
    [-0.44, 0.78, 0.25, -0.52],
  ],
  K: [
    [0.51, 0.29, -0.84, 0.37],
    [-0.22, -0.66, 0.48, 0.71],
  ],
  V: [
    [0.33, -0.74, 0.59, 0.14],
    [0.85, 0.19, -0.41, -0.63],
  ],
};

function convertFromBf16(
  weights: number[][],
  target: BasicTarget
): {
  bf16: number[][];
  stored: string[][];
  dequant: number[][];
  scale: number | null;
  mae: number;
} {
  // First snap inputs to BF16 baseline
  const bf16 = weights.map((r) => r.map((w) => toBfloat16Approx(w)));
  const flat = bf16.flat();
  const absMax = Math.max(...flat.map(Math.abs), 1e-6);

  let scale: number | null = null;
  if (target === 'int8') scale = absMax / 127;
  if (target === 'int4') scale = absMax / 7;

  const stored: string[][] = [];
  const dequant: number[][] = [];
  let errSum = 0;

  for (let r = 0; r < bf16.length; r++) {
    const sRow: string[] = [];
    const dRow: number[] = [];
    for (let c = 0; c < bf16[r].length; c++) {
      const w = bf16[r][c];
      if (target === 'fp16') {
        const v = toFp16Approx(w);
        sRow.push(v.toFixed(4));
        dRow.push(v);
        errSum += Math.abs(w - v);
      } else if (target === 'fp8') {
        const fp8 = toFp8E4M3(w);
        sRow.push(fp8.hex);
        dRow.push(fp8.val);
        errSum += Math.abs(w - fp8.val);
      } else if (target === 'int8') {
        const q = Math.max(-127, Math.min(127, Math.round(w / scale!)));
        const v = q * scale!;
        sRow.push(String(q));
        dRow.push(v);
        errSum += Math.abs(w - v);
      } else {
        const q = Math.max(-7, Math.min(7, Math.round(w / scale!)));
        const v = q * scale!;
        sRow.push(String(q));
        dRow.push(v);
        errSum += Math.abs(w - v);
      }
    }
    stored.push(sRow);
    dequant.push(dRow);
  }

  return {
    bf16,
    stored,
    dequant,
    scale,
    mae: errSum / flat.length,
  };
}

function matVec(mat: number[][], vec: number[]): number[] {
  return mat.map((row) => row.reduce((acc, w, i) => acc + w * vec[i], 0));
}

export const BasicModeView: React.FC = () => {
  const [target, setTarget] = useState<BasicTarget>('int4');
  const [xVec, setXVec] = useState<number[]>(INITIAL_VECTOR_X);
  const [qkv, setQkv] = useState(INITIAL_QKV);
  const [activeMat, setActiveMat] = useState<'Q' | 'K' | 'V'>('Q');

  const blocks: MatrixBlock[] = useMemo(
    () => [
      { key: 'Q', title: 'Query (W_Q)', role: 'Projects x → Query', weights: qkv.Q },
      { key: 'K', title: 'Key (W_K)', role: 'Projects x → Key', weights: qkv.K },
      { key: 'V', title: 'Value (W_V)', role: 'Projects x → Value', weights: qkv.V },
    ],
    [qkv]
  );

  const computed = useMemo(() => {
    const map: Record<
      'Q' | 'K' | 'V',
      ReturnType<typeof convertFromBf16> & {
        outBf16: number[];
        outQuant: number[];
      }
    > = {} as never;

    for (const b of blocks) {
      const conv = convertFromBf16(b.weights, target);
      map[b.key] = {
        ...conv,
        outBf16: matVec(conv.bf16, xVec),
        outQuant: matVec(conv.dequant, xVec),
      };
    }
    return map;
  }, [blocks, target, xVec]);

  const activeData = computed[activeMat];
  const targetMeta: Record<
    BasicTarget,
    { label: string; bits: number; savingsVsBf16: string; desc: string }
  > = {
    fp16: {
      label: 'FP16',
      bits: 16,
      savingsVsBf16: '1.0× (same size, higher mantissa precision)',
      desc: 'Converts BF16 (1S+8E+7M) to IEEE FP16 (1S+5E+10M). Same 16-bit size, finer decimal precision.',
    },
    fp8: {
      label: 'FP8 (E4M3)',
      bits: 8,
      savingsVsBf16: '2.0× smaller HBM vs BF16',
      desc: 'Shrinks each weight from 16 bits to 8-bit floating point (1S+4E+3M) — no integer scale needed.',
    },
    int8: {
      label: 'INT8',
      bits: 8,
      savingsVsBf16: '2.0× smaller HBM vs BF16',
      desc: 'Maps BF16 weights onto [-127, 127] integers using a symmetric scale S = max(|W|)/127.',
    },
    int4: {
      label: 'INT4',
      bits: 4,
      savingsVsBf16: '4.0× smaller HBM vs BF16',
      desc: 'Maps BF16 weights onto [-7, 7] integers (2 weights packed per byte) using scale S = max(|W|)/7.',
    },
  };

  const info = targetMeta[target];

  const updateWeight = (mat: 'Q' | 'K' | 'V', r: number, c: number, val: number) => {
    setQkv((prev) => ({
      ...prev,
      [mat]: prev[mat].map((row, ri) =>
        ri === r ? row.map((w, ci) => (ci === c ? val : w)) : row
      ),
    }));
  };

  // Show INT4 packed bytes for row 0 of active matrix when INT4 is selected
  const packedPreview = useMemo(() => {
    if (target !== 'int4') return [];
    const row0 = activeData.stored[0].map(Number);
    return [
      packInt4PairToHex(row0[0], row0[1]),
      packInt4PairToHex(row0[2], row0[3]),
    ];
  }, [target, activeData]);

  return (
    <div className="space-y-5">
      {/* Basic Mode Assumptions Banner */}
      <section className="rounded-2xl border border-indigo-500/25 bg-gradient-to-r from-indigo-500/5 via-white to-emerald-500/5 dark:from-indigo-950/30 dark:via-slate-900 dark:to-emerald-950/20 p-4 sm:p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-500" />
            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
              Basic Mode: BF16 Transformer Q, K, V Quantization
            </h2>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border border-emerald-500/20">
            {info.savingsVsBf16}
          </span>
        </div>
        <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          Assumes weights start in <strong>BF16</strong>, a small 4-element token vector{' '}
          <span className="font-mono font-semibold">x</span>, and small{' '}
          <span className="font-mono font-semibold">2×4</span>{' '}
          <strong>Q, K, V</strong> projection matrices. Pick a target format below to see how weights
          compress in GPU HBM and reconstruct for <span className="font-mono">Q = W_Q·x</span>,{' '}
          <span className="font-mono">K = W_K·x</span>, <span className="font-mono">V = W_V·x</span>.
        </p>

        {/* Format Selector */}
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 mr-1">
            Convert BF16 →
          </span>
          {(['fp16', 'fp8', 'int8', 'int4'] as BasicTarget[]).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTarget(t)}
              className={`px-3.5 py-1.5 rounded-xl text-xs font-mono font-bold transition ${
                target === t
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              {targetMeta[t].label}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">{info.desc}</p>
      </section>

      {/* Input Vector x + Q/K/V Selector */}
      <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-slate-700 dark:text-slate-200">
              Small Input Vector x (BF16):
            </span>
            <div className="grid grid-cols-4 gap-1.5 w-64">
              {xVec.map((val, idx) => (
                <input
                  key={`bx-${idx}`}
                  type="number"
                  step="0.1"
                  value={val}
                  onChange={(e) => {
                    const p = parseFloat(e.target.value);
                    if (!Number.isNaN(p)) {
                      setXVec((prev) => prev.map((v, i) => (i === idx ? p : v)));
                    }
                  }}
                  aria-label={`Basic input x${idx}`}
                  className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 px-2 py-1 text-center font-mono text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
              ))}
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            {blocks.map((b) => (
              <button
                key={b.key}
                type="button"
                onClick={() => setActiveMat(b.key)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                  activeMat === b.key
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                }`}
              >
                {b.title}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Active Q/K/V Matrix Transformation: BF16 -> Stored -> Dequantized */}
      <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 sm:p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-500" />
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              {blocks.find((b) => b.key === activeMat)?.title} (2×4): BF16 → {info.label} → Dequantized
            </h3>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono">
            {activeData.scale !== null && (
              <span className="text-indigo-600 dark:text-indigo-400">
                Scale S = {activeData.scale.toFixed(4)}
              </span>
            )}
            <span className="text-amber-600 dark:text-amber-400">
              MAE = {activeData.mae.toFixed(5)}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
          {/* 1. BF16 */}
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/50 p-3 space-y-2">
            <div className="text-xs font-semibold text-slate-700 dark:text-slate-200">
              1. Source BF16 (Editable)
            </div>
            <div className="grid grid-cols-4 gap-1.5">
              {qkv[activeMat].map((row, r) =>
                row.map((w, c) => (
                  <input
                    key={`bw-${r}-${c}`}
                    type="number"
                    step="0.05"
                    value={w}
                    onChange={(e) => {
                      const p = parseFloat(e.target.value);
                      if (!Number.isNaN(p)) updateWeight(activeMat, r, c, p);
                    }}
                    className="rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 py-1.5 px-1 text-center font-mono text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  />
                ))
              )}
            </div>
          </div>

          {/* 2. Stored in HBM */}
          <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3 space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold text-emerald-700 dark:text-emerald-300">
              <span>2. Stored in GPU HBM ({info.label})</span>
              {target === 'int4' && (
                <span className="font-mono text-[10px]">
                  Row0: {packedPreview.join(' ')}
                </span>
              )}
            </div>
            <div className="grid grid-cols-4 gap-1.5">
              {activeData.stored.map((row, r) =>
                row.map((val, c) => (
                  <div
                    key={`bs-${r}-${c}`}
                    className="rounded-lg border border-emerald-500/20 bg-white dark:bg-slate-900 py-1.5 px-1 text-center font-mono text-xs font-bold text-emerald-700 dark:text-emerald-300"
                  >
                    {val}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* 3. Dequantized in SM Registers */}
          <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 space-y-2">
            <div className="text-xs font-semibold text-amber-700 dark:text-amber-300">
              3. Dequantized in SM Registers (Ŵ)
            </div>
            <div className="grid grid-cols-4 gap-1.5">
              {activeData.dequant.map((row, r) =>
                row.map((val, c) => (
                  <div
                    key={`bd-${r}-${c}`}
                    className="rounded-lg border border-amber-500/20 bg-white dark:bg-slate-900 py-1.5 px-1 text-center font-mono text-xs text-slate-800 dark:text-slate-100"
                  >
                    {val.toFixed(3)}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Q, K, V Output Vectors Comparison */}
      <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 sm:p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-emerald-500" />
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Resulting Q, K, V Projection Outputs (BF16 Baseline vs. {info.label})
            </h3>
          </div>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            Computed in Tensor Cores via <span className="font-mono">Ŵ · x</span>
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {blocks.map((b) => {
            const d = computed[b.key];
            return (
              <div
                key={`out-${b.key}`}
                className={`rounded-xl border p-3.5 space-y-2.5 transition cursor-pointer ${
                  activeMat === b.key
                    ? 'border-indigo-500 bg-indigo-500/5'
                    : 'border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/40'
                }`}
                onClick={() => setActiveMat(b.key)}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-900 dark:text-slate-100">
                    {b.title} · x
                  </span>
                  <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                    MAE: {d.mae.toFixed(4)}
                  </span>
                </div>

                {[0, 1].map((rowIdx) => {
                  const base = d.outBf16[rowIdx];
                  const quant = d.outQuant[rowIdx];
                  const diff = base - quant;
                  return (
                    <div
                      key={`qkv-r-${rowIdx}`}
                      className="rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 px-2.5 py-1.5 text-xs font-mono flex items-center justify-between"
                    >
                      <span className="text-slate-500">
                        {b.key}[{rowIdx}]
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-blue-600 dark:text-blue-400">
                          {base.toFixed(3)}
                        </span>
                        <ArrowRight className="w-3 h-3 text-slate-400" />
                        <span className="text-amber-600 dark:text-amber-400 font-semibold">
                          {quant.toFixed(3)}
                        </span>
                        <span className="text-[10px] text-rose-500">
                          (Δ{diff >= 0 ? '+' : ''}
                          {diff.toFixed(3)})
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
};
