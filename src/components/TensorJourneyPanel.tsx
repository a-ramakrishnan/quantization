import React, { useMemo, useState } from 'react';
import { Binary, Cpu, HardDrive, Sparkles } from 'lucide-react';
import {
  packInt4PairToHex,
  toBfloat16Approx,
  toFp16Approx,
  toTwosComplementBits,
} from '../utils/quantization';

interface TensorJourneyPanelProps {
  activeRowWeights: number[];
  activeRowIndex: number;
}

type VectorSource = 'transformer8' | 'outlier8' | 'matrixRow';

const TRANSFORMER_8 = [-1.0, -0.8, -0.3, -0.1, 0.1, 0.3, 0.8, 1.0];
const OUTLIER_8 = [0.01, -0.02, 0.03, -0.04, 0.05, 0.01, 0.02, 1.0];

export const TensorJourneyPanel: React.FC<TensorJourneyPanelProps> = ({
  activeRowWeights,
  activeRowIndex,
}) => {
  const [source, setSource] = useState<VectorSource>('transformer8');
  const [groupSplit, setGroupSplit] = useState<boolean>(false);

  const vec = useMemo(() => {
    if (source === 'transformer8') return TRANSFORMER_8;
    if (source === 'outlier8') return OUTLIER_8;
    return activeRowWeights;
  }, [source, activeRowWeights]);

  // Compute INT8 (symmetric [-127, 127]) and INT4 (symmetric [-7, 7])
  const analysis = useMemo(() => {
    const absMaxAll = Math.max(...vec.map((v) => Math.abs(v)), 1e-6);
    const scale8 = absMaxAll / 127;
    const scale4Global = absMaxAll / 7;

    // Optional group-wise INT4 for outlier8 (first 7 vs last 1, or halves)
    const getInt4Scale = (idx: number) => {
      if (!groupSplit) return scale4Global;
      if (source === 'outlier8') {
        const smallMax = Math.max(...vec.slice(0, 7).map(Math.abs), 1e-6);
        return idx < 7 ? smallMax / 7 : Math.abs(vec[7]) / 7;
      }
      const mid = Math.ceil(vec.length / 2);
      const slice = idx < mid ? vec.slice(0, mid) : vec.slice(mid);
      const localMax = Math.max(...slice.map(Math.abs), 1e-6);
      return localMax / 7;
    };

    const rows = vec.map((w, idx) => {
      const fp16 = toFp16Approx(w);
      const bf16 = toBfloat16Approx(w);

      const q8 = Math.max(-127, Math.min(127, Math.round(w / scale8)));
      const w8 = q8 * scale8;

      const s4 = getInt4Scale(idx);
      const q4 = Math.max(-7, Math.min(7, Math.round(w / s4)));
      const w4 = q4 * s4;

      return {
        idx,
        w,
        fp16,
        bf16,
        q8,
        w8,
        err8: Math.abs(w - w8),
        bin8: toTwosComplementBits(q8, 8),
        s4,
        q4,
        w4,
        err4: Math.abs(w - w4),
        bin4: toTwosComplementBits(q4, 4),
      };
    });

    const mae8 = rows.reduce((a, r) => a + r.err8, 0) / rows.length;
    const mae4 = rows.reduce((a, r) => a + r.err4, 0) / rows.length;

    // Pairwise INT4 bit-packing
    const packedBytes: {
      pairIdx: number;
      hiQ: number;
      loQ: number;
      hiBin: string;
      loBin: string;
      hex: string;
    }[] = [];
    for (let i = 0; i < rows.length; i += 2) {
      const hi = rows[i];
      const lo = rows[i + 1] ?? { q4: 0, bin4: '0000' };
      packedBytes.push({
        pairIdx: i / 2,
        hiQ: hi.q4,
        loQ: lo.q4,
        hiBin: hi.bin4,
        loBin: lo.bin4,
        hex: packInt4PairToHex(hi.q4, lo.q4),
      });
    }

    const fp32Bytes = vec.length * 4;
    const fp16Bytes = vec.length * 2;
    const int8Bytes = vec.length * 1;
    const int4PackedBytes = packedBytes.length;
    const scaleOverheadBytes = groupSplit ? 4 : 2; // FP16 scale(s)

    return {
      scale8,
      scale4Global,
      rows,
      mae8,
      mae4,
      packedBytes,
      fp32Bytes,
      fp16Bytes,
      int8Bytes,
      int4PackedBytes,
      scaleOverheadBytes,
    };
  }, [vec, groupSplit, source]);

  return (
    <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 sm:p-5 space-y-5">
      {/* Header + Vector Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-violet-500/10 text-violet-600 dark:text-violet-400">
            <Binary className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Progressive Format Journey: FP32 → FP16 / BF16 → INT8 → Packed INT4
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Track every single weight in a vector: numerical precision, binary/hex memory layout, and 9B LLM scale.
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {(
            [
              ['transformer8', '8 Weights (Transformer Row)'],
              ['outlier8', '8 Weights (With Outlier 1.0)'],
              ['matrixRow', `Your Matrix Row ${activeRowIndex}`],
            ] as [VectorSource, string][]
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              onClick={() => setSource(key)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
                source === key
                  ? 'bg-violet-600 text-white shadow-xs'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Outlier Group-wise toggle */}
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 px-3.5 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="text-slate-600 dark:text-slate-300">
          {source === 'outlier8' && !groupSplit ? (
            <span>
              <strong className="text-rose-600 dark:text-rose-400">Outlier Trap:</strong>{' '}
              The single <span className="font-mono">1.0</span> weight forces{' '}
              <span className="font-mono">s = 0.1429</span>, crushing all 7 small weights to{' '}
              <strong className="font-mono">q = 0</strong>! Enable Group-wise INT4 to rescue them.
            </span>
          ) : (
            <span>
              <strong className="text-slate-800 dark:text-slate-100">Scales:</strong>{' '}
              INT8 <span className="font-mono">s = {analysis.scale8.toFixed(6)}</span> · INT4{' '}
              <span className="font-mono">
                {groupSplit ? 'Group-wise (2 groups)' : `s = ${analysis.scale4Global.toFixed(6)}`}
              </span>
            </span>
          )}
        </div>
        <label className="inline-flex items-center gap-2 font-medium text-indigo-600 dark:text-indigo-400 cursor-pointer">
          <input
            type="checkbox"
            checked={groupSplit}
            onChange={(e) => setGroupSplit(e.target.checked)}
            className="rounded accent-indigo-600"
          />
          Group-wise INT4 Scaling
        </label>
      </div>

      {/* 1. Element-by-Element Numerical Transformation Table */}
      <div className="w-full overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-slate-100/80 dark:bg-slate-800/70 text-slate-600 dark:text-slate-300 font-semibold border-b border-slate-200 dark:border-slate-800">
              <th className="py-2 px-3">Weight</th>
              <th className="py-2 px-3">FP32 (4B)</th>
              <th className="py-2 px-3">FP16 (2B)</th>
              <th className="py-2 px-3">BF16 (2B)</th>
              <th className="py-2 px-3 text-emerald-700 dark:text-emerald-300">
                INT8 (q / Binary)
              </th>
              <th className="py-2 px-3">INT8 Ŵ</th>
              <th className="py-2 px-3 text-indigo-700 dark:text-indigo-300">
                INT4 (q / Nibble)
              </th>
              <th className="py-2 px-3">INT4 Ŵ</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200/80 dark:divide-slate-800 font-mono">
            {analysis.rows.map((r) => (
              <tr
                key={r.idx}
                className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition"
              >
                <td className="py-2 px-3 font-semibold text-slate-500 dark:text-slate-400">
                  W{r.idx + 1}
                </td>
                <td className="py-2 px-3 font-semibold text-slate-900 dark:text-slate-100">
                  {r.w.toFixed(4)}
                </td>
                <td className="py-2 px-3 text-slate-600 dark:text-slate-300">
                  {r.fp16.toFixed(5)}
                </td>
                <td className="py-2 px-3 text-slate-600 dark:text-slate-300">
                  {r.bf16.toFixed(5)}
                </td>
                <td className="py-2 px-3">
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">
                    {r.q8}
                  </span>{' '}
                  <span className="text-[10px] text-slate-400">{r.bin8}</span>
                </td>
                <td className="py-2 px-3 text-slate-700 dark:text-slate-200">
                  {r.w8.toFixed(4)}
                </td>
                <td className="py-2 px-3">
                  <span
                    className={`font-bold ${
                      r.q4 === 0 && r.w !== 0
                        ? 'text-rose-500'
                        : 'text-indigo-600 dark:text-indigo-400'
                    }`}
                  >
                    {r.q4}
                  </span>{' '}
                  <span className="text-[10px] text-slate-400">{r.bin4}</span>
                </td>
                <td className="py-2 px-3 text-slate-700 dark:text-slate-200">
                  {r.w4.toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mean Absolute Error comparison */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
        <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/5 p-3 flex items-center justify-between">
          <div>
            <div className="font-semibold text-emerald-700 dark:text-emerald-300">
              INT8 Mean Absolute Error
            </div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">
              8-bit integer (255 symmetric levels) — very close to FP32
            </div>
          </div>
          <span className="font-mono text-base font-bold text-emerald-700 dark:text-emerald-300">
            {analysis.mae8.toFixed(5)}
          </span>
        </div>
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/5 p-3 flex items-center justify-between">
          <div>
            <div className="font-semibold text-amber-700 dark:text-amber-300">
              INT4 Mean Absolute Error
            </div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">
              4-bit integer (15 symmetric levels) — half the storage of INT8
            </div>
          </div>
          <span className="font-mono text-base font-bold text-amber-700 dark:text-amber-300">
            {analysis.mae4.toFixed(5)}
          </span>
        </div>
      </div>

      {/* 2. Physical INT4 Bit-Packing + 3. 9B LLM Scale-Up */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Physical Bit-Packing */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-800 dark:text-slate-200">
              <HardDrive className="w-4 h-4 text-indigo-500" />
              Physical Memory: INT4 Nibble Bit-Packing
            </div>
            <span className="text-[11px] font-mono text-indigo-600 dark:text-indigo-400">
              {analysis.fp32Bytes}B → {analysis.int4PackedBytes}B + {analysis.scaleOverheadBytes}B scale
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {analysis.packedBytes.map((pb) => (
              <div
                key={pb.pairIdx}
                className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-2 text-center space-y-1"
              >
                <div className="text-[10px] font-mono text-slate-400">
                  Byte {pb.pairIdx}
                </div>
                <div className="flex items-center justify-center gap-1 font-mono text-[11px]">
                  <span className="px-1 py-0.5 rounded bg-indigo-500/10 text-indigo-700 dark:text-indigo-300">
                    {pb.hiBin}
                  </span>
                  <span className="px-1 py-0.5 rounded bg-violet-500/10 text-violet-700 dark:text-violet-300">
                    {pb.loBin}
                  </span>
                </div>
                <div className="text-[10px] font-mono text-slate-600 dark:text-slate-300">
                  ({pb.hiQ}, {pb.loQ})
                </div>
                <div className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400">
                  {pb.hex}
                </div>
              </div>
            ))}
          </div>

          <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
            Two signed 4-bit weights pack into 1 byte (high + low nibble). Including{' '}
            {analysis.scaleOverheadBytes}B FP16 scale metadata gives{' '}
            <strong className="font-mono text-slate-700 dark:text-slate-200">
              {(
                analysis.fp32Bytes /
                (analysis.int4PackedBytes + analysis.scaleOverheadBytes)
              ).toFixed(2)}
              ×
            </strong>{' '}
            actual compression on this {vec.length}-weight slice.
          </p>
        </div>

        {/* 9B Parameter LLM Memory Footprint */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-800 dark:text-slate-200">
              <Cpu className="w-4 h-4 text-emerald-500" />
              Scaling Up: 9B-Parameter LLM Weight Storage
            </div>
            <span className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-700 dark:text-emerald-300">
              <Sparkles className="w-3 h-3" />
              W4A16
            </span>
          </div>

          <div className="space-y-2 text-xs">
            {[
              { label: 'FP32 (4 bytes/param)', gb: '36.0 GB', pct: 100, color: 'bg-slate-400' },
              { label: 'FP16 / BF16 (2 bytes/param)', gb: '18.0 GB', pct: 50, color: 'bg-blue-500' },
              { label: 'INT8 (1 byte/param)', gb: '9.0 GB', pct: 25, color: 'bg-emerald-500' },
              {
                label: 'INT4 + Group-128 FP16 Scales',
                gb: '4.64 GB',
                pct: 13,
                color: 'bg-indigo-600',
              },
            ].map((item) => (
              <div key={item.label} className="space-y-1">
                <div className="flex justify-between font-mono text-[11px]">
                  <span className="text-slate-600 dark:text-slate-300">{item.label}</span>
                  <span className="font-bold text-slate-900 dark:text-slate-100">{item.gb}</span>
                </div>
                <div className="h-2 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${item.color}`}
                    style={{ width: `${item.pct}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
            With group size 128, 70.3M FP16 scales add only ~0.14 GB on top of 4.5 GB of packed INT4
            weights—while activations <span className="font-mono">x</span> stay in FP16 (W4A16).
          </p>
        </div>
      </div>
    </section>
  );
};
