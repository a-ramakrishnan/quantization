import React from 'react';
import { X, Github, Cpu, Layers, Sparkles } from 'lucide-react';

interface AboutModalProps {
  open: boolean;
  onClose: () => void;
}

export const AboutModal: React.FC<AboutModalProps> = ({ open, onClose }) => {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/60 backdrop-blur-xs"
      role="dialog"
      aria-modal="true"
      aria-label="About Linear Layer Quantization Visualizer"
    >
      <div className="relative w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-2xl p-6 space-y-5">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-indigo-500/15 text-indigo-600 dark:text-indigo-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
                About This Visualizer
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Interactive Linear Layer Weight Quantization, Dequantization &amp; GPU Execution
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
            aria-label="Close About modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4 text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          <p>
            Modern neural networks and Large Language Models (LLMs) spend most of their inference time
            waiting to read weights from GPU High Bandwidth Memory (HBM).{' '}
            <strong className="text-slate-900 dark:text-slate-100">Weight quantization</strong> compresses
            32-bit floating-point weights into compact integers (INT8, INT4, INT2) so they fit in less
            VRAM and stream across the memory bus 4×–8× faster.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3.5 space-y-1.5">
              <div className="flex items-center gap-1.5 font-semibold text-slate-900 dark:text-slate-100 text-xs">
                <Layers className="w-4 h-4 text-indigo-500" />
                Key Concepts Explored
              </div>
              <ul className="text-xs space-y-1 list-disc list-inside text-slate-600 dark:text-slate-400">
                <li><strong>Symmetric vs. Asymmetric:</strong> Zero-point Z = 0 vs. affine shift.</li>
                <li><strong>Per-Tensor vs. Per-Channel / Group:</strong> Taming outliers with finer scales.</li>
                <li><strong>Clipping Calibration:</strong> Balancing outlier clipping vs. inlier rounding error.</li>
              </ul>
            </div>

            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3.5 space-y-1.5">
              <div className="flex items-center gap-1.5 font-semibold text-slate-900 dark:text-slate-100 text-xs">
                <Cpu className="w-4 h-4 text-emerald-500" />
                GPU Execution Flow
              </div>
              <ul className="text-xs space-y-1 list-disc list-inside text-slate-600 dark:text-slate-400">
                <li><strong>HBM:</strong> Stores packed INT weights + (S, Z).</li>
                <li><strong>L2 &amp; SM SRAM:</strong> Streams compressed tiles on-chip.</li>
                <li><strong>Registers &amp; Tensor Cores:</strong> Dequantizes Ŵ = S(W_q − Z) in registers &amp; computes ŷ = Ŵx.</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="pt-3 border-t border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <a
            href="https://github.com/a-ramakrishnan/quantization"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline"
          >
            <Github className="w-4 h-4" />
            github.com/a-ramakrishnan/quantization
          </a>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition"
          >
            Back to Visualizer
          </button>
        </div>
      </div>
    </div>
  );
};
