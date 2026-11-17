import React, { useState } from 'react';
import { Cpu, Layers, Zap } from 'lucide-react';
import {
  CellResult,
  GpuComponentId,
  QuantConfig,
  QuantizationResult,
} from '../types/quantization';
import { GPU_COMPONENT_DETAILS, PIPELINE_STEPS } from '../data/pipelineSteps';

interface GpuArchitectureProps {
  activeStep: number;
  config: QuantConfig;
  result: QuantizationResult;
  selectedCell: CellResult;
  inputs: number[];
}

export const GpuArchitecture: React.FC<GpuArchitectureProps> = ({
  activeStep,
  config,
  result,
  selectedCell,
  inputs,
}) => {
  const [inspectedId, setInspectedId] = useState<GpuComponentId | null>(null);
  const stepInfo = PIPELINE_STEPS[activeStep] ?? PIPELINE_STEPS[0];
  const focusedId = inspectedId ?? stepInfo.gpuTarget;
  const detail = GPU_COMPONENT_DETAILS[focusedId];

  const isActive = (id: GpuComponentId) => stepInfo.gpuTarget === id;

  const boxClass = (id: GpuComponentId) => {
    const active = isActive(id);
    const inspected = inspectedId === id;
    if (active) {
      return 'stroke-indigo-500 dark:stroke-indigo-400 fill-indigo-50/90 dark:fill-indigo-950/60';
    }
    if (inspected) {
      return 'stroke-amber-500 dark:stroke-amber-400 fill-amber-50/70 dark:fill-amber-950/40';
    }
    return 'stroke-slate-300 dark:stroke-slate-700 fill-white/90 dark:fill-slate-900/90';
  };

  const rowIdx = selectedCell.row;
  const yTrueVal = result.yTrue[rowIdx] ?? 0;
  const yQuantVal = result.yQuant[rowIdx] ?? 0;

  return (
    <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm overflow-hidden">
      <div className="px-5 py-3.5 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              GPU Hardware Architecture &amp; Quantized Linear Layer Dataflow
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Click any hardware unit to inspect its role. Active step highlights live data movement.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="px-2.5 py-1 rounded-full font-mono font-medium bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border border-emerald-500/20">
            {result.memory.compressionRatio.toFixed(2)}× HBM Compression
          </span>
          <span className="px-2.5 py-1 rounded-full font-mono font-medium bg-indigo-500/10 text-indigo-700 dark:text-indigo-300 border border-indigo-500/20">
            INT{config.bits} Weights → FP32 Compute
          </span>
        </div>
      </div>

      <div className="p-4 lg:p-5 space-y-4">
        {/* Responsive SVG GPU Architecture */}
        <div className="w-full overflow-x-auto">
          <svg
            viewBox="0 0 1040 380"
            className="w-full min-w-[780px] h-auto select-none"
            role="img"
            aria-label="GPU Architecture for Weight Quantization and Dequantization"
          >
            <defs>
              <marker
                id="arrow-active"
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 10 5 L 0 9 z" className="fill-indigo-500 dark:fill-indigo-400" />
              </marker>
              <marker
                id="arrow-idle"
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 10 5 L 0 9 z" className="fill-slate-400 dark:fill-slate-600" />
              </marker>
            </defs>

            {/* 1. HOST CPU & QUANTIZER */}
            <g
              className="cursor-pointer transition-all"
              onClick={() => setInspectedId('host')}
            >
              <rect
                x="14"
                y="42"
                width="175"
                height="296"
                rx="14"
                strokeWidth={isActive('host') ? 2.5 : 1.5}
                className={boxClass('host')}
              />
              <text
                x="101"
                y="68"
                textAnchor="middle"
                className="fill-slate-900 dark:fill-slate-100 text-[12px] font-bold"
              >
                Host CPU / Calibrator
              </text>
              <text
                x="101"
                y="85"
                textAnchor="middle"
                className="fill-slate-500 dark:fill-slate-400 text-[10px]"
              >
                Offline Quantization
              </text>

              {/* FP32 Weights box */}
              <rect
                x="28"
                y="102"
                width="147"
                height="62"
                rx="8"
                className="fill-slate-100 dark:fill-slate-800/80 stroke-slate-300 dark:stroke-slate-700"
                strokeWidth="1"
              />
              <text x="38" y="121" className="fill-slate-700 dark:fill-slate-200 text-[10px] font-semibold">
                FP32 Weights (W)
              </text>
              <text x="38" y="138" className="fill-slate-500 dark:fill-slate-400 text-[10px] font-mono">
                W[{selectedCell.row},{selectedCell.col}] = {selectedCell.wFp32.toFixed(2)}
              </text>
              <text x="38" y="153" className="fill-slate-500 dark:fill-slate-400 text-[9px] font-mono">
                {result.memory.fp32Bits} bits total
              </text>

              {/* Quantizer formula box */}
              <rect
                x="28"
                y="180"
                width="147"
                height="76"
                rx="8"
                className="fill-indigo-500/10 stroke-indigo-500/30"
                strokeWidth="1"
              />
              <text x="38" y="199" className="fill-indigo-700 dark:fill-indigo-300 text-[10px] font-semibold">
                Quantize → INT{config.bits}
              </text>
              <text x="38" y="216" className="fill-slate-700 dark:fill-slate-200 text-[9.5px] font-mono">
                S = {selectedCell.scale.toFixed(4)}
              </text>
              <text x="38" y="231" className="fill-slate-700 dark:fill-slate-200 text-[9.5px] font-mono">
                Z = {selectedCell.zeroPoint} ({config.scheme === 'symmetric' ? 'Sym' : 'Asym'})
              </text>
              <text x="38" y="246" className="fill-indigo-600 dark:fill-indigo-400 text-[9.5px] font-mono font-semibold">
                W_q = {selectedCell.wq}
              </text>

              {/* Input X */}
              <rect
                x="28"
                y="270"
                width="147"
                height="52"
                rx="8"
                className="fill-slate-100 dark:fill-slate-800/80 stroke-slate-300 dark:stroke-slate-700"
                strokeWidth="1"
              />
              <text x="38" y="289" className="fill-slate-700 dark:fill-slate-200 text-[10px] font-semibold">
                Input Vector (X)
              </text>
              <text x="38" y="306" className="fill-slate-500 dark:fill-slate-400 text-[9.5px] font-mono">
                [{inputs.map((v) => v.toFixed(1)).join(', ')}]
              </text>
            </g>

            {/* Arrow Host -> GPU HBM (PCIe) */}
            <path
              d="M 189 190 L 238 190"
              strokeWidth={activeStep === 1 || activeStep === 2 ? 2.5 : 1.5}
              className={
                activeStep === 1 || activeStep === 2
                  ? 'stroke-indigo-500 dark:stroke-indigo-400 animate-dash-flow'
                  : 'stroke-slate-400 dark:stroke-slate-600'
              }
              markerEnd={activeStep === 1 || activeStep === 2 ? 'url(#arrow-active)' : 'url(#arrow-idle)'}
            />
            <text
              x="213"
              y="179"
              textAnchor="middle"
              className="fill-slate-500 dark:fill-slate-400 text-[9px] font-mono"
            >
              PCIe
            </text>

            {/* GPU PACKAGE BOUNDARY */}
            <rect
              x="244"
              y="16"
              width="782"
              height="348"
              rx="18"
              className="fill-slate-50/70 dark:fill-slate-950/60 stroke-slate-300 dark:stroke-slate-700"
              strokeWidth="1.5"
              strokeDasharray="6 4"
            />
            <text
              x="264"
              y="36"
              className="fill-slate-700 dark:fill-slate-300 text-[11px] font-bold tracking-wide uppercase"
            >
              GPU Device (e.g., Tensor Core Architecture)
            </text>

            {/* 2. HBM / VRAM */}
            <g
              className="cursor-pointer transition-all"
              onClick={() => setInspectedId('hbm')}
            >
              <rect
                x="262"
                y="52"
                width="168"
                height="292"
                rx="12"
                strokeWidth={isActive('hbm') ? 2.5 : 1.5}
                className={boxClass('hbm')}
              />
              <text
                x="346"
                y="76"
                textAnchor="middle"
                className="fill-slate-900 dark:fill-slate-100 text-[12px] font-bold"
              >
                HBM / VRAM
              </text>
              <text
                x="346"
                y="92"
                textAnchor="middle"
                className="fill-slate-500 dark:fill-slate-400 text-[9.5px]"
              >
                Global Memory (~2–3 TB/s)
              </text>

              {/* Packed W_q */}
              <rect
                x="275"
                y="106"
                width="142"
                height="66"
                rx="8"
                className="fill-emerald-500/10 stroke-emerald-500/30"
                strokeWidth="1"
              />
              <text x="285" y="124" className="fill-emerald-700 dark:fill-emerald-300 text-[10px] font-semibold">
                Packed W_q (INT{config.bits})
              </text>
              <text x="285" y="141" className="fill-slate-700 dark:fill-slate-200 text-[9.5px] font-mono">
                {result.memory.quantizedWeightBits} bits ({result.memory.bandwidthSavedPct.toFixed(0)}% saved)
              </text>
              <text x="285" y="158" className="fill-slate-500 dark:fill-slate-400 text-[9px] font-mono">
                + Meta (S,Z): {result.memory.metadataBits}b
              </text>

              {/* Input Activations X */}
              <rect
                x="275"
                y="184"
                width="142"
                height="50"
                rx="8"
                className="fill-slate-100 dark:fill-slate-800/80 stroke-slate-300 dark:stroke-slate-700"
                strokeWidth="1"
              />
              <text x="285" y="203" className="fill-slate-700 dark:fill-slate-200 text-[10px] font-semibold">
                Activations X (FP32)
              </text>
              <text x="285" y="220" className="fill-slate-500 dark:fill-slate-400 text-[9.5px] font-mono">
                x[{selectedCell.col}] = {inputs[selectedCell.col]?.toFixed(2)}
              </text>

              {/* Output Activations Y_hat */}
              <rect
                x="275"
                y="246"
                width="142"
                height="82"
                rx="8"
                className={
                  activeStep === 5
                    ? 'fill-indigo-500/15 stroke-indigo-500/50'
                    : 'fill-slate-100 dark:fill-slate-800/80 stroke-slate-300 dark:stroke-slate-700'
                }
                strokeWidth="1"
              />
              <text x="285" y="265" className="fill-slate-700 dark:fill-slate-200 text-[10px] font-semibold">
                Output ŷ (FP32)
              </text>
              <text x="285" y="283" className="fill-indigo-600 dark:fill-indigo-400 text-[9.5px] font-mono font-semibold">
                ŷ[{rowIdx}] = {yQuantVal.toFixed(3)}
              </text>
              <text x="285" y="299" className="fill-slate-500 dark:fill-slate-400 text-[9px] font-mono">
                True y[{rowIdx}] = {yTrueVal.toFixed(3)}
              </text>
              <text x="285" y="315" className="fill-rose-600 dark:fill-rose-400 text-[9px] font-mono">
                Err = {(yTrueVal - yQuantVal).toFixed(4)}
              </text>
            </g>

            {/* Arrow HBM -> L2 */}
            <path
              d="M 430 175 L 468 175"
              strokeWidth={activeStep === 3 ? 2.5 : 1.5}
              className={
                activeStep === 3
                  ? 'stroke-indigo-500 dark:stroke-indigo-400 animate-dash-flow'
                  : 'stroke-slate-400 dark:stroke-slate-600'
              }
              markerEnd={activeStep === 3 ? 'url(#arrow-active)' : 'url(#arrow-idle)'}
            />

            {/* 3. L2 CACHE */}
            <g
              className="cursor-pointer transition-all"
              onClick={() => setInspectedId('l2')}
            >
              <rect
                x="474"
                y="96"
                width="82"
                height="162"
                rx="10"
                strokeWidth={isActive('l2') ? 2.5 : 1.5}
                className={boxClass('l2')}
              />
              <text
                x="515"
                y="156"
                textAnchor="middle"
                className="fill-slate-900 dark:fill-slate-100 text-[11px] font-bold"
              >
                L2 Cache
              </text>
              <text
                x="515"
                y="174"
                textAnchor="middle"
                className="fill-slate-500 dark:fill-slate-400 text-[9px]"
              >
                Tile Reuse
              </text>
              <text
                x="515"
                y="192"
                textAnchor="middle"
                className="fill-indigo-600 dark:fill-indigo-400 text-[9px] font-mono"
              >
                W_q, S, Z
              </text>
            </g>

            {/* Arrow L2 -> SM SRAM */}
            <path
              d="M 556 175 L 594 175"
              strokeWidth={activeStep === 3 ? 2.5 : 1.5}
              className={
                activeStep === 3
                  ? 'stroke-indigo-500 dark:stroke-indigo-400 animate-dash-flow'
                  : 'stroke-slate-400 dark:stroke-slate-600'
              }
              markerEnd={activeStep === 3 ? 'url(#arrow-active)' : 'url(#arrow-idle)'}
            />

            {/* STREAMING MULTIPROCESSOR (SM) CONTAINER */}
            <rect
              x="600"
              y="52"
              width="410"
              height="292"
              rx="14"
              className="fill-slate-100/80 dark:fill-slate-900/90 stroke-slate-400 dark:stroke-slate-600"
              strokeWidth="1.5"
            />
            <text
              x="616"
              y="74"
              className="fill-slate-900 dark:fill-slate-100 text-[11.5px] font-bold"
            >
              Streaming Multiprocessor (SM) — On-Chip Compute
            </text>

            {/* 4. SHARED MEMORY (SRAM) */}
            <g
              className="cursor-pointer transition-all"
              onClick={() => setInspectedId('sram')}
            >
              <rect
                x="616"
                y="90"
                width="115"
                height="182"
                rx="10"
                strokeWidth={isActive('sram') ? 2.5 : 1.5}
                className={boxClass('sram')}
              />
              <text
                x="673"
                y="114"
                textAnchor="middle"
                className="fill-slate-900 dark:fill-slate-100 text-[11px] font-bold"
              >
                Shared Mem
              </text>
              <text
                x="673"
                y="129"
                textAnchor="middle"
                className="fill-slate-500 dark:fill-slate-400 text-[9px]"
              >
                SRAM (~20+ TB/s)
              </text>
              <rect
                x="626"
                y="144"
                width="95"
                height="48"
                rx="6"
                className="fill-slate-100 dark:fill-slate-800 stroke-slate-300 dark:stroke-slate-700"
              />
              <text x="673" y="164" textAnchor="middle" className="fill-slate-700 dark:fill-slate-200 text-[9.5px] font-mono">
                Tile W_q[{rowIdx},:]
              </text>
              <text x="673" y="180" textAnchor="middle" className="fill-slate-500 dark:fill-slate-400 text-[9px] font-mono">
                [{result.cells[rowIdx].map((c) => c.wq).join(',')}]
              </text>

              <rect
                x="626"
                y="202"
                width="95"
                height="52"
                rx="6"
                className="fill-slate-100 dark:fill-slate-800 stroke-slate-300 dark:stroke-slate-700"
              />
              <text x="673" y="222" textAnchor="middle" className="fill-slate-700 dark:fill-slate-200 text-[9.5px] font-mono">
                Tile X, S, Z
              </text>
              <text x="673" y="238" textAnchor="middle" className="fill-indigo-600 dark:fill-indigo-400 text-[9px] font-mono">
                S={selectedCell.scale.toFixed(3)}
              </text>
            </g>

            {/* Arrow SRAM -> Register File */}
            <path
              d="M 731 180 L 758 180"
              strokeWidth={activeStep === 4 ? 2.5 : 1.5}
              className={
                activeStep === 4
                  ? 'stroke-indigo-500 dark:stroke-indigo-400 animate-dash-flow'
                  : 'stroke-slate-400 dark:stroke-slate-600'
              }
              markerEnd={activeStep === 4 ? 'url(#arrow-active)' : 'url(#arrow-idle)'}
            />

            {/* 5. REGISTER FILE & DEQUANTIZER */}
            <g
              className="cursor-pointer transition-all"
              onClick={() => setInspectedId('registers')}
            >
              <rect
                x="764"
                y="90"
                width="115"
                height="182"
                rx="10"
                strokeWidth={isActive('registers') ? 2.5 : 1.5}
                className={boxClass('registers')}
              />
              <text
                x="821"
                y="114"
                textAnchor="middle"
                className="fill-slate-900 dark:fill-slate-100 text-[11px] font-bold"
              >
                Register File
              </text>
              <text
                x="821"
                y="129"
                textAnchor="middle"
                className="fill-amber-600 dark:fill-amber-400 text-[9px] font-semibold"
              >
                On-the-Fly Dequant
              </text>

              <rect
                x="773"
                y="144"
                width="97"
                height="110"
                rx="6"
                className="fill-amber-500/10 stroke-amber-500/30"
              />
              <text x="821" y="162" textAnchor="middle" className="fill-slate-700 dark:fill-slate-200 text-[9.5px] font-mono font-semibold">
                Ŵ = S·(W_q - Z)
              </text>
              <text x="821" y="182" textAnchor="middle" className="fill-slate-600 dark:fill-slate-300 text-[9px] font-mono">
                {selectedCell.scale.toFixed(3)}·({selectedCell.wq}-{selectedCell.zeroPoint})
              </text>
              <text x="821" y="204" textAnchor="middle" className="fill-amber-700 dark:fill-amber-300 text-[11px] font-mono font-bold">
                = {selectedCell.wDequant.toFixed(3)}
              </text>
              <text x="821" y="226" textAnchor="middle" className="fill-slate-500 dark:fill-slate-400 text-[8.5px]">
                True: {selectedCell.wFp32.toFixed(3)}
              </text>
              <text x="821" y="241" textAnchor="middle" className="fill-rose-600 dark:fill-rose-400 text-[8.5px] font-mono">
                Δ={selectedCell.error.toFixed(3)}
              </text>
            </g>

            {/* Arrow Registers -> Tensor Cores */}
            <path
              d="M 879 180 L 902 180"
              strokeWidth={activeStep === 5 ? 2.5 : 1.5}
              className={
                activeStep === 5
                  ? 'stroke-indigo-500 dark:stroke-indigo-400 animate-dash-flow'
                  : 'stroke-slate-400 dark:stroke-slate-600'
              }
              markerEnd={activeStep === 5 ? 'url(#arrow-active)' : 'url(#arrow-idle)'}
            />

            {/* 6. TENSOR CORES & FP32 ACCUMULATOR */}
            <g
              className="cursor-pointer transition-all"
              onClick={() => setInspectedId('tensor_cores')}
            >
              <rect
                x="908"
                y="90"
                width="90"
                height="182"
                rx="10"
                strokeWidth={isActive('tensor_cores') ? 2.5 : 1.5}
                className={boxClass('tensor_cores')}
              />
              <text
                x="953"
                y="114"
                textAnchor="middle"
                className="fill-slate-900 dark:fill-slate-100 text-[10.5px] font-bold"
              >
                Tensor Cores
              </text>
              <text
                x="953"
                y="129"
                textAnchor="middle"
                className="fill-slate-500 dark:fill-slate-400 text-[8.5px]"
              >
                FP32 Accum
              </text>

              <rect
                x="916"
                y="146"
                width="74"
                height="108"
                rx="6"
                className="fill-indigo-500/10 stroke-indigo-500/30"
              />
              <text x="953" y="168" textAnchor="middle" className="fill-slate-700 dark:fill-slate-200 text-[9.5px] font-mono">
                ŷ = Ŵ · x
              </text>
              <text x="953" y="190" textAnchor="middle" className="fill-indigo-600 dark:fill-indigo-400 text-[10.5px] font-mono font-bold">
                ŷ[{rowIdx}]=
              </text>
              <text x="953" y="206" textAnchor="middle" className="fill-indigo-600 dark:fill-indigo-400 text-[10.5px] font-mono font-bold">
                {yQuantVal.toFixed(2)}
              </text>
              <text x="953" y="232" textAnchor="middle" className="fill-slate-500 dark:fill-slate-400 text-[8px]">
                Write back
              </text>
              <text x="953" y="244" textAnchor="middle" className="fill-slate-500 dark:fill-slate-400 text-[8px]">
                to HBM
              </text>
            </g>

            {/* Return path Tensor Core -> HBM Output */}
            <path
              d="M 953 272 L 953 310 L 417 310"
              fill="none"
              strokeWidth={activeStep === 5 ? 2.5 : 1.5}
              className={
                activeStep === 5
                  ? 'stroke-indigo-500 dark:stroke-indigo-400 animate-dash-flow'
                  : 'stroke-slate-400 dark:stroke-slate-600'
              }
              markerEnd={activeStep === 5 ? 'url(#arrow-active)' : 'url(#arrow-idle)'}
            />
            <text
              x="720"
              y="326"
              textAnchor="middle"
              className="fill-slate-500 dark:fill-slate-400 text-[9.5px] font-mono"
            >
              Write FP32 output ŷ back to HBM (Ŵ is never written back to HBM!)
            </text>
          </svg>
        </div>

        {/* Interactive Hardware Inspector Card */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-indigo-500/15 text-indigo-700 dark:text-indigo-300">
                <Layers className="w-3 h-3" />
                {detail.title}
              </span>
              <span className="text-xs text-slate-500 dark:text-slate-400">
                {detail.subtitle}
              </span>
            </div>
            <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed">
              {detail.role}
            </p>
            <p className="text-xs text-emerald-700 dark:text-emerald-300 flex items-start gap-1.5">
              <Zap className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <span>
                <strong>Why quantization helps:</strong> {detail.quantBenefit}
              </span>
            </p>
          </div>
          {inspectedId && (
            <button
              type="button"
              onClick={() => setInspectedId(null)}
              className="self-start sm:self-center shrink-0 text-xs px-2.5 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:bg-slate-200/60 dark:hover:bg-slate-700/60 text-slate-700 dark:text-slate-200 transition"
            >
              Follow Active Step
            </button>
          )}
        </div>
      </div>
    </section>
  );
};
