import { useEffect, useMemo, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Eye,
  Github,
  Layers,
  Moon,
  RotateCcw,
  Shuffle,
  Sliders,
  Sun,
} from 'lucide-react';
import {
  Granularity,
  PresetKey,
  QuantConfig,
  QuantScheme,
} from './types/quantization';
import {
  PRESETS,
  generateRandomWeights,
  quantizeLinearLayer,
} from './utils/quantization';
import { PIPELINE_STEPS } from './data/pipelineSteps';
import { GpuArchitecture } from './components/GpuArchitecture';
import { GpuBoxBreakdown } from './components/GpuBoxBreakdown';
import { MatrixPipeline } from './components/MatrixPipeline';
import { NumberLineVisualizer } from './components/NumberLineVisualizer';
import { MathBreakdown } from './components/MathBreakdown';
import { ForwardPassPanel } from './components/ForwardPassPanel';
import { TensorJourneyPanel } from './components/TensorJourneyPanel';
import { AboutModal } from './components/AboutModal';

const TIMELINE_STAGES = [
  {
    id: 0,
    title: '1. Weights & Calibrate',
    subtitle: 'Provide W & quantization settings → compute Scale (S) & Zero-Point (Z)',
    gpuStep: 0,
  },
  {
    id: 1,
    title: '2. Quantize (FP32 → INT)',
    subtitle: 'Snap continuous floats to discrete integers W_q',
    gpuStep: 1,
  },
  {
    id: 2,
    title: '3. GPU HBM & Dequantize',
    subtitle: 'Store W_q in HBM, stream to SM SRAM, dequantize Ŵ in registers',
    gpuStep: 4,
  },
  {
    id: 3,
    title: '4. Forward Pass (y = Wx)',
    subtitle: 'Provide input vector x → compute ŷ = Ŵx in Tensor Cores & compare error',
    gpuStep: 5,
  },
];

export function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [aboutOpen, setAboutOpen] = useState(false);

  // Timeline state: start clean on Stage 0
  const [timelineStage, setTimelineStage] = useState<number>(0);
  const [highestStage, setHighestStage] = useState<number>(0);
  const [showAllMode, setShowAllMode] = useState<boolean>(false);
  const [gpuSubStep, setGpuSubStep] = useState<number>(4); // 2=HBM, 3=SRAM, 4=Registers
  const [gpuViewMode, setGpuViewMode] = useState<'breakdown' | 'reconstructed'>('breakdown');

  const [presetKey, setPresetKey] = useState<PresetKey>('outliers');
  const [weights, setWeights] = useState<number[][]>(() =>
    PRESETS.outliers.weights.map((r) => [...r])
  );
  const [inputs, setInputs] = useState<number[]>(() => [
    ...PRESETS.outliers.inputs,
  ]);

  const [config, setConfig] = useState<QuantConfig>({
    bits: 4,
    scheme: 'symmetric',
    granularity: 'per-tensor',
    clipRatio: 1.0,
    manualOverride: false,
    manualScale: 0.5,
    manualZeroPoint: 0,
  });

  const [selectedPos, setSelectedPos] = useState<{ row: number; col: number }>({
    row: 0,
    col: 2,
  });

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) root.classList.add('dark');
    else root.classList.remove('dark');
  }, [darkMode]);

  const result = useMemo(
    () => quantizeLinearLayer(weights, inputs, config),
    [weights, inputs, config]
  );

  const selectedCell =
    result.cells[selectedPos.row]?.[selectedPos.col] ?? result.cells[0][0];

  const selectedGroup =
    result.groups.find((g) => g.id === selectedCell.groupId) ?? result.groups[0];

  const allCellsInGroup = useMemo(
    () =>
      result.cells
        .flat()
        .filter((c) => c.groupId === selectedCell.groupId),
    [result.cells, selectedCell.groupId]
  );

  const goToStage = (stage: number) => {
    setTimelineStage(stage);
    setHighestStage((prev) => Math.max(prev, stage));
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleSelectPreset = (key: PresetKey) => {
    setPresetKey(key);
    setWeights(PRESETS[key].weights.map((r) => [...r]));
    setInputs([...PRESETS[key].inputs]);
    setConfig((prev) => ({ ...prev, manualOverride: false }));
  };

  const handleRandomize = () => {
    setWeights(generateRandomWeights());
    setConfig((prev) => ({ ...prev, manualOverride: false }));
  };

  const handleUpdateWeight = (r: number, c: number, val: number) => {
    setWeights((prev) =>
      prev.map((rowArr, ri) =>
        ri === r ? rowArr.map((w, ci) => (ci === c ? val : w)) : rowArr
      )
    );
  };

  const handleUpdateInput = (idx: number, val: number) => {
    setInputs((prev) => prev.map((x, i) => (i === idx ? val : x)));
  };

  const activeGpuStep =
    timelineStage === 2 ? gpuSubStep : TIMELINE_STAGES[timelineStage].gpuStep;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
      {/* Top Navbar */}
      <header className="sticky top-0 z-30 border-b border-slate-200 dark:border-slate-800 bg-white/85 dark:bg-slate-950/85 backdrop-blur-md">
        <div className="max-w-[1320px] mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white font-mono font-bold text-sm shadow-sm">
              Q{config.bits}
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-slate-900 dark:text-slate-100">
                Linear Layer Quantization &amp; GPU Execution
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Guided Timeline: FP32 Weights → INT{config.bits} → GPU Dequantize → y = Wx
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowAllMode((v) => !v)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
                showAllMode
                  ? 'border-indigo-500 bg-indigo-500/10 text-indigo-600 dark:text-indigo-400'
                  : 'border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800'
              }`}
              title="Toggle between step-by-step timeline and full dashboard view"
            >
              <Eye className="w-3.5 h-3.5" />
              {showAllMode ? 'Timeline View' : 'Show All'}
            </button>
            <button
              type="button"
              onClick={() => setAboutOpen(true)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
            >
              <BookOpen className="w-3.5 h-3.5 text-indigo-500" />
              About
            </button>
            <a
              href="https://github.com/a-ramakrishnan/quantization"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
            >
              <Github className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Repo</span>
            </a>
            <button
              type="button"
              onClick={() => setDarkMode((d) => !d)}
              className="p-2 rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
              aria-label="Toggle Light/Dark Mode"
              title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {darkMode ? (
                <Sun className="w-4 h-4 text-amber-400" />
              ) : (
                <Moon className="w-4 h-4 text-slate-700" />
              )}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-[1320px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Interactive Timeline Stepper */}
        {!showAllMode && (
          <nav
            aria-label="Quantization Timeline"
            className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-3 sm:p-4"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
              {TIMELINE_STAGES.map((stage) => {
                const isCurrent = timelineStage === stage.id;
                const isDone = stage.id < timelineStage || stage.id <= highestStage;
                return (
                  <button
                    key={stage.id}
                    type="button"
                    onClick={() => goToStage(stage.id)}
                    className={`text-left rounded-xl p-3 border transition flex items-start gap-2.5 ${
                      isCurrent
                        ? 'border-indigo-500 bg-indigo-500/10 dark:bg-indigo-950/50 shadow-xs'
                        : isDone
                        ? 'border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/40 hover:border-indigo-400/60'
                        : 'border-slate-200/70 dark:border-slate-800/70 opacity-75 hover:opacity-100'
                    }`}
                  >
                    <div
                      className={`mt-0.5 w-5 h-5 rounded-full flex items-center justify-center text-[11px] font-bold shrink-0 ${
                        isCurrent
                          ? 'bg-indigo-600 text-white'
                          : isDone
                          ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
                          : 'bg-slate-200 dark:bg-slate-800 text-slate-500'
                      }`}
                    >
                      {isDone && !isCurrent ? (
                        <CheckCircle2 className="w-3.5 h-3.5" />
                      ) : (
                        stage.id + 1
                      )}
                    </div>
                    <div className="min-w-0">
                      <div
                        className={`text-xs font-semibold truncate ${
                          isCurrent
                            ? 'text-indigo-700 dark:text-indigo-300'
                            : 'text-slate-800 dark:text-slate-200'
                        }`}
                      >
                        {stage.title}
                      </div>
                      <div className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-2 mt-0.5">
                        {stage.subtitle}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </nav>
        )}

        {/* STAGE 1: INPUT WEIGHTS & CALIBRATION */}
        {(showAllMode || timelineStage === 0) && (
          <div className="space-y-5">
            {/* Stage 1 Explanation Header */}
            <div className="rounded-2xl border border-indigo-500/25 bg-gradient-to-r from-indigo-500/5 via-white to-violet-500/5 dark:from-indigo-950/30 dark:via-slate-900 dark:to-violet-950/20 p-4 sm:p-5 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                  Stage 1: Provide FP32 Weights &amp; Calibration Settings
                </h2>
                <span className="text-xs font-mono px-2.5 py-1 rounded-full bg-indigo-500/10 text-indigo-700 dark:text-indigo-300">
                  First-Level Output: Scale (S) &amp; Zero-Point (Z)
                </span>
              </div>
              <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
                Start by choosing a weight preset (or editing any cell in the 4×4 matrix{' '}
                <span className="font-mono font-semibold">W</span>) and selecting your quantization
                settings. Before any integers are created, we calibrate the range{' '}
                <span className="font-mono">[α_min, α_max]</span> to compute{' '}
                <strong>Scale (S)</strong> and <strong>Zero-Point (Z)</strong>.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
              {/* Left: Controls + Editable FP32 Matrix */}
              <div className="lg:col-span-7 space-y-4">
                <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 space-y-4">
                  <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 dark:text-slate-200">
                    <Sliders className="w-4 h-4 text-indigo-500" />
                    1A. Choose Preset &amp; Quantization Parameters
                  </div>

                  {/* Presets */}
                  <div className="space-y-1.5">
                    <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                      Weight Distribution Preset
                    </label>
                    <div className="flex flex-wrap gap-1.5">
                      {(Object.keys(PRESETS) as PresetKey[]).map((key) => (
                        <button
                          key={key}
                          type="button"
                          onClick={() => handleSelectPreset(key)}
                          className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition ${
                            presetKey === key
                              ? 'bg-indigo-600 text-white shadow-xs'
                              : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                          }`}
                        >
                          {PRESETS[key].name}
                        </button>
                      ))}
                      <button
                        type="button"
                        onClick={handleRandomize}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition"
                      >
                        <Shuffle className="w-3 h-3" />
                        Random
                      </button>
                    </div>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      {PRESETS[presetKey].description}
                    </p>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-slate-200 dark:border-slate-800">
                    {/* Bit-Width */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                          Bit-Width
                        </label>
                        <span className="text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400">
                          INT{config.bits} ({Math.pow(2, config.bits)} levels)
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        {[2, 4, 8].map((b) => (
                          <button
                            key={b}
                            type="button"
                            onClick={() => setConfig((c) => ({ ...c, bits: b }))}
                            className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold transition ${
                              config.bits === b
                                ? 'bg-indigo-600 text-white'
                                : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                            }`}
                          >
                            INT{b}
                          </button>
                        ))}
                        <input
                          type="range"
                          min={2}
                          max={8}
                          value={config.bits}
                          onChange={(e) =>
                            setConfig((c) => ({ ...c, bits: Number(e.target.value) }))
                          }
                          aria-label="Bit width slider"
                          className="flex-1 accent-indigo-600"
                        />
                      </div>
                    </div>

                    {/* Scheme */}
                    <div className="space-y-1.5">
                      <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        Quantization Scheme
                      </label>
                      <div className="grid grid-cols-2 gap-1.5">
                        {(['symmetric', 'asymmetric'] as QuantScheme[]).map((scheme) => (
                          <button
                            key={scheme}
                            type="button"
                            onClick={() => setConfig((c) => ({ ...c, scheme }))}
                            className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition ${
                              config.scheme === scheme
                                ? 'bg-indigo-600 text-white'
                                : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                            }`}
                          >
                            {scheme === 'symmetric' ? 'Symmetric (Z=0)' : 'Asymmetric'}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Granularity */}
                    <div className="space-y-1.5">
                      <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                        Scale Granularity
                      </label>
                      <div className="grid grid-cols-3 gap-1">
                        {(
                          [
                            ['per-tensor', 'Tensor'],
                            ['per-channel', 'Channel'],
                            ['per-group', 'Group(2)'],
                          ] as [Granularity, string][]
                        ).map(([gran, label]) => (
                          <button
                            key={gran}
                            type="button"
                            onClick={() =>
                              setConfig((c) => ({
                                ...c,
                                granularity: gran,
                                manualOverride:
                                  gran === 'per-tensor' ? c.manualOverride : false,
                              }))
                            }
                            className={`px-2 py-1.5 rounded-lg text-xs font-medium transition ${
                              config.granularity === gran
                                ? 'bg-indigo-600 text-white'
                                : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Clipping */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                          Outlier Clipping (α)
                        </label>
                        <span className="text-xs font-mono font-semibold text-amber-600 dark:text-amber-400">
                          {(config.clipRatio * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <input
                          type="range"
                          min={0.2}
                          max={1.0}
                          step={0.05}
                          value={config.clipRatio}
                          onChange={(e) =>
                            setConfig((c) => ({
                              ...c,
                              clipRatio: Number(e.target.value),
                              manualOverride: false,
                            }))
                          }
                          aria-label="Outlier clipping ratio"
                          className="flex-1 accent-amber-500"
                        />
                        {config.clipRatio < 1 && (
                          <button
                            type="button"
                            onClick={() =>
                              setConfig((c) => ({ ...c, clipRatio: 1.0 }))
                            }
                            className="text-[11px] px-2 py-1 rounded bg-slate-100 dark:bg-slate-800"
                          >
                            Reset
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </section>

                <MatrixPipeline
                  weights={weights}
                  result={result}
                  config={config}
                  selectedPos={selectedPos}
                  onSelectCell={(row, col) => setSelectedPos({ row, col })}
                  onUpdateWeight={handleUpdateWeight}
                  activeStep={0}
                  visibleColumns={['fp32']}
                  title="1B. Input FP32 Weight Matrix (W)"
                  subtitle="Click any number to edit it directly. Click a cell to inspect its group's calibration."
                />
              </div>

              {/* Right: First-Level Output (Calibration Result) */}
              <div className="lg:col-span-5 space-y-4">
                <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 sm:p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                        <Layers className="w-4 h-4" />
                      </div>
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                          First-Level Output: Calibration Parameters
                        </h3>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                          Computed for <span className="font-medium">{selectedGroup.label}</span>
                        </p>
                      </div>
                    </div>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800">
                      {result.groups.length} group{result.groups.length > 1 ? 's' : ''}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3">
                      <div className="text-[11px] text-slate-500 dark:text-slate-400">
                        Observed Range [min, max]
                      </div>
                      <div className="mt-1 text-sm font-mono font-bold text-slate-900 dark:text-slate-100">
                        [{selectedGroup.minVal.toFixed(2)}, {selectedGroup.maxVal.toFixed(2)}]
                      </div>
                    </div>
                    <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50 p-3">
                      <div className="text-[11px] text-slate-500 dark:text-slate-400">
                        Target Integer Range
                      </div>
                      <div className="mt-1 text-sm font-mono font-bold text-emerald-600 dark:text-emerald-400">
                        [{selectedGroup.qMin}, {selectedGroup.qMax}] (INT{config.bits})
                      </div>
                    </div>
                    <div className="rounded-xl border border-indigo-500/30 bg-indigo-500/5 p-3">
                      <div className="text-[11px] text-indigo-600 dark:text-indigo-400 font-medium">
                        Computed Scale (S)
                      </div>
                      <div className="mt-1 text-lg font-mono font-bold text-indigo-700 dark:text-indigo-300">
                        {selectedGroup.scale.toFixed(5)}
                      </div>
                    </div>
                    <div className="rounded-xl border border-indigo-500/30 bg-indigo-500/5 p-3">
                      <div className="text-[11px] text-indigo-600 dark:text-indigo-400 font-medium">
                        Computed Zero-Point (Z)
                      </div>
                      <div className="mt-1 text-lg font-mono font-bold text-indigo-700 dark:text-indigo-300">
                        {selectedGroup.zeroPoint}
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl bg-slate-50 dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800 p-3 space-y-1.5 text-xs text-slate-600 dark:text-slate-300">
                    <div className="font-semibold text-slate-800 dark:text-slate-200">
                      What happens next?
                    </div>
                    <p className="leading-relaxed">
                      Every float in this group will be divided by{' '}
                      <span className="font-mono font-semibold text-indigo-600 dark:text-indigo-400">
                        S = {selectedGroup.scale.toFixed(4)}
                      </span>
                      , shifted by{' '}
                      <span className="font-mono font-semibold text-indigo-600 dark:text-indigo-400">
                        Z = {selectedGroup.zeroPoint}
                      </span>
                      , and rounded to an integer in{' '}
                      <span className="font-mono">
                        [{selectedGroup.qMin}, {selectedGroup.qMax}]
                      </span>
                      .
                    </p>
                  </div>

                  {!showAllMode && (
                    <button
                      type="button"
                      onClick={() => goToStage(1)}
                      className="w-full py-2.5 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold shadow-sm flex items-center justify-center gap-2 transition"
                    >
                      Quantize Weights (FP32 → INT{config.bits})
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  )}
                </section>
              </div>
            </div>
          </div>
        )}

        {/* STAGE 2: QUANTIZE (FP32 -> INT) */}
        {(showAllMode || timelineStage === 1) && (
          <div className="space-y-5">
            {!showAllMode && (
              <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 px-4 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="font-semibold text-slate-700 dark:text-slate-200">
                    Stage 1 Inputs:
                  </span>
                  <span className="font-mono px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800">
                    {PRESETS[presetKey].name}
                  </span>
                  <span className="font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">
                    INT{config.bits} · {config.scheme} · {config.granularity}
                  </span>
                  <span className="font-mono text-slate-500 dark:text-slate-400">
                    S={selectedGroup.scale.toFixed(4)}, Z={selectedGroup.zeroPoint}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => goToStage(0)}
                  className="text-indigo-600 dark:text-indigo-400 font-medium hover:underline"
                >
                  Edit Stage 1
                </button>
              </div>
            )}

            <div className="rounded-2xl border border-indigo-500/25 bg-gradient-to-r from-indigo-500/5 via-white to-violet-500/5 dark:from-indigo-950/30 dark:via-slate-900 dark:to-violet-950/20 p-4 sm:p-5 space-y-1.5">
              <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                Stage 2: Quantize &amp; Pack (FP32 → INT{config.bits})
              </h2>
              <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
                Continuous FP32 weights are mapped onto discrete integer buckets via{' '}
                <span className="font-mono">W_q = clamp(round(W / S) + Z, q_min, q_max)</span>.
                Click any cell to see it snap on the number line below.
              </p>
            </div>

            <MatrixPipeline
              weights={weights}
              result={result}
              config={config}
              selectedPos={selectedPos}
              onSelectCell={(row, col) => setSelectedPos({ row, col })}
              onUpdateWeight={handleUpdateWeight}
              activeStep={1}
              visibleColumns={['fp32', 'quantized']}
              title="FP32 Weights (W) → Quantized Integers (W_q)"
              subtitle="Notice how extreme outliers force smaller weights into identical integer buckets unless clipped or grouped."
            />

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-7">
                <NumberLineVisualizer
                  selectedCell={selectedCell}
                  groupParams={selectedGroup}
                  allCellsInGroup={allCellsInGroup}
                />
              </div>
              <div className="lg:col-span-5">
                <MathBreakdown
                  selectedCell={selectedCell}
                  groupParams={selectedGroup}
                  config={config}
                />
              </div>
            </div>

            <TensorJourneyPanel
              activeRowWeights={weights[selectedPos.row] ?? weights[0]}
              activeRowIndex={selectedPos.row}
            />

            {!showAllMode && (
              <div className="flex items-center justify-between pt-1">
                <button
                  type="button"
                  onClick={() => goToStage(0)}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-semibold transition"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back to Calibration
                </button>
                <button
                  type="button"
                  onClick={() => goToStage(2)}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold shadow-sm transition"
                >
                  Upload to GPU &amp; Dequantize
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        )}

        {/* STAGE 3: GPU HBM & ON-THE-FLY DEQUANTIZATION */}
        {(showAllMode || timelineStage === 2) && (
          <div className="space-y-5">
            <div className="rounded-2xl border border-indigo-500/25 bg-gradient-to-r from-indigo-500/5 via-white to-violet-500/5 dark:from-indigo-950/30 dark:via-slate-900 dark:to-violet-950/20 p-4 sm:p-5 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                  Stage 3: GPU Memory Hierarchy &amp; On-the-Fly Dequantization
                </h2>
                {!showAllMode && (
                  <div className="flex flex-wrap items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => setGpuViewMode('breakdown')}
                      className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
                        gpuViewMode === 'breakdown'
                          ? 'bg-indigo-600 text-white'
                          : 'bg-white/80 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700'
                      }`}
                    >
                      Break Down Boxes
                    </button>
                    <button
                      type="button"
                      onClick={() => setGpuViewMode('reconstructed')}
                      className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
                        gpuViewMode === 'reconstructed'
                          ? 'bg-emerald-600 text-white'
                          : 'bg-white/80 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700'
                      }`}
                    >
                      Reconstruct Together
                    </button>
                    {gpuViewMode === 'reconstructed' &&
                      [
                        { step: 2, label: 'HBM' },
                        { step: 3, label: 'SRAM' },
                        { step: 4, label: 'Registers' },
                      ].map((s) => (
                        <button
                          key={s.step}
                          type="button"
                          onClick={() => setGpuSubStep(s.step)}
                          className={`px-2 py-1 rounded-lg text-xs font-medium transition ${
                            gpuSubStep === s.step
                              ? 'bg-indigo-500 text-white'
                              : 'bg-white/80 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700'
                          }`}
                        >
                          {s.label}
                        </button>
                      ))}
                  </div>
                )}
              </div>
              <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
                {gpuViewMode === 'breakdown' && !showAllMode
                  ? 'Inspect each hardware box independently to see how your weight row changes inside it, then reconstruct all boxes together into the full GPU architecture.'
                  : PIPELINE_STEPS[activeGpuStep]?.explanation}
              </p>
            </div>

            {(showAllMode || gpuViewMode === 'breakdown') && (
              <GpuBoxBreakdown
                config={config}
                result={result}
                selectedCell={selectedCell}
                inputs={inputs}
                onReconstructAll={() => setGpuViewMode('reconstructed')}
              />
            )}

            {(showAllMode || gpuViewMode === 'reconstructed') && (
              <GpuArchitecture
                activeStep={activeGpuStep}
                config={config}
                result={result}
                selectedCell={selectedCell}
                inputs={inputs}
              />
            )}

            <MatrixPipeline
              weights={weights}
              result={result}
              config={config}
              selectedPos={selectedPos}
              onSelectCell={(row, col) => setSelectedPos({ row, col })}
              onUpdateWeight={handleUpdateWeight}
              activeStep={4}
              visibleColumns={['quantized', 'dequantized', 'error']}
              title="HBM Integer Weights (W_q) → Register Dequantized (Ŵ) & Error (W − Ŵ)"
              subtitle="Ŵ is reconstructed on-the-fly in SM registers and never written back to HBM."
            />

            {!showAllMode && (
              <div className="flex items-center justify-between pt-1">
                <button
                  type="button"
                  onClick={() => goToStage(1)}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-semibold transition"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back to Quantize
                </button>
                <button
                  type="button"
                  onClick={() => goToStage(3)}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold shadow-sm transition"
                >
                  Provide Input Vector x &amp; Run Linear Layer
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        )}

        {/* STAGE 4: LINEAR LAYER FORWARD PASS (y = Wx) */}
        {(showAllMode || timelineStage === 3) && (
          <div className="space-y-5">
            <div className="rounded-2xl border border-emerald-500/25 bg-gradient-to-r from-emerald-500/5 via-white to-indigo-500/5 dark:from-emerald-950/30 dark:via-slate-900 dark:to-indigo-950/20 p-4 sm:p-5 space-y-1.5">
              <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                Stage 4: Provide Input Activation (x) &amp; Execute Linear Layer (ŷ = Ŵx)
              </h2>
              <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
                Now edit the input vector <span className="font-mono font-semibold">x</span> below.
                Tensor Cores multiply the dequantized weights{' '}
                <span className="font-mono font-semibold">Ŵ</span> by{' '}
                <span className="font-mono font-semibold">x</span>, accumulate in FP32, and write{' '}
                <span className="font-mono font-semibold">ŷ</span> back to HBM. Compare{' '}
                <span className="font-mono">ŷ</span> against the true unquantized{' '}
                <span className="font-mono">y = Wx</span>!
              </p>
            </div>

            <ForwardPassPanel
              inputs={inputs}
              onUpdateInput={handleUpdateInput}
              result={result}
            />

            {!showAllMode && (
              <div className="flex items-center justify-between pt-1">
                <button
                  type="button"
                  onClick={() => goToStage(2)}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-semibold transition"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back to GPU &amp; Dequantize
                </button>
                <button
                  type="button"
                  onClick={() => goToStage(0)}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-indigo-500/40 bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-700 dark:text-indigo-300 text-xs font-semibold transition"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Restart Timeline
                </button>
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <footer className="pt-2 pb-6 text-center text-xs text-slate-500 dark:text-slate-400 flex flex-wrap items-center justify-center gap-3">
          <span>
            Built for understanding Neural Network &amp; LLM Weight Quantization on GPUs
          </span>
          <span>·</span>
          <button
            type="button"
            onClick={() => setAboutOpen(true)}
            className="text-indigo-600 dark:text-indigo-400 hover:underline font-medium"
          >
            About &amp; Architecture Guide
          </button>
          <span>·</span>
          <a
            href="https://github.com/a-ramakrishnan/quantization"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-indigo-600 dark:text-indigo-400 hover:underline font-medium"
          >
            <Github className="w-3.5 h-3.5" />
            a-ramakrishnan/quantization
          </a>
        </footer>
      </main>

      <AboutModal open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  );
}

export default App;
