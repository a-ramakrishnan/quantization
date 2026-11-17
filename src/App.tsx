import { useEffect, useMemo, useState } from 'react';
import {
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Github,
  Moon,
  Pause,
  Play,
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
import { MatrixPipeline } from './components/MatrixPipeline';
import { NumberLineVisualizer } from './components/NumberLineVisualizer';
import { MathBreakdown } from './components/MathBreakdown';
import { ForwardPassPanel } from './components/ForwardPassPanel';
import { AboutModal } from './components/AboutModal';

export function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [aboutOpen, setAboutOpen] = useState(false);

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

  const [activeStep, setActiveStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) root.classList.add('dark');
    else root.classList.remove('dark');
  }, [darkMode]);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = window.setInterval(() => {
      setActiveStep((prev) => (prev + 1) % PIPELINE_STEPS.length);
    }, 2800);
    return () => window.clearInterval(timer);
  }, [isPlaying]);

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

  const currentStepInfo = PIPELINE_STEPS[activeStep] ?? PIPELINE_STEPS[0];

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
      {/* Top Navbar */}
      <header className="sticky top-0 z-30 border-b border-slate-200 dark:border-slate-800 bg-white/85 dark:bg-slate-950/85 backdrop-blur-md">
        <div className="max-w-[1400px] mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white font-mono font-bold text-sm shadow-sm">
              Q4
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-slate-900 dark:text-slate-100">
                Linear Layer Quantization &amp; GPU Execution
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Interactive FP32 → INT{config.bits} → Dequantized Ŵ → Tensor Core Matmul Visualizer
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
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

      <main className="max-w-[1400px] mx-auto px-4 sm:px-6 py-5 space-y-5">
        {/* Controls Panel */}
        <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/80 shadow-sm p-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 dark:text-slate-200">
              <Sliders className="w-4 h-4 text-indigo-500" />
              Quantization &amp; Calibration Controls
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {PRESETS[presetKey].description}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
            {/* Presets */}
            <div className="space-y-1.5">
              <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Weight Preset
              </label>
              <div className="flex flex-wrap gap-1.5">
                {(Object.keys(PRESETS) as PresetKey[]).map((key) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => handleSelectPreset(key)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
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
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition"
                  title="Generate random weights"
                >
                  <Shuffle className="w-3 h-3" />
                  Random
                </button>
              </div>
            </div>

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
                    className={`px-2.5 py-1.5 rounded-lg text-xs font-medium capitalize transition ${
                      config.scheme === scheme
                        ? 'bg-indigo-600 text-white'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                    }`}
                  >
                    {scheme === 'symmetric' ? 'Symmetric (Z=0)' : 'Asymmetric (Affine)'}
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
                        manualOverride: gran === 'per-tensor' ? c.manualOverride : false,
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

            {/* Calibration / Clipping Ratio */}
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
                    onClick={() => setConfig((c) => ({ ...c, clipRatio: 1.0 }))}
                    className="text-[11px] px-2 py-1 rounded bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700"
                  >
                    100%
                  </button>
                )}
              </div>
              {config.granularity === 'per-tensor' && (
                <label className="flex items-center gap-1.5 text-[11px] text-slate-600 dark:text-slate-400 cursor-pointer pt-0.5">
                  <input
                    type="checkbox"
                    checked={config.manualOverride}
                    onChange={(e) =>
                      setConfig((c) => ({
                        ...c,
                        manualOverride: e.target.checked,
                        manualScale: selectedGroup.scale,
                        manualZeroPoint: selectedGroup.zeroPoint,
                      }))
                    }
                    className="rounded accent-indigo-600"
                  />
                  Manual S / Z override
                </label>
              )}
            </div>
          </div>

          {/* Manual Scale/ZP Sliders when enabled */}
          {config.manualOverride && config.granularity === 'per-tensor' && (
            <div className="pt-3 border-t border-slate-200 dark:border-slate-800 grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono w-28">
                  Scale S = {config.manualScale.toFixed(3)}
                </span>
                <input
                  type="range"
                  min={0.02}
                  max={2.0}
                  step={0.01}
                  value={config.manualScale}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      manualScale: Number(e.target.value),
                    }))
                  }
                  className="flex-1 accent-indigo-600"
                />
              </div>
              {config.scheme === 'asymmetric' && (
                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono w-28">
                    Zero-Pt Z = {config.manualZeroPoint}
                  </span>
                  <input
                    type="range"
                    min={result.qMin}
                    max={result.qMax}
                    step={1}
                    value={config.manualZeroPoint}
                    onChange={(e) =>
                      setConfig((c) => ({
                        ...c,
                        manualZeroPoint: Number(e.target.value),
                      }))
                    }
                    className="flex-1 accent-indigo-600"
                  />
                </div>
              )}
            </div>
          )}
        </section>

        {/* Step-by-Step Playback & Explanation Banner */}
        <section className="rounded-2xl border border-indigo-500/30 bg-gradient-to-r from-indigo-500/5 via-white to-violet-500/5 dark:from-indigo-950/40 dark:via-slate-900 dark:to-violet-950/30 shadow-sm p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-1.5">
              {PIPELINE_STEPS.map((step) => (
                <button
                  key={step.id}
                  type="button"
                  onClick={() => {
                    setIsPlaying(false);
                    setActiveStep(step.id);
                  }}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition ${
                    activeStep === step.id
                      ? 'bg-indigo-600 text-white shadow-xs'
                      : 'bg-white/80 dark:bg-slate-800/80 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-700'
                  }`}
                >
                  {step.shortTitle}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  setActiveStep(
                    (s) => (s - 1 + PIPELINE_STEPS.length) % PIPELINE_STEPS.length
                  );
                }}
                className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700"
                aria-label="Previous step"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={() => setIsPlaying((p) => !p)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition"
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-3.5 h-3.5" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" />
                    Play Steps
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  setActiveStep((s) => (s + 1) % PIPELINE_STEPS.length);
                }}
                className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700"
                aria-label="Next step"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={() => {
                  setIsPlaying(false);
                  setActiveStep(0);
                }}
                className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700"
                aria-label="Reset steps"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="rounded-xl border border-slate-200/80 dark:border-slate-800 bg-white/90 dark:bg-slate-950/60 p-3.5 grid grid-cols-1 lg:grid-cols-3 gap-3 items-start">
            <div className="lg:col-span-2 space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  {currentStepInfo.title}
                </h3>
                <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-indigo-500/10 text-indigo-700 dark:text-indigo-300">
                  {currentStepInfo.gpuLabel}
                </span>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                {currentStepInfo.explanation}
              </p>
            </div>
            <div className="rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-2.5 space-y-1">
              <div className="text-[11px] font-mono font-semibold text-indigo-600 dark:text-indigo-400">
                {currentStepInfo.formula}
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
                <strong>Why it matters:</strong> {currentStepInfo.whyItMatters}
              </p>
            </div>
          </div>
        </section>

        {/* 1. GPU Architecture Diagram */}
        <GpuArchitecture
          activeStep={activeStep}
          config={config}
          result={result}
          selectedCell={selectedCell}
          inputs={inputs}
        />

        {/* 2. Side-by-Side Weight Matrices */}
        <MatrixPipeline
          weights={weights}
          result={result}
          config={config}
          selectedPos={selectedPos}
          onSelectCell={(row, col) => setSelectedPos({ row, col })}
          onUpdateWeight={handleUpdateWeight}
          activeStep={activeStep}
        />

        {/* 3. Number-Line + Live Math Breakdown */}
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

        {/* 4. Linear Layer Forward Pass */}
        <ForwardPassPanel
          inputs={inputs}
          onUpdateInput={handleUpdateInput}
          result={result}
        />

        {/* Footer / Inline About */}
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
