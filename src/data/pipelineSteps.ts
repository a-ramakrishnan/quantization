import { GpuComponentId, PipelineStepInfo } from '../types/quantization';

export const PIPELINE_STEPS: PipelineStepInfo[] = [
  {
    id: 0,
    shortTitle: '1. Calibrate FP32',
    title: 'Step 1: Inspect FP32 Weights & Calibrate Range',
    gpuTarget: 'host',
    gpuLabel: 'Offline / Host Calibration',
    formula: 'S = (α_max - α_min) / (q_max - q_min)',
    explanation:
      'Before running inference, we inspect the FP32 weight tensor W to find its minimum and maximum values (optionally clipped by α to tame outliers). From this range and the target bit-width, we compute the Scale (S) and Zero-Point (Z).',
    whyItMatters:
      'A single extreme outlier stretches S, forcing all normal weights into just a few integer buckets and destroying accuracy. Clipping or per-channel/per-group scaling prevents this.',
  },
  {
    id: 1,
    shortTitle: '2. Quantize (FP32→INT)',
    title: 'Step 2: Quantize & Pack Weights to Integers',
    gpuTarget: 'host',
    gpuLabel: 'Quantizer (Offline / Load)',
    formula: 'W_q = clamp(round(W / S) + Z, q_min, q_max)',
    explanation:
      'Every FP32 weight is divided by Scale S, rounded to the nearest integer, shifted by Zero-Point Z, and clamped to [q_min, q_max]. Multiple INT4 or INT2 values can be bit-packed into a single 32-bit word.',
    whyItMatters:
      'Rounding introduces small quantization error, while clamping introduces clipping error. These integer values W_q are what actually get stored in GPU memory.',
  },
  {
    id: 2,
    shortTitle: '3. Store in GPU HBM',
    title: 'Step 3: Store Compressed W_q + (S, Z) in GPU HBM (VRAM)',
    gpuTarget: 'hbm',
    gpuLabel: 'GPU HBM / VRAM',
    formula: 'HBM Footprint = N × b bits + (#groups × 48 bits)',
    explanation:
      'The quantized integer weights W_q and tiny per-tensor/channel/group metadata (Scale S and Zero-Point Z) are transferred over PCIe and stored in GPU High Bandwidth Memory (HBM / VRAM), alongside input activations X.',
    whyItMatters:
      'LLM inference is heavily memory-bandwidth bound. Storing INT4 instead of FP32 cuts HBM footprint and memory traffic by up to ~8× (or ~4× vs FP16).',
  },
  {
    id: 3,
    shortTitle: '4. Stream to SM SRAM',
    title: 'Step 4: Stream Tiles from HBM → L2 Cache → SM Shared Memory',
    gpuTarget: 'sram',
    gpuLabel: 'L2 Cache & SM Shared Memory (SRAM)',
    formula: 'HBM (~2 TB/s) → L2 Cache → Shared Memory (~20+ TB/s)',
    explanation:
      'When the linear layer kernel launches, Streaming Multiprocessors (SMs) fetch tiles of packed W_q, metadata (S, Z), and input vector X from HBM through L2 Cache into ultra-fast on-chip Shared Memory (SRAM).',
    whyItMatters:
      'Because W_q is packed into fewer bits, each memory transaction across the HBM bus brings in 4×–8× more weights per byte.',
  },
  {
    id: 4,
    shortTitle: '5. Dequantize in Registers',
    title: 'Step 5: On-the-Fly Dequantization in SM Register File',
    gpuTarget: 'registers',
    gpuLabel: 'SM Register File / CUDA Cores',
    formula: 'Ŵ = S · (W_q - Z)',
    explanation:
      'Inside each SM, threads unpack W_q into registers and immediately dequantize back to FP16/FP32 using the tile’s Scale S and Zero-Point Z. Notice we never write Ŵ back to HBM!',
    whyItMatters:
      'Fusing dequantization into the matmul kernel keeps the reconstructed float Ŵ purely in fast registers, giving us the memory savings of INT4/INT8 with float compute precision.',
  },
  {
    id: 5,
    shortTitle: '6. Tensor Core Matmul',
    title: 'Step 6: Matrix Multiply (ŷ = Ŵx) in Tensor Cores & Write Output',
    gpuTarget: 'tensor_cores',
    gpuLabel: 'Tensor Cores / FP32 Accumulators',
    formula: 'ŷ_i = Σ_j Ŵ_{i,j} · x_j   (Error = y - ŷ)',
    explanation:
      'Tensor Cores / FMA units multiply the dequantized weights Ŵ by input activations X, accumulate dot products in high-precision FP32 accumulators, and write the final output vector ŷ back through SRAM/L2 to HBM.',
    whyItMatters:
      'Comparing y = Wx (true FP32) against ŷ = Ŵx shows how weight quantization error propagates through the linear layer.',
  },
];

export const GPU_COMPONENT_DETAILS: Record<
  GpuComponentId,
  {
    title: string;
    subtitle: string;
    role: string;
    quantBenefit: string;
  }
> = {
  host: {
    title: 'Host CPU & Quantizer',
    subtitle: 'Calibration & Weight Packing',
    role: 'Loads model weights, computes calibration ranges (min/max or clipped α), derives Scale (S) and Zero-Point (Z), and quantizes FP32 → INTb before uploading to the GPU.',
    quantBenefit: 'Reduces PCIe transfer time and prepares compact packed integers.',
  },
  hbm: {
    title: 'GPU HBM / VRAM (Global Memory)',
    subtitle: 'High-Capacity Off-Chip Memory (~2–3 TB/s)',
    role: 'Stores the full quantized weight matrix W_q, quantization metadata (S, Z), input activations X, and final outputs ŷ.',
    quantBenefit:
      'Primary bottleneck in linear-layer inference! Quantizing weights from FP32/FP16 to INT4/INT8 shrinks VRAM usage and HBM traffic by 2×–8×.',
  },
  l2: {
    title: 'L2 Cache',
    subtitle: 'Shared On-Die Cache Across All SMs',
    role: 'Caches recently fetched tiles of W_q, S, Z, and X from HBM so multiple Streaming Multiprocessors can reuse weights without hitting off-chip HBM again.',
    quantBenefit: 'Smaller quantized weights fit much more easily in L2, boosting cache hit rates.',
  },
  sram: {
    title: 'Shared Memory (SRAM)',
    subtitle: 'Ultra-Fast On-Chip SM Memory (~20+ TB/s)',
    role: 'Software-managed scratchpad inside each SM where thread blocks stage tiles of packed W_q and input vector X for cooperative reuse.',
    quantBenefit: 'Allows fused kernels to stage more weight tiles per SM without spilling.',
  },
  registers: {
    title: 'Register File & Dequantization Unit',
    subtitle: 'Per-Thread Fastest Storage',
    role: 'Unpacks INT2/INT4/INT8 weights from SRAM and executes Ŵ = S · (W_q − Z) in registers right before multiplication.',
    quantBenefit:
      'Zero extra HBM round-trips: Ŵ exists only transiently in registers.',
  },
  tensor_cores: {
    title: 'Tensor Cores & FP32 Accumulators',
    subtitle: 'Matrix Multiply-Accumulate (FMA) Engines',
    role: 'Performs high-throughput dot products ŷ_i = Σ Ŵ_{i,j} · x_j and accumulates partial sums in FP32 to prevent overflow/underflow.',
    quantBenefit:
      'Keeps arithmetic accurate while feeding data at quantized memory speed.',
  },
};
