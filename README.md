# Linear Layer Quantization & GPU Execution Visualizer

An interactive React + TypeScript + Tailwind CSS visualization that demonstrates how neural network weights are **calibrated, quantized, stored in GPU HBM, streamed to SM Shared Memory, dequantized on-the-fly in registers, and multiplied in Tensor Cores** for a linear layer ($y = Wx$).

> Repository: [github.com/a-ramakrishnan/quantization](https://github.com/a-ramakrishnan/quantization)

## Features

- **Interactive GPU Architecture Diagram**
  - Visualizes the full hardware dataflow: **Host CPU / Calibrator → PCIe → GPU HBM (VRAM) → L2 Cache → Streaming Multiprocessor (SM) Shared Memory (SRAM) → Register File (Dequantizer) → Tensor Cores (FP32 Accumulators)**.
  - Live-synchronized with the 6-step playback and displays real numbers from your active weight matrix, plus HBM memory compression stats.
  - Click any hardware block to inspect its role and why quantization accelerates it.
- **Side-by-Side Matrix Pipeline (4×4 Linear Layer)**
  - **FP32 Weights ($W$)**: Directly editable cells with color-coded heatmap.
  - **Quantized Integers ($W_q$)**: Shows exact `INT2`–`INT8` integers and highlights clipped values.
  - **Dequantized Weights ($\hat{W}$)**: Reconstructed float values computed in SM registers.
  - **Quantization Error ($W - \hat{W}$)**: Diverging error heatmap.
- **Number-Line Bin Mapping**
  - Shows continuous FP32 weights snapping to discrete integer quantization levels separated by step size $\Delta = S$, with clipping bounds $[\alpha_{\min}, \alpha_{\max}]$ and Zero-Point $Z$.
- **Live Step-by-Step Math Breakdown**
  - Plugs the selected cell's exact numbers into every formula: Scale $S$, Zero-Point $Z$, Quantization $W_q$, Dequantization $\hat{W}$, and Error $e$.
- **Linear Layer Forward Pass ($y = Wx$ vs. $\hat{y} = \hat{W}x$)**
  - Editable input activation vector $x$ with bar chart comparison and summary metrics: **Weight MSE**, **Weight SQNR (dB)**, **Output MSE**, **Cosine Similarity**, and **HBM Memory Savings**.
- **Rich Controls & Light/Dark Mode**
  - Presets: *Normal Weights*, *With Outliers*, *Shifted / Biased*, *Uniform Spread*, plus *Randomize*.
  - Bit-widths: `INT2`, `INT4`, `INT8`, or any custom bit-width from 2 to 8.
  - Schemes: **Symmetric** ($Z = 0$) vs. **Asymmetric / Affine**.
  - Granularities: **Per-Tensor**, **Per-Channel**, and **Per-Group ($g=2$)**.
  - **Outlier Clipping ($\alpha$)** slider & optional manual Scale / Zero-Point override.

## Quantization Math

### 1. Symmetric Quantization ($Z = 0$)
$$q_{\max} = 2^{b-1} - 1, \quad q_{\min} = -q_{\max}$$
$$S = \frac{\max(|W_{\min}|, |W_{\max}|) \cdot \alpha}{q_{\max}}, \quad W_q = \text{clamp}\left(\text{round}\left(\frac{W}{S}\right), q_{\min}, q_{\max}\right)$$
$$\hat{W} = S \cdot W_q$$

### 2. Asymmetric (Affine) Quantization
$$q_{\max} = 2^{b-1} - 1, \quad q_{\min} = -2^{b-1}$$
$$S = \frac{\alpha_{\max} - \alpha_{\min}}{q_{\max} - q_{\min}}, \quad Z = \text{clamp}\left(\text{round}\left(q_{\min} - \frac{\alpha_{\min}}{S}\right), q_{\min}, q_{\max}\right)$$
$$W_q = \text{clamp}\left(\text{round}\left(\frac{W}{S}\right) + Z, q_{\min}, q_{\max}\right), \quad \hat{W} = S \cdot (W_q - Z)$$

## Getting Started

```bash
npm install
npm run dev
```

Build for production:

```bash
npm run build
```

## Contributing

Every check-in should go through a Pull Request:
1. Create a feature branch (`git checkout -b feat/your-feature`).
2. Commit your changes and run `npm run build` to verify TypeScript & Vite bundling.
3. Open a Pull Request against `main`.
