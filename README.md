# int_spectrum

DSP frequency discriminator and multi-tone spectral analyzer using crude integer methods, Bresenham remainder distribution, and joint least-squares fitting.

## Overview

`int_spectrum` is a C++17 fixed-point spectral decomposition library and evaluation suite designed for bounded-memory embedded targets where dynamic heap allocation and traditional floating-point FFTs are undesirable. It supports custom, non-uniform frequency grids (e.g., logarithmic, linear, Bark/ERB, or harmonic scales) and iterative multi-tone fitting with Wiener coefficient shrinkage.

The repository includes:
- **Core Library (`include/int_spectrum/`)**: C++ template headers (`bresenham_spectrum.hpp`, `multi_tone_comb.hpp`) and C API header (`c_api.h`).
- **Shared Library (`libint_spectrum.so`)**: C-compatible shared library for integration with C, C++, Python, and GUIs.
- **Evaluation & Test Suite (`tests/`)**: C++ unit tests, Python ctypes unit tests, and automated benchmark evaluation runner.
- **Linux GUI Application (`src/int_spectrum_gui.py` / `bin/int_spectrum_gui`)**: Desktop GUI application for real-time evaluation, interactive parameter tuning, signal generation, audio file testing, and live audio analysis.

---

## Features

- **Integer Modular Phase Arithmetic**: 32-bit unsigned phase accumulators wrap modulo $2^{32}$. Frequency increments are scheduled with Bresenham quotient/remainder accumulators.
- **Fixed-Point Correlations**: Q15 lookup-table basis functions and Q8 residual/model sample units.
- **Custom Search Grids**: Arbitrary caller-supplied frequency grids (logarithmic-Hz, linear-Hz, or custom Q16 DFT-bin arrays).
- **Multi-Tone Peeling & Joint Refitting**: Greedy matching pursuit combined with a centered Gram matrix solver using pivoted fixed-point Gaussian elimination.
- **Wiener Coefficient Shrinkage**: Empirical-Bayes / Wiener power gain filtering for noise suppression.
- **Interactive Graphical Interface**: Full Linux desktop GUI built with Tkinter and Matplotlib for real-time time-domain, spectral, and tabular visualization.

---

## Building

### Requirements
- C++17 compiler (`g++` or `clang++`)
- `make`
- Python 3 with `numpy`, `scipy`, `matplotlib`, and `tkinter` (optional `sounddevice` for microphone streaming)

### Build Commands

```sh
# Build C++ shared library, examples, and tests
make

# Run unit tests (C++ and Python)
make test

# Run full evaluation benchmark suite
make eval

# Launch the Linux GUI application
make gui
```

---

## Architecture & C API

The shared library `libint_spectrum.so` exports C functions for dynamic block sizes $N \in \{64, 128, 256, 512, 1024, 2048, 4096\}$:

```c
#include "int_spectrum/c_api.h"

// Generate a 129-point logarithmic grid from 100 Hz to 700 Hz
uint32_t gridQ16[129];
int_spectrum_make_log_grid(512, 100.0, 700.0, 8000.0, gridQ16, 129);

// Decompose sample buffer
IntSpectrumTone tones[5];
size_t toneCount = 0;
int64_t residualQ8[512];
int32_t removedDc = 0;

int ok = int_spectrum_decompose_grid(
    512, samples, gridQ16, 129,
    500 /* min amplitude */, 5 /* max tones */,
    tones, &toneCount, residualQ8, &removedDc,
    NULL, NULL
);
```

---

## Linux GUI Application

The GUI launcher is located at `bin/int_spectrum_gui` (or `make gui`).

### GUI Capabilities:
1. **Signal Inputs**:
   - **Synthetic Generator**: Configurable multi-tone synthesizer with DC offset and Gaussian noise controls.
   - **WAV Audio File**: Load and inspect 8-bit or 16-bit PCM WAV audio files.
   - **Live / Continuous Input**: Live microphone input via PortAudio / `sounddevice` with automatic fallback to simulated real-time stream.
2. **DSP Controls**:
   - Select search modes: Logarithmic Grid, Linear Grid, or Two-Bank Comb.
   - Adjust block size $N$, sample rate $F_s$, maximum tone count, and detection thresholds.
   - Toggle Wiener coefficient shrinkage filtering.
3. **Visualizations**:
   - **Time-Domain Plot**: Overlay of input signal, decomposed model, and residual error.
   - **Frequency-Domain Plot**: High-resolution reference FFT spectrum overlay with detected integer-grid tone markers.
   - **Results Table**: Detailed detection table showing Q16 bin, converted Hz, peak amplitude, and Wiener gain.

---

## Evaluation Suite Results

Run `make eval` to execute the benchmark suite:

```sh
=========================================================================
           INT_SPECTRUM DSP ALGORITHM EVALUATION SUITE
=========================================================================
Configuration: N=512, Fs=8000.0 Hz, Grid=129 points log (100-700 Hz)
-------------------------------------------------------------------------
Test Case                 | Detected | Freq Err (Hz)  | Amp Err (%)  | Residual RMS
-------------------------------------------------------------------------
Clean Single Tone (222Hz) | 1        | 0.0000         | 0.00        % | 0.40
Clean Two Tones           | 2        | 0.0000         | 0.00        % | 0.45
Clean Three Tones         | 3        | 0.0001         | 0.00        % | 0.45
Noisy 20dB SNR 3 Tones    | 3        | 0.0428         | 0.43        % | 471.10
Noisy 10dB SNR 3 Tones    | 3        | 0.1174         | 1.24        % | 1478.36
-------------------------------------------------------------------------
```

---

## License

Standard open-source project.
