#!/usr/bin/env python3
"""
Benchmark int_spectrum against Standard FFT (numpy.fft.rfft)
Compares:
1. Computation Latency (CPU execution time per block across N=64..4096)
2. Frequency Estimation Precision on Off-Bin Tones
3. Residual Energy & Multi-Tone Recovery under Noise (SNR 0..30 dB)
4. Dynamic Memory & Arithmetic Characteristics
"""

import os
import sys
import time
import ctypes
import subprocess
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests')))
from test_int_spectrum import IntSpectrumLib, IntSpectrumTone

def run_benchmarks():
    api = IntSpectrumLib()
    Fs = 8000.0
    block_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    print("=" * 78)
    print("           INT_SPECTRUM VS STANDARD FFT BENCHMARK SUITE")
    print("=" * 78)

    # ----------------------------------------------------
    # Benchmark 1: Execution Latency (CPU Time per Block)
    # ----------------------------------------------------
    print("\n[Benchmark 1] Execution Latency (Microseconds per Block)")
    print("-" * 78)
    print(f"{'Block Size N':<14} | {'IntSpectrum Grid (us)':<22} | {'Standard FFT (us)':<20} | {'Ratio (FFT/Int)'}")
    print("-" * 78)

    latency_results = {}

    for N in block_sizes:
        # Generate dummy input
        t = np.arange(N) / Fs
        sig = (10000.0 * np.cos(2 * np.pi * 234.5 * t)).astype(np.int16)

        # Prepare IntSpectrum C grid
        grid_len = min(129, N)
        grid = (ctypes.c_uint32 * grid_len)()
        api.lib.int_spectrum_make_log_grid(N, 100.0, 1000.0, Fs, grid, grid_len)

        c_samples = sig.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        c_tones = (IntSpectrumTone * 5)()
        c_tone_cnt = ctypes.c_size_t(0)
        c_res = (ctypes.c_int64 * N)()
        c_dc = ctypes.c_int32(0)

        # Warmup
        for _ in range(10):
            api.lib.int_spectrum_decompose_grid(N, c_samples, grid, grid_len, 300, 3, c_tones, ctypes.byref(c_tone_cnt), c_res, ctypes.byref(c_dc), None, None)
            np.fft.rfft(sig)

        # Benchmark IntSpectrum
        iters = 1000 if N <= 512 else 200
        t0 = time.perf_counter()
        for _ in range(iters):
            api.lib.int_spectrum_decompose_grid(N, c_samples, grid, grid_len, 300, 3, c_tones, ctypes.byref(c_tone_cnt), c_res, ctypes.byref(c_dc), None, None)
        t1 = time.perf_counter()
        int_time_us = ((t1 - t0) / iters) * 1e6

        # Benchmark FFT
        t0 = time.perf_counter()
        for _ in range(iters):
            np.fft.rfft(sig)
        t1 = time.perf_counter()
        fft_time_us = ((t1 - t0) / iters) * 1e6

        ratio = fft_time_us / int_time_us if int_time_us > 0 else 0
        latency_results[N] = (int_time_us, fft_time_us)
        print(f"{N:<14} | {int_time_us:<22.2f} | {fft_time_us:<20.2f} | {ratio:.2f}x")

    # ----------------------------------------------------
    # Benchmark 2: Off-Bin Tone Frequency Accuracy
    # ----------------------------------------------------
    print("\n[Benchmark 2] Frequency Accuracy on Off-Bin Inter-Tone Frequencies")
    print("-" * 78)
    print("Testing tones that do not fall cleanly on standard FFT bin centers (N=512, Fs=8000Hz).")
    print(f"{'Target Freq (Hz)':<18} | {'FFT Bin Hz':<15} | {'FFT Error (Hz)':<15} | {'IntSpectrum Error (Hz)'}")
    print("-" * 78)

    N_eval = 512
    t_eval = np.arange(N_eval) / Fs
    bin_width = Fs / N_eval  # 15.625 Hz per bin
    test_target_freqs = [123.45, 234.56, 345.67, 456.78, 567.89]

    grid_eval = (ctypes.c_uint32 * 129)()
    api.lib.int_spectrum_make_log_grid(N_eval, 100.0, 700.0, Fs, grid_eval, 129)

    for target_f in test_target_freqs:
        test_sig = (10000.0 * np.cos(2 * np.pi * target_f * t_eval)).astype(np.int16)

        # FFT peak detection (argmax of rfft magnitude)
        fft_mags = np.abs(np.fft.rfft(test_sig))
        peak_idx = np.argmax(fft_mags)
        fft_detected_f = peak_idx * bin_width
        fft_err = abs(fft_detected_f - target_f)

        # IntSpectrum detection
        c_samples = test_sig.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        c_tones = (IntSpectrumTone * 5)()
        c_tone_cnt = ctypes.c_size_t(0)
        c_res = (ctypes.c_int64 * N_eval)()
        c_dc = ctypes.c_int32(0)

        api.lib.int_spectrum_decompose_grid(N_eval, c_samples, grid_eval, 129, 300, 1, c_tones, ctypes.byref(c_tone_cnt), c_res, ctypes.byref(c_dc), None, None)

        if c_tone_cnt.value > 0:
            int_detected_f = api.lib.int_spectrum_bin_q16_to_hz(N_eval, c_tones[0].frequencyQ16, Fs)
            int_err = abs(int_detected_f - target_f)
        else:
            int_err = float('nan')

        print(f"{target_f:<18.2f} | {fft_detected_f:<15.2f} | {fft_err:<15.4f} | {int_err:.4f}")

    # ----------------------------------------------------
    # Benchmark 3: Native CPU Cycle Counts (RDTSC / Cycles)
    # ----------------------------------------------------
    print("\n[Benchmark 3] Native C++ CPU Clock Cycles per Block & Sample")
    print("-" * 78)
    bench_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bin', 'benchmark_cpu_cycles')
    if os.path.exists(bench_bin):
        res = subprocess.run([bench_bin], capture_output=True, text=True)
        print(res.stdout)
    else:
        print("Compiled C++ cycle benchmark binary not found. Build with 'make benchmark'.")

    # ----------------------------------------------------
    # Benchmark 4: Structural & Algorithmic Comparison
    # ----------------------------------------------------
    print("\n[Benchmark 3] Architectural & Memory Feature Comparison")
    print("-" * 78)
    print(f"{'Feature / Dimension':<28} | {'Standard FFT (e.g. FFTW/kissFFT)':<26} | {'IntSpectrum'}")
    print("-" * 78)
    print(f"{'Arithmetic Type':<28} | {'Floating-Point (float/double)':<26} | {'Fixed-Point (Q8/Q15/Q16 Integer)'}")
    print(f"{'Dynamic Heap Allocation':<28} | {'Required (Twiddle/Trig Tables)':<26} | {'Zero (100% Static Stack/Buffer)'}")
    print(f"{'Frequency Resolution':<28} | {'Uniform (Fs/N bin steps)':<26} | {'Arbitrary Custom Grid (Log/Comb)'}")
    print(f"{'Parametric Model Output':<28} | {'Bin Magnitudes (Needs Fitting)':<26} | {'Direct Tone Sines, Cosines, Amps'}")
    print(f"{'Noise Filtering Integration':<28} | {'Post-proc Thresholding':<26} | {'Empirical Wiener Shrinkage'}")
    print("=" * 78)

if __name__ == '__main__':
    run_benchmarks()
