#!/usr/bin/env python3
"""
Evaluation and Benchmark Suite for int_spectrum DSP algorithms.
Evaluates frequency accuracy, amplitude accuracy, residual RMS, and SNR gain.
"""

import sys
import os
import ctypes
import numpy as np

# Add tests directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_int_spectrum import IntSpectrumLib, IntSpectrumTone, IntSpectrumWienerOptions, IntSpectrumWienerReport

def run_benchmarks():
    api = IntSpectrumLib()
    N = 512
    Fs = 8000.0
    GRID_POINTS = 129

    grid = (ctypes.c_uint32 * GRID_POINTS)()
    api.lib.int_spectrum_make_log_grid(N, 100.0, 700.0, Fs, grid, GRID_POINTS)

    print("=========================================================================")
    print("           INT_SPECTRUM DSP ALGORITHM EVALUATION SUITE            ")
    print("=========================================================================")
    print(f"Configuration: N={N}, Fs={Fs} Hz, Grid={GRID_POINTS} points log (100-700 Hz)")
    print("-------------------------------------------------------------------------")
    print(f"{'Test Case':<25} | {'Detected':<8} | {'Freq Err (Hz)':<14} | {'Amp Err (%)':<12} | {'Residual RMS':<12}")
    print("-------------------------------------------------------------------------")

    test_cases = [
        ("Clean Single Tone (222Hz)", [(222.65625, 12000.0)], 0.0),
        ("Clean Two Tones", [(222.65625, 12000.0), (432.03125, 6500.0)], 0.0),
        ("Clean Three Tones", [(222.65625, 12000.0), (432.03125, 6500.0), (517.1875, 3500.0)], 0.0),
        ("Noisy 20dB SNR 3 Tones", [(222.65625, 12000.0), (432.03125, 6500.0), (517.1875, 3500.0)], 500.0),
        ("Noisy 10dB SNR 3 Tones", [(222.65625, 12000.0), (432.03125, 6500.0), (517.1875, 3500.0)], 1500.0),
    ]

    np.random.seed(1234)

    for name, true_tones, noise_sigma in test_cases:
        t = np.arange(N) / Fs
        sig = np.zeros(N)
        for hz, amp in true_tones:
            sig += amp * np.sin(2 * np.pi * hz * t)

        if noise_sigma > 0:
            sig += np.random.normal(0, noise_sigma, N)

        samples = np.clip(np.round(sig), -32768, 32767).astype(np.int16)

        c_samples = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        c_tones = (IntSpectrumTone * 8)()
        c_tone_count = ctypes.c_size_t(0)
        c_residual = (ctypes.c_int64 * N)()
        c_dc = ctypes.c_int32(0)

        ok = api.lib.int_spectrum_decompose_grid(
            N, c_samples, grid, GRID_POINTS, 500, 8, c_tones, ctypes.byref(c_tone_count), c_residual, ctypes.byref(c_dc), None, None
        )

        detected_count = c_tone_count.value
        freq_errors = []
        amp_errors = []

        # Match detected tones to ground truth
        for true_hz, true_amp in true_tones:
            best_match = None
            min_df = 1e9
            for i in range(detected_count):
                det_hz = api.lib.int_spectrum_bin_q16_to_hz(N, c_tones[i].frequencyQ16, Fs)
                df = abs(det_hz - true_hz)
                if df < min_df:
                    min_df = df
                    best_match = c_tones[i]

            if best_match and min_df < 10.0:
                freq_errors.append(min_df)
                amp_err = abs(float(best_match.peakAmplitude) - true_amp) / true_amp * 100.0
                amp_errors.append(amp_err)

        mean_f_err = np.mean(freq_errors) if freq_errors else float('nan')
        mean_a_err = np.mean(amp_errors) if amp_errors else float('nan')

        res_rms = float(api.lib.int_spectrum_residual_rms_q8(N, c_residual)) / 256.0

        print(f"{name:<25} | {detected_count:<8} | {mean_f_err:<14.4f} | {mean_a_err:<12.2f}% | {res_rms:<12.2f}")

    print("-------------------------------------------------------------------------")

if __name__ == "__main__":
    run_benchmarks()
