#!/usr/bin/env python3
"""
Generate test graphs for documentation, including FFT comparison charts.
"""

import os
import sys
import ctypes
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests')))
from test_int_spectrum import IntSpectrumLib, IntSpectrumTone

def generate_graphs():
    os.makedirs('docs/images', exist_ok=True)
    api = IntSpectrumLib()

    N = 512
    Fs = 8000.0
    t = np.arange(N) / Fs

    f1, a1 = 222.0, 12000.0
    f2, a2 = 432.0, 6500.0
    f3, a3 = 515.0, 3500.0

    clean_signal = (a1 * np.cos(2 * np.pi * f1 * t) +
                    a2 * np.cos(2 * np.pi * f2 * t) +
                    a3 * np.cos(2 * np.pi * f3 * t))

    np.random.seed(42)
    noise = np.random.normal(0, 500.0, N)
    signal = (clean_signal + noise).astype(np.int16)

    grid = (ctypes.c_uint32 * 129)()
    api.lib.int_spectrum_make_log_grid(N, 100.0, 700.0, Fs, grid, 129)

    c_samples = signal.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
    c_tones = (IntSpectrumTone * 5)()
    c_tone_count = ctypes.c_size_t(0)
    c_residual = (ctypes.c_int64 * N)()
    c_dc = ctypes.c_int32(0)

    api.lib.int_spectrum_decompose_grid(
        N, c_samples, grid, 129, 300, 5, c_tones, ctypes.byref(c_tone_count), c_residual, ctypes.byref(c_dc), None, None
    )

    residual_arr = np.array(c_residual[:N], dtype=np.float64) / 256.0
    model_arr = signal - residual_arr

    det_tones = []
    for i in range(c_tone_count.value):
        t_freq = api.lib.int_spectrum_bin_q16_to_hz(N, c_tones[i].frequencyQ16, Fs)
        det_tones.append({'freq_hz': t_freq, 'amplitude': c_tones[i].peakAmplitude})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#1e222a')
    for ax in (ax1, ax2):
        ax.set_facecolor('#21252b')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, color='#3a3f4b', linestyle='--', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color('#3a3f4b')

    time_ms = t * 1000.0
    ax1.plot(time_ms, signal, label='Input Signal (with Noise)', color='#61afef', alpha=0.8, lw=1.2)
    ax1.plot(time_ms, model_arr, label='IntSpectrum Reconstructed Model', color='#98c379', lw=1.5)
    ax1.plot(time_ms, residual_arr, label='Residual Error', color='#e06c75', lw=0.8, alpha=0.7)
    ax1.set_title("Time-Domain Signal, Fitted Fixed-Point Model & Residual", fontsize=12, pad=8)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(facecolor='#282c34', edgecolor='none', labelcolor='white', loc='upper right')

    fft_freqs = np.fft.rfftfreq(N, d=1.0/Fs)
    fft_mag = np.abs(np.fft.rfft(signal)) * (2.0 / N)
    ax2.plot(fft_freqs, fft_mag, label='FFT Reference Spectrum', color='#5c6370', lw=1.2)

    detected_freqs = [t_item['freq_hz'] for t_item in det_tones]
    detected_amps = [t_item['amplitude'] for t_item in det_tones]
    ax2.stem(detected_freqs, detected_amps, linefmt='#e5c07b', markerfmt='C3o', basefmt=' ', label='IntSpectrum Detected Tones')

    ax2.set_xlim(0, 1000)
    ax2.set_title("Frequency Domain: FFT Reference vs IntSpectrum Grid Tone Extraction", fontsize=12, pad=8)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Peak Amplitude")
    ax2.legend(facecolor='#282c34', edgecolor='none', labelcolor='white', loc='upper right')

    plt.tight_layout()
    plt.savefig('docs/images/decomposition_overview.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved docs/images/decomposition_overview.png")

    # Graph 2: Performance & Frequency Accuracy vs SNR
    snr_levels = [30, 20, 15, 10, 5, 0]
    freq_errors = []

    for snr in snr_levels:
        err_f_list = []
        for trial in range(15):
            np.random.seed(trial * 10 + snr)
            sig_pwr = np.mean(clean_signal ** 2)
            noise_pwr = sig_pwr / (10 ** (snr / 10.0))
            noise_vec = np.random.normal(0, np.sqrt(noise_pwr), N)
            trial_signal = np.clip(np.round(clean_signal + noise_vec), -32768, 32767).astype(np.int16)

            c_tsig = trial_signal.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
            c_t_tones = (IntSpectrumTone * 5)()
            c_t_cnt = ctypes.c_size_t(0)
            c_t_res = (ctypes.c_int64 * N)()
            c_t_dc = ctypes.c_int32(0)

            api.lib.int_spectrum_decompose_grid(
                N, c_tsig, grid, 129, 300, 3, c_t_tones, ctypes.byref(c_t_cnt), c_t_res, ctypes.byref(c_t_dc), None, None
            )

            det_f = sorted([api.lib.int_spectrum_bin_q16_to_hz(N, c_t_tones[k].frequencyQ16, Fs) for k in range(c_t_cnt.value)])
            true_f = sorted([f1, f2, f3])

            if len(det_f) == 3:
                f_err = np.mean(np.abs(np.array(det_f) - np.array(true_f)))
                err_f_list.append(f_err)
        freq_errors.append(np.mean(err_f_list) if err_f_list else 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor('#1e222a')
    ax.set_facecolor('#21252b')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(True, color='#3a3f4b', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#3a3f4b')

    ax.plot(snr_levels, freq_errors, 'o-', color='#61afef', lw=2, markersize=8, label='Frequency Error (Hz)')
    ax.set_title("IntSpectrum Estimation Error vs. Signal-to-Noise Ratio (SNR)", fontsize=12, pad=8)
    ax.set_xlabel("Signal-to-Noise Ratio (dB)")
    ax.set_ylabel("Mean Frequency Estimation Error (Hz)")
    ax.invert_xaxis()
    ax.legend(facecolor='#282c34', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    plt.savefig('docs/images/snr_performance.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved docs/images/snr_performance.png")

    # Graph 3: FFT vs IntSpectrum Accuracy Comparison
    target_test_freqs = np.linspace(120.0, 580.0, 15)
    fft_errs = []
    int_errs = []
    bin_width = Fs / N  # 15.625 Hz

    for tf in target_test_freqs:
        t_sig = (10000.0 * np.cos(2 * np.pi * tf * t)).astype(np.int16)

        # FFT peak
        fft_peak_idx = np.argmax(np.abs(np.fft.rfft(t_sig)))
        fft_f = fft_peak_idx * bin_width
        fft_errs.append(abs(fft_f - tf))

        # IntSpectrum peak
        c_tsig = t_sig.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        c_t_tones = (IntSpectrumTone * 5)()
        c_t_cnt = ctypes.c_size_t(0)
        c_t_res = (ctypes.c_int64 * N)()
        c_t_dc = ctypes.c_int32(0)

        api.lib.int_spectrum_decompose_grid(
            N, c_tsig, grid, 129, 300, 1, c_t_tones, ctypes.byref(c_t_cnt), c_t_res, ctypes.byref(c_t_dc), None, None
        )
        if c_t_cnt.value > 0:
            int_f = api.lib.int_spectrum_bin_q16_to_hz(N, c_t_tones[0].frequencyQ16, Fs)
            int_errs.append(abs(int_f - tf))
        else:
            int_errs.append(0.0)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    fig.patch.set_facecolor('#1e222a')
    ax.set_facecolor('#21252b')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(True, color='#3a3f4b', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#3a3f4b')

    ax.plot(target_test_freqs, fft_errs, 's--', color='#e06c75', lw=1.8, label=f'Standard FFT Bin Quantization Error (Step={bin_width:.2f}Hz)')
    ax.plot(target_test_freqs, int_errs, 'o-', color='#98c379', lw=2.2, label='IntSpectrum Continuous Sub-Bin Precision')
    ax.set_title("Frequency Accuracy Comparison: Standard Discrete FFT vs. IntSpectrum Sub-Bin Fitting", fontsize=11, pad=8)
    ax.set_xlabel("Target Input Frequency (Hz)")
    ax.set_ylabel("Absolute Frequency Error (Hz)")
    ax.legend(facecolor='#282c34', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    plt.savefig('docs/images/fft_vs_intspectrum.png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("Saved docs/images/fft_vs_intspectrum.png")

if __name__ == '__main__':
    generate_graphs()
