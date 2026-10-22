#!/usr/bin/env python3
import sys
import os
import ctypes
import numpy as np
import unittest

# Load shared library
lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bin', 'libint_spectrum.so')

class IntSpectrumTone(ctypes.Structure):
    _fields_ = [
        ("frequencyQ16", ctypes.c_uint32),
        ("peakAmplitude", ctypes.c_uint32),
        ("cosineQ8", ctypes.c_int32),
        ("sineQ8", ctypes.c_int32),
        ("discoveryBank", ctypes.c_uint8),
        ("wienerGainQ15", ctypes.c_uint16)
    ]

class IntSpectrumWienerOptions(ctypes.Structure):
    _fields_ = [
        ("noiseSigmaQ8", ctypes.c_uint32),
        ("noisePowerScaleQ8", ctypes.c_uint16),
        ("minimumGainQ15", ctypes.c_uint16),
        ("suppressBelowGainQ15", ctypes.c_uint16)
    ]

class IntSpectrumWienerReport(ctypes.Structure):
    _fields_ = [
        ("noiseSigmaQ8", ctypes.c_uint32),
        ("noiseCoefficientPowerQ16", ctypes.c_uint64),
        ("tonesAttenuated", ctypes.c_uint16),
        ("tonesSuppressed", ctypes.c_uint16)
    ]

class IntSpectrumLib:
    def __init__(self, path=lib_path):
        self.lib = ctypes.CDLL(path)

        self.lib.int_spectrum_make_linear_grid.argtypes = [ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
        self.lib.int_spectrum_make_linear_grid.restype = ctypes.c_int

        self.lib.int_spectrum_make_log_grid.argtypes = [ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
        self.lib.int_spectrum_make_log_grid.restype = ctypes.c_int

        self.lib.int_spectrum_bin_q16_to_hz.argtypes = [ctypes.c_size_t, ctypes.c_uint32, ctypes.c_double]
        self.lib.int_spectrum_bin_q16_to_hz.restype = ctypes.c_double

        self.lib.int_spectrum_hz_to_bin_q16.argtypes = [ctypes.c_size_t, ctypes.c_double, ctypes.c_double]
        self.lib.int_spectrum_hz_to_bin_q16.restype = ctypes.c_uint32

        self.lib.int_spectrum_decompose_grid.argtypes = [
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
            ctypes.c_uint32, ctypes.c_size_t, ctypes.POINTER(IntSpectrumTone), ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(IntSpectrumWienerOptions), ctypes.POINTER(IntSpectrumWienerReport)
        ]
        self.lib.int_spectrum_decompose_grid.restype = ctypes.c_int

        self.lib.int_spectrum_decompose_comb.argtypes = [
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int16), ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_size_t, ctypes.POINTER(IntSpectrumTone), ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)
        ]
        self.lib.int_spectrum_decompose_comb.restype = ctypes.c_int

        self.lib.int_spectrum_reconstruct_denoised.argtypes = [
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int16)
        ]
        self.lib.int_spectrum_reconstruct_denoised.restype = None

        self.lib.int_spectrum_residual_rms_q8.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64)]
        self.lib.int_spectrum_residual_rms_q8.restype = ctypes.c_uint64

class TestIntSpectrum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = IntSpectrumLib()

    def test_grid_generators(self):
        N = 512
        Fs = 8000.0
        grid = (ctypes.c_uint32 * 129)()
        ok = self.api.lib.int_spectrum_make_log_grid(N, 100.0, 700.0, Fs, grid, 129)
        self.assertEqual(ok, 1)
        self.assertGreater(grid[128], grid[0])

    def test_synthetic_decomposition(self):
        N = 512
        Fs = 8000.0
        grid = (ctypes.c_uint32 * 129)()
        self.api.lib.int_spectrum_make_log_grid(N, 100.0, 700.0, Fs, grid, 129)

        # Generate signal: 2 tones (200Hz amp 10000, 450Hz amp 5000)
        t = np.arange(N) / Fs
        sig = 10000.0 * np.sin(2 * np.pi * 200.0 * t) + 5000.0 * np.sin(2 * np.pi * 450.0 * t) + 1200.0
        samples = np.clip(np.round(sig), -32768, 32767).astype(np.int16)
        expected_dc = int(np.round(np.mean(samples)))

        c_samples = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        c_tones = (IntSpectrumTone * 5)()
        c_tone_count = ctypes.c_size_t(0)
        c_residual = (ctypes.c_int64 * N)()
        c_dc = ctypes.c_int32(0)

        ok = self.api.lib.int_spectrum_decompose_grid(
            N, c_samples, grid, 129, 500, 5, c_tones, ctypes.byref(c_tone_count), c_residual, ctypes.byref(c_dc), None, None
        )

        self.assertEqual(ok, 1)
        self.assertEqual(c_tone_count.value, 2)
        self.assertAlmostEqual(c_dc.value, expected_dc, delta=5)

        freqs = [self.api.lib.int_spectrum_bin_q16_to_hz(N, c_tones[i].frequencyQ16, Fs) for i in range(c_tone_count.value)]
        amps = [c_tones[i].peakAmplitude for i in range(c_tone_count.value)]

        self.assertAlmostEqual(freqs[0], 200.0, delta=2.0)
        self.assertAlmostEqual(freqs[1], 450.0, delta=2.0)

    def test_wiener_denoising(self):
        N = 512
        Fs = 8000.0
        grid = (ctypes.c_uint32 * 129)()
        self.api.lib.int_spectrum_make_log_grid(N, 100.0, 700.0, Fs, grid, 129)

        np.random.seed(42)
        t = np.arange(N) / Fs
        clean = 10000.0 * np.sin(2 * np.pi * 300.0 * t)
        noise = np.random.normal(0, 1000, N)
        samples = np.clip(np.round(clean + noise), -32768, 32767).astype(np.int16)

        c_samples = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        c_tones = (IntSpectrumTone * 5)()
        c_tone_count = ctypes.c_size_t(0)
        c_residual = (ctypes.c_int64 * N)()
        c_dc = ctypes.c_int32(0)

        w_opt = IntSpectrumWienerOptions(noiseSigmaQ8=0, noisePowerScaleQ8=256, minimumGainQ15=0, suppressBelowGainQ15=0)
        w_rep = IntSpectrumWienerReport()

        ok = self.api.lib.int_spectrum_decompose_grid(
            N, c_samples, grid, 129, 300, 5, c_tones, ctypes.byref(c_tone_count), c_residual, ctypes.byref(c_dc), ctypes.byref(w_opt), ctypes.byref(w_rep)
        )
        self.assertEqual(ok, 1)

        denoised = (ctypes.c_int16 * N)()
        self.api.lib.int_spectrum_reconstruct_denoised(N, c_samples, c_residual, denoised)
        denoised_arr = np.array(denoised, dtype=np.int16)

        # SNR improvement
        noisy_err = np.mean((samples - clean)**2)
        denoised_err = np.mean((denoised_arr - clean)**2)
        self.assertLess(denoised_err, noisy_err)

if __name__ == '__main__':
    unittest.main()
