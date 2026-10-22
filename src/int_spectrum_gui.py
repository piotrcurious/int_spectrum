#!/usr/bin/env python3
"""
int_spectrum_gui.py
Full-featured Linux GUI application for integer spectrum analysis, testing, and real-time visualization.
Supports generated signals, audio file load/playback, and live audio input (mic / synthetic stream).
"""

import sys
import os
import ctypes
import math
import time
import wave
import threading
import queue
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Safely try importing sounddevice for live microphone input
HAS_SOUNDDEVICE = False
try:
    import sounddevice as sd
    # Test if PortAudio library is actually loadable
    sd.query_devices()
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

# Import IntSpectrum C API definitions
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
try:
    from test_int_spectrum import IntSpectrumLib, IntSpectrumTone, IntSpectrumWienerOptions, IntSpectrumWienerReport
except ImportError:
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


class IntSpectrumGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("int_spectrum — Integer DSP Spectral Analyzer & Test Suite")
        self.geometry("1280x850")
        self.minsize(1024, 700)

        # Initialize DSP library
        try:
            self.api = IntSpectrumLib()
        except Exception as e:
            messagebox.showerror("Library Error", f"Failed to load libint_spectrum.so: {e}\nPlease run 'make' first.")
            sys.exit(1)

        # State Variables
        self.input_mode = tk.StringVar(value="generated") # "generated", "file", "live"
        self.sample_rate = tk.DoubleVar(value=8000.0)
        self.block_size = tk.IntVar(value=512)
        self.max_tones = tk.IntVar(value=5)
        self.min_amplitude = tk.IntVar(value=300)
        self.grid_type = tk.StringVar(value="log") # "log", "linear", "comb"
        self.grid_points = tk.IntVar(value=129)
        self.grid_min_hz = tk.DoubleVar(value=100.0)
        self.grid_max_hz = tk.DoubleVar(value=700.0)

        # Wiener filter options
        self.use_wiener = tk.BooleanVar(value=False)
        self.wiener_noise_scale = tk.DoubleVar(value=1.0)

        # Signal Generator state
        self.dc_offset = tk.DoubleVar(value=1800.0)
        self.noise_std = tk.DoubleVar(value=0.0)
        self.tone1_freq = tk.DoubleVar(value=222.65625)
        self.tone1_amp = tk.DoubleVar(value=12000.0)
        self.tone1_enable = tk.BooleanVar(value=True)

        self.tone2_freq = tk.DoubleVar(value=432.03125)
        self.tone2_amp = tk.DoubleVar(value=6500.0)
        self.tone2_enable = tk.BooleanVar(value=True)

        self.tone3_freq = tk.DoubleVar(value=517.1875)
        self.tone3_amp = tk.DoubleVar(value=3500.0)
        self.tone3_enable = tk.BooleanVar(value=True)

        # File input state
        self.audio_file_path = ""
        self.audio_samples = None
        self.file_read_pos = 0

        # Live streaming state
        self.is_streaming = False
        self.stream_thread = None
        self.audio_queue = queue.Queue(maxsize=10)

        self._build_gui()
        self._on_params_changed()

        # Periodically refresh live GUI
        self.after(50, self._gui_update_loop)

    def _build_gui(self):
        # Top control frame and PanedWindow layout
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Control Notebook Panel
        left_frame = ttk.Frame(main_paned, width=380)
        main_paned.add(left_frame, weight=0)

        # Right Visualization Panel
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # --- LEFT PANEL: CONTROL TABS ---
        ctrl_notebook = ttk.Notebook(left_frame)
        ctrl_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Signal Source & Generator
        source_tab = ttk.Frame(ctrl_notebook, padding=10)
        ctrl_notebook.add(source_tab, text="Signal Source")
        self._build_source_tab(source_tab)

        # Tab 2: DSP Algorithm Settings
        dsp_tab = ttk.Frame(ctrl_notebook, padding=10)
        ctrl_notebook.add(dsp_tab, text="DSP Parameters")
        self._build_dsp_tab(dsp_tab)

        # --- RIGHT PANEL: PLOTS & RESULTS ---
        # Matplotlib Figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_time = self.fig.add_subplot(211)
        self.ax_freq = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom Results Table & Status
        results_frame = ttk.LabelFrame(right_frame, text="Decomposition Results & Detection Table", padding=5)
        results_frame.pack(fill=tk.X, padx=5, pady=5)

        cols = ("idx", "bank", "freq_bin", "freq_hz", "peak_amp", "wiener_gain")
        self.results_tree = ttk.Treeview(results_frame, columns=cols, show="headings", height=5)
        self.results_tree.heading("idx", text="#")
        self.results_tree.column("idx", width=30, anchor="center")
        self.results_tree.heading("bank", text="Bank/Grid")
        self.results_tree.column("bank", width=90, anchor="center")
        self.results_tree.heading("freq_bin", text="Bin (Q16)")
        self.results_tree.column("freq_bin", width=90, anchor="center")
        self.results_tree.heading("freq_hz", text="Freq (Hz)")
        self.results_tree.column("freq_hz", width=100, anchor="center")
        self.results_tree.heading("peak_amp", text="Peak Amp")
        self.results_tree.column("peak_amp", width=90, anchor="center")
        self.results_tree.heading("wiener_gain", text="Wiener Gain")
        self.results_tree.column("wiener_gain", width=90, anchor="center")
        self.results_tree.pack(fill=tk.X, side=tk.LEFT, expand=True)

        # Status & Metrics bar
        self.status_label = ttk.Label(results_frame, text="Ready.", font=("Monospace", 9))
        self.status_label.pack(fill=tk.Y, side=tk.RIGHT, padx=10)

    def _build_source_tab(self, parent):
        # Input Mode Selector
        mode_frame = ttk.LabelFrame(parent, text="Input Source Mode", padding=5)
        mode_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(mode_frame, text="Synthetic Generator", variable=self.input_mode, value="generated", command=self._on_params_changed).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="WAV Audio File", variable=self.input_mode, value="file", command=self._on_params_changed).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Live / Continuous Audio", variable=self.input_mode, value="live", command=self._on_params_changed).pack(anchor=tk.W)

        # File Controls Frame
        file_frame = ttk.LabelFrame(parent, text="Audio File Controls", padding=5)
        file_frame.pack(fill=tk.X, pady=5)

        self.file_label = ttk.Label(file_frame, text="No file loaded", wraplength=280)
        self.file_label.pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Browse WAV File...", command=self._browse_wav_file).pack(fill=tk.X, pady=2)

        # Synthetic Generator Controls Frame
        gen_frame = ttk.LabelFrame(parent, text="Synthetic Signal Synthesizer", padding=5)
        gen_frame.pack(fill=tk.X, pady=5)

        ttk.Label(gen_frame, text="DC Offset:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(gen_frame, textvariable=self.dc_offset, width=10).grid(row=0, column=1)

        ttk.Label(gen_frame, text="Noise StdDev:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(gen_frame, textvariable=self.noise_std, width=10).grid(row=1, column=1)

        # Tone 1
        ttk.Checkbutton(gen_frame, text="Tone 1 Enable", variable=self.tone1_enable, command=self._on_params_changed).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5,0))
        ttk.Label(gen_frame, text="  Freq (Hz):").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(gen_frame, textvariable=self.tone1_freq, width=10).grid(row=3, column=1)
        ttk.Label(gen_frame, text="  Amp:").grid(row=3, column=2, sticky=tk.W)
        ttk.Entry(gen_frame, textvariable=self.tone1_amp, width=10).grid(row=3, column=3)

        # Tone 2
        ttk.Checkbutton(gen_frame, text="Tone 2 Enable", variable=self.tone2_enable, command=self._on_params_changed).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5,0))
        ttk.Label(gen_frame, text="  Freq (Hz):").grid(row=5, column=0, sticky=tk.W)
        ttk.Entry(gen_frame, textvariable=self.tone2_freq, width=10).grid(row=5, column=1)
        ttk.Label(gen_frame, text="  Amp:").grid(row=5, column=2, sticky=tk.W)
        ttk.Entry(gen_frame, textvariable=self.tone2_amp, width=10).grid(row=5, column=3)

        # Tone 3
        ttk.Checkbutton(gen_frame, text="Tone 3 Enable", variable=self.tone3_enable, command=self._on_params_changed).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(5,0))
        ttk.Label(gen_frame, text="  Freq (Hz):").grid(row=7, column=0, sticky=tk.W)
        ttk.Entry(gen_frame, textvariable=self.tone3_freq, width=10).grid(row=7, column=1)
        ttk.Label(gen_frame, text="  Amp:").grid(row=7, column=2, sticky=tk.W)
        ttk.Entry(gen_frame, textvariable=self.tone3_amp, width=10).grid(row=7, column=3)

        # Live Controls
        live_frame = ttk.LabelFrame(parent, text="Live Input / Streaming", padding=5)
        live_frame.pack(fill=tk.X, pady=5)

        self.stream_btn = ttk.Button(live_frame, text="Start Streaming", command=self._toggle_live_streaming)
        self.stream_btn.pack(fill=tk.X)

        ttk.Button(parent, text="Update / Run Analysis", command=self._on_params_changed).pack(fill=tk.X, pady=10)

    def _build_dsp_tab(self, parent):
        grid_frame = ttk.LabelFrame(parent, text="Frequency Search Grid & Mode", padding=5)
        grid_frame.pack(fill=tk.X, pady=5)

        ttk.Label(grid_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W)
        mode_cb = ttk.Combobox(grid_frame, textvariable=self.grid_type, values=["log", "linear", "comb"], state="readonly", width=12)
        mode_cb.grid(row=0, column=1, sticky=tk.W)
        mode_cb.bind("<<ComboboxSelected>>", lambda e: self._on_params_changed())

        ttk.Label(grid_frame, text="Block Size N:").grid(row=1, column=0, sticky=tk.W)
        n_cb = ttk.Combobox(grid_frame, textvariable=self.block_size, values=[64, 128, 256, 512, 1024, 2048, 4096], state="readonly", width=12)
        n_cb.grid(row=1, column=1, sticky=tk.W)
        n_cb.bind("<<ComboboxSelected>>", lambda e: self._on_params_changed())

        ttk.Label(grid_frame, text="Sample Rate Fs (Hz):").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(grid_frame, textvariable=self.sample_rate, width=12).grid(row=2, column=1)

        ttk.Label(grid_frame, text="Grid Points:").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(grid_frame, textvariable=self.grid_points, width=12).grid(row=3, column=1)

        ttk.Label(grid_frame, text="Min Freq (Hz):").grid(row=4, column=0, sticky=tk.W)
        ttk.Entry(grid_frame, textvariable=self.grid_min_hz, width=12).grid(row=4, column=1)

        ttk.Label(grid_frame, text="Max Freq (Hz):").grid(row=5, column=0, sticky=tk.W)
        ttk.Entry(grid_frame, textvariable=self.grid_max_hz, width=12).grid(row=5, column=1)

        algo_frame = ttk.LabelFrame(parent, text="Decomposition & Thresholds", padding=5)
        algo_frame.pack(fill=tk.X, pady=5)

        ttk.Label(algo_frame, text="Max Tones:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(algo_frame, textvariable=self.max_tones, width=12).grid(row=0, column=1)

        ttk.Label(algo_frame, text="Min Amplitude:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(algo_frame, textvariable=self.min_amplitude, width=12).grid(row=1, column=1)

        wiener_frame = ttk.LabelFrame(parent, text="Wiener Shrinkage Filter", padding=5)
        wiener_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(wiener_frame, text="Enable Wiener Shrinkage", variable=self.use_wiener, command=self._on_params_changed).pack(anchor=tk.W)
        ttk.Label(wiener_frame, text="Noise Power Scale:").pack(anchor=tk.W)
        ttk.Entry(wiener_frame, textvariable=self.wiener_noise_scale, width=12).pack(anchor=tk.W)

    def _browse_wav_file(self):
        filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if filename:
            try:
                with wave.open(filename, 'rb') as wf:
                    nchannels, sampwidth, framerate, nframes, comptype, compname = wf.getparams()
                    raw = wf.readframes(nframes)
                    if sampwidth == 2:
                        samples = np.frombuffer(raw, dtype=np.int16)
                    elif sampwidth == 1:
                        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128) * 256
                        samples = samples.astype(np.int16)
                    else:
                        messagebox.showerror("Format Error", "Only 8-bit or 16-bit PCM WAV files supported.")
                        return

                    if nchannels > 1:
                        samples = samples[::nchannels] # take first channel

                    self.audio_file_path = filename
                    self.audio_samples = samples
                    self.sample_rate.set(float(framerate))
                    self.file_read_pos = 0
                    self.file_label.config(text=f"{os.path.basename(filename)} ({len(samples)} samples, {framerate} Hz)")
                    self.input_mode.set("file")
                    self._on_params_changed()
            except Exception as e:
                messagebox.showerror("File Error", f"Error opening WAV file: {e}")

    def _toggle_live_streaming(self):
        if self.is_streaming:
            self.is_streaming = False
            self.stream_btn.config(text="Start Streaming")
        else:
            self.is_streaming = True
            self.input_mode.set("live")
            self.stream_btn.config(text="Stop Streaming")
            self.stream_thread = threading.Thread(target=self._live_audio_worker, daemon=True)
            self.stream_thread.start()

    def _live_audio_worker(self):
        """Worker thread to capture live microphone or simulated live stream."""
        if HAS_SOUNDDEVICE:
            try:
                def audio_callback(indata, frames, time_info, status):
                    if status:
                        pass
                    int_samples = (indata[:, 0] * 32767.0).astype(np.int16)
                    if not self.audio_queue.full():
                        self.audio_queue.put(int_samples)

                fs = int(self.sample_rate.get())
                N = self.block_size.get()
                with sd.InputStream(samplerate=fs, channels=1, blocksize=N, callback=audio_callback):
                    while self.is_streaming:
                        time.sleep(0.01)
                return
            except Exception as e:
                print(f"Hardware audio device error, falling back to simulated live stream: {e}")

        # Fallback simulated live stream (sweeping frequencies)
        t_accum = 0.0
        while self.is_streaming:
            N = self.block_size.get()
            Fs = self.sample_rate.get()
            f1 = 200.0 + 100.0 * math.sin(2.0 * math.pi * 0.2 * t_accum)
            f2 = 450.0 + 50.0 * math.cos(2.0 * math.pi * 0.1 * t_accum)
            t = np.arange(N) / Fs + t_accum
            t_accum += N / Fs

            sig = 10000.0 * np.sin(2.0 * np.pi * f1 * t) + 6000.0 * np.sin(2.0 * np.pi * f2 * t)
            sig += np.random.normal(0, 300, N)
            int_samples = np.clip(np.round(sig), -32768, 32767).astype(np.int16)

            if not self.audio_queue.full():
                self.audio_queue.put(int_samples)
            time.sleep(N / Fs)

    def _gui_update_loop(self):
        if self.is_streaming and not self.audio_queue.empty():
            try:
                samples = self.audio_queue.get_nowait()
                self._run_analysis_on_samples(samples)
            except queue.Empty:
                pass
        self.after(50, self._gui_update_loop)

    def _on_params_changed(self):
        if self.is_streaming:
            return

        mode = self.input_mode.get()
        N = self.block_size.get()
        Fs = self.sample_rate.get()

        if mode == "generated":
            t = np.arange(N) / Fs
            sig = np.full(N, self.dc_offset.get())

            if self.tone1_enable.get():
                sig += self.tone1_amp.get() * np.sin(2.0 * np.pi * self.tone1_freq.get() * t)
            if self.tone2_enable.get():
                sig += self.tone2_amp.get() * np.sin(2.0 * np.pi * self.tone2_freq.get() * t)
            if self.tone3_enable.get():
                sig += self.tone3_amp.get() * np.sin(2.0 * np.pi * self.tone3_freq.get() * t)

            if self.noise_std.get() > 0:
                sig += np.random.normal(0, self.noise_std.get(), N)

            samples = np.clip(np.round(sig), -32768, 32767).astype(np.int16)
            self._run_analysis_on_samples(samples)

        elif mode == "file":
            if self.audio_samples is not None and len(self.audio_samples) >= N:
                if self.file_read_pos + N > len(self.audio_samples):
                    self.file_read_pos = 0
                samples = self.audio_samples[self.file_read_pos : self.file_read_pos + N]
                self.file_read_pos += N // 2 # advance 50% overlap
                self._run_analysis_on_samples(samples)

    def _run_analysis_on_samples(self, samples):
        N = len(samples)
        Fs = self.sample_rate.get()
        grid_mode = self.grid_type.get()
        pts = self.grid_points.get()
        max_t = self.max_tones.get()
        min_amp = self.min_amplitude.get()

        c_samples = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        c_tones = (IntSpectrumTone * max_t)()
        c_tone_count = ctypes.c_size_t(0)
        c_residual = (ctypes.c_int64 * N)()
        c_dc = ctypes.c_int32(0)

        t0 = time.perf_counter()
        ok = False

        wiener_opt = None
        wiener_rep = None
        if self.use_wiener.get():
            w_scale = int(round(self.wiener_noise_scale.get() * 256.0))
            wiener_opt = IntSpectrumWienerOptions(noiseSigmaQ8=0, noisePowerScaleQ8=w_scale, minimumGainQ15=0, suppressBelowGainQ15=0)
            wiener_rep = IntSpectrumWienerReport()

        if grid_mode in ["log", "linear"]:
            grid = (ctypes.c_uint32 * pts)()
            if grid_mode == "log":
                self.api.lib.int_spectrum_make_log_grid(N, self.grid_min_hz.get(), self.grid_max_hz.get(), Fs, grid, pts)
            else:
                self.api.lib.int_spectrum_make_linear_grid(N, self.grid_min_hz.get(), self.grid_max_hz.get(), Fs, grid, pts)

            ok = self.api.lib.int_spectrum_decompose_grid(
                N, c_samples, grid, pts, min_amp, max_t, c_tones, ctypes.byref(c_tone_count), c_residual, ctypes.byref(c_dc),
                ctypes.byref(wiener_opt) if wiener_opt else None,
                ctypes.byref(wiener_rep) if wiener_rep else None
            )
        else: # Comb mode
            first_bin = max(1, int(round(self.grid_min_hz.get() * N / Fs)))
            last_bin = min(N // 2 - 1, int(round(self.grid_max_hz.get() * N / Fs)))
            ok = self.api.lib.int_spectrum_decompose_comb(
                N, c_samples, first_bin, last_bin, min_amp, max_t, c_tones, ctypes.byref(c_tone_count), c_residual, ctypes.byref(c_dc)
            )

        t_elapsed = (time.perf_counter() - t0) * 1000.0 # ms

        if not ok:
            self.status_label.config(text="Analysis failed: parameters out of range.")
            return

        tone_count = c_tone_count.value
        residual_arr = np.ctypeslib.as_array(c_residual)
        res_rms = self.api.lib.int_spectrum_residual_rms_q8(N, c_residual) / 256.0

        # Reconstruct fitted tone signal
        denoised = (ctypes.c_int16 * N)()
        self.api.lib.int_spectrum_reconstruct_denoised(N, c_samples, c_residual, denoised)
        denoised_samples = np.ctypeslib.as_array(denoised)

        # Update Treeview
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        detected_tones_list = []
        for i in range(tone_count):
            t = c_tones[i]
            hz = self.api.lib.int_spectrum_bin_q16_to_hz(N, t.frequencyQ16, Fs)
            bank_str = "Custom Grid" if t.discoveryBank == 2 else ("Integer" if t.discoveryBank == 0 else "Half-Bin")
            w_gain = f"{t.wienerGainQ15 / 32768.0:.2f}"
            self.results_tree.insert("", "end", values=(i+1, bank_str, t.frequencyQ16, f"{hz:.2f}", t.peakAmplitude, w_gain))
            detected_tones_list.append((hz, t.peakAmplitude))

        self.status_label.config(text=f"Time: {t_elapsed:.2f}ms | DC: {c_dc.value} | Res RMS: {res_rms:.2f} samples")

        # Update Plots
        self._update_plots(samples, denoised_samples, residual_arr / 256.0, detected_tones_list, Fs, N)

    def _update_plots(self, input_samples, model_samples, residual_samples, detected_tones, Fs, N):
        time_axis = np.arange(N) / Fs * 1000.0 # ms

        # Time Plot
        self.ax_time.clear()
        self.ax_time.plot(time_axis, input_samples, label="Input Signal", color="#1f77b4", alpha=0.7)
        self.ax_time.plot(time_axis, model_samples, label="Decomposed Model", color="#2ca02c", linewidth=1.5)
        self.ax_time.plot(time_axis, residual_samples, label="Residual Error", color="#d62728", linestyle="--", alpha=0.6)
        self.ax_time.set_title("Time Domain Signal, Model & Residual")
        self.ax_time.set_xlabel("Time (ms)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.legend(loc="upper right", fontsize=8)
        self.ax_time.grid(True, linestyle=":", alpha=0.6)

        # Spectrum / Frequency Plot (FFT reference vs Detected Tones)
        self.ax_freq.clear()
        fft_mags = np.abs(np.fft.rfft(input_samples)) * (2.0 / N)
        fft_freqs = np.fft.rfftfreq(N, d=1.0/Fs)

        self.ax_freq.plot(fft_freqs, fft_mags, label="Reference FFT Spectrum", color="#7f7f7f", alpha=0.5)

        # Plot detected tones as discrete spectral lines
        if detected_tones:
            det_freqs, det_amps = zip(*detected_tones)
            self.ax_freq.vlines(det_freqs, 0, det_amps, color="#d62728", linewidth=2, label="IntSpectrum Tones")
            self.ax_freq.scatter(det_freqs, det_amps, color="#d62728", zorder=5)

        self.ax_freq.set_title("Frequency Domain (FFT Reference vs. Integer Grid Tones)")
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_ylabel("Peak Amplitude")
        self.ax_freq.set_xlim(0, Fs / 2.0)
        self.ax_freq.legend(loc="upper right", fontsize=8)
        self.ax_freq.grid(True, linestyle=":", alpha=0.6)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


if __name__ == "__main__":
    app = IntSpectrumGUI()
    app.mainloop()
