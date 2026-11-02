#!/usr/bin/env python3
"""
int_spectrum_gui.py
Full-featured, modern Linux GUI application for integer spectrum analysis, evaluation, and testing.
Supports synthetic signal synthesis (Sine, Square, Sawtooth, Chirp), WAV file loading/playback,
live microphone / simulated streaming, multi-tab Matplotlib visualizations (Time/Spectrum, Spectrogram/Waterfall,
Harmonic Analysis & Consensus Diagnostics), interactive zoom/pan toolbars, DSP presets, WAV recording,
audio playback, and full DSP parameter controls.
"""

import sys
import os
import ctypes
import math
import time
import wave
import csv
import threading
import queue
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Safely try importing sounddevice for live microphone input and audio playback
HAS_SOUNDDEVICE = False
try:
    import sounddevice as sd
    sd.query_devices()
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

# Import C API structures and loader
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

class IntSpectrumFusionResult(ctypes.Structure):
    _fields_ = [
        ("coarseBin", ctypes.c_uint32),
        ("offsetQ16", ctypes.c_int32),
        ("phaseOffsetQ16", ctypes.c_int32),
        ("frequencyQ16", ctypes.c_uint32),
        ("peakAmplitude", ctypes.c_uint32),
        ("strength", ctypes.c_uint64),
        ("phaseDifferenceQ30", ctypes.c_int64),
        ("estimatorValidMask", ctypes.c_uint32),
        ("estimatorInlierMask", ctypes.c_uint32),
        ("consensusSpreadQ16", ctypes.c_uint32),
        ("consensusConfidenceQ15", ctypes.c_uint16),
        ("selectedEstimator", ctypes.c_uint8)
    ]


class IntSpectrumGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("int_spectrum — Modern Linux Integer DSP Evaluation & Test Suite")
        self.geometry("1440x940")
        self.minsize(1120, 740)

        # Initialize C DSP Library
        try:
            self.api = IntSpectrumLib()
            self.api.lib.int_spectrum_fuse_estimates.argtypes = [
                ctypes.c_size_t, ctypes.POINTER(ctypes.c_int16), ctypes.c_uint32, ctypes.POINTER(IntSpectrumFusionResult)
            ]
            self.api.lib.int_spectrum_fuse_estimates.restype = ctypes.c_int
        except Exception as e:
            messagebox.showerror("Library Error", f"Failed to load libint_spectrum.so: {e}\nPlease run 'make' first.")
            sys.exit(1)

        # Apply dark theme styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background='#1e222a', foreground='#abb2bf', fieldbackground='#21252b')
        self.style.configure('TNotebook', background='#1e222a', borderwidth=0)
        self.style.configure('TNotebook.Tab', background='#282c34', foreground='#abb2bf', padding=[12, 5], font=('Helvetica', 9, 'bold'))
        self.style.map('TNotebook.Tab', background=[('selected', '#3b4048')], foreground=[('selected', '#61afef')])
        self.style.configure('TLabelframe', background='#1e222a', foreground='#61afef', bordercolor='#3b4048')
        self.style.configure('TLabelframe.Label', background='#1e222a', foreground='#61afef', font=('Helvetica', 10, 'bold'))

        # DSP & Audio State
        self.input_mode = tk.StringVar(value="generated") # "generated", "file", "live"
        self.waveform_type = tk.StringVar(value="Sine Multi-Tone")
        self.preset_scenario = tk.StringVar(value="Default Multi-Tone")
        self.sample_rate = tk.DoubleVar(value=8000.0)
        self.block_size = tk.IntVar(value=512)
        self.max_tones = tk.IntVar(value=5)
        self.min_amplitude = tk.IntVar(value=300)
        self.grid_type = tk.StringVar(value="log") # "log", "linear", "comb"
        self.grid_points = tk.IntVar(value=129)
        self.grid_min_hz = tk.DoubleVar(value=100.0)
        self.grid_max_hz = tk.DoubleVar(value=700.0)

        # Visualizations & Display Options
        self.freq_scale_mode = tk.StringVar(value="linear") # "linear", "log"
        self.waterfall_cmap = tk.StringVar(value="magma") # "magma", "viridis", "plasma", "coolwarm"
        self.show_peak_markers = tk.BooleanVar(value=True)
        self.show_peak_labels = tk.BooleanVar(value=True)

        # Wiener Filter Options
        self.use_wiener = tk.BooleanVar(value=False)
        self.wiener_noise_scale = tk.DoubleVar(value=1.0)

        # Synthesizer State (5 Channels)
        self.dc_offset = tk.DoubleVar(value=1800.0)
        self.noise_std = tk.DoubleVar(value=0.0)
        self.tones_config = [
            {"freq": tk.DoubleVar(value=222.65625), "amp": tk.DoubleVar(value=12000.0), "enable": tk.BooleanVar(value=True)},
            {"freq": tk.DoubleVar(value=432.03125), "amp": tk.DoubleVar(value=6500.0), "enable": tk.BooleanVar(value=True)},
            {"freq": tk.DoubleVar(value=517.1875),  "amp": tk.DoubleVar(value=3500.0), "enable": tk.BooleanVar(value=True)},
            {"freq": tk.DoubleVar(value=1200.0),    "amp": tk.DoubleVar(value=2000.0), "enable": tk.BooleanVar(value=False)},
            {"freq": tk.DoubleVar(value=2500.0),    "amp": tk.DoubleVar(value=1500.0), "enable": tk.BooleanVar(value=False)},
        ]

        # File & Playback State
        self.audio_file_path = ""
        self.audio_samples = None
        self.file_read_pos = 0
        self.is_file_playing = False

        # Live Input & Audio Playback State
        self.is_streaming = False
        self.stream_thread = None
        self.audio_queue = queue.Queue(maxsize=25)
        self.audio_playback_active = False

        # Waterfall / Spectrogram History Buffer
        self.waterfall_history = []
        self.max_waterfall_history = 35

        # Last Analysis Results Cache
        self.last_input_samples = None
        self.last_model_samples = None
        self.last_residual_samples = None
        self.last_detected_tones = []
        self.last_fusion_result = None

        # Persistent plot artists and state to prevent axis rescaling jitter
        self.spectrum_annotations = []
        self.last_plotted_fs = None
        self.last_scale_mode = None

        # Full-file WAV Spectrogram & Playback State
        self.wav_spec_cache = None     # { 'S_db': 2D ndarray, 'times': 1D ndarray, 'freqs': 1D ndarray, 'detected_tones': list of (t, hz, amp), 'Fs': float, 'duration': float }
        self.wav_playhead_sec = 0.0
        self.wav_is_playing = False
        self.wav_scale_mode = tk.StringVar(value="linear")
        self.wav_audio_stream = None
        self.wav_playback_thread = None
        self.wav_playback_proc = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_gui()
        self._init_persistent_plots()
        self._on_params_changed()

        # Periodically refresh GUI
        self.after(40, self._gui_update_loop)

    def _on_close(self):
        self.is_streaming = False
        self.is_file_playing = False
        self.audio_playback_active = False
        self._stop_wav_playback()
        self.destroy()

    def _build_gui(self):
        # Top Header Bar
        hdr_frame = tk.Frame(self, bg='#181a1f', height=42)
        hdr_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(hdr_frame, text="INT_SPECTRUM — Fixed-Point DSP Evaluation & Visualization",
                             bg='#181a1f', fg='#98c379', font=('Helvetica', 12, 'bold'))
        title_lbl.pack(side=tk.LEFT, padx=15, pady=6)

        # Header Action Buttons
        btn_rec_wav = tk.Button(hdr_frame, text="Record/Save WAV", bg='#e06c75', fg='#ffffff',
                                activebackground='#be5046', bd=0, padx=10, pady=2, command=self._save_audio_wav)
        btn_rec_wav.pack(side=tk.RIGHT, padx=5, pady=6)

        btn_play_audio = tk.Button(hdr_frame, text="Play Audio Audio", bg='#98c379', fg='#1e222a',
                                   activebackground='#7da85e', bd=0, padx=10, pady=2, command=self._play_current_audio)
        btn_play_audio.pack(side=tk.RIGHT, padx=5, pady=6)

        btn_exp_csv = tk.Button(hdr_frame, text="Export CSV", bg='#3b4048', fg='#ffffff',
                                activebackground='#4b5263', bd=0, padx=10, pady=2, command=self._export_csv)
        btn_exp_csv.pack(side=tk.RIGHT, padx=5, pady=6)

        btn_exp_png = tk.Button(hdr_frame, text="Save Plot PNG", bg='#3b4048', fg='#ffffff',
                                activebackground='#4b5263', bd=0, padx=10, pady=2, command=self._export_png)
        btn_exp_png.pack(side=tk.RIGHT, padx=5, pady=6)

        # Paned Window (Left Controls / Right Plots)
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left_frame = ttk.Frame(main_paned, width=450)
        main_paned.add(left_frame, weight=0)

        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # --- LEFT PANEL: CONTROLS ---
        ctrl_notebook = ttk.Notebook(left_frame)
        ctrl_notebook.pack(fill=tk.BOTH, expand=True)

        source_tab = ttk.Frame(ctrl_notebook, padding=8)
        ctrl_notebook.add(source_tab, text="Synthesizer & Input")
        self._build_source_tab(source_tab)

        dsp_tab = ttk.Frame(ctrl_notebook, padding=8)
        ctrl_notebook.add(dsp_tab, text="DSP Params & Grid")
        self._build_dsp_tab(dsp_tab)

        viz_tab = ttk.Frame(ctrl_notebook, padding=8)
        ctrl_notebook.add(viz_tab, text="Display & Presets")
        self._build_viz_tab(viz_tab)

        # --- RIGHT PANEL: VISUALIZATION TABS ---
        self.plot_notebook = ttk.Notebook(right_frame)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Time Domain & Spectrum
        tab_overview = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(tab_overview, text="Time & Spectrum Overview")

        self.fig_overview = Figure(figsize=(8, 5.5), dpi=100, facecolor='#1e222a')
        self.ax_time = self.fig_overview.add_subplot(211, facecolor='#21252b')
        self.ax_freq = self.fig_overview.add_subplot(212, facecolor='#21252b')
        self.fig_overview.tight_layout(pad=2.5)

        self.canvas_overview = FigureCanvasTkAgg(self.fig_overview, master=tab_overview)
        self.canvas_overview.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame_overview = tk.Frame(tab_overview, bg='#1e222a')
        toolbar_frame_overview.pack(fill=tk.X, side=tk.BOTTOM)

        btn_reset_zoom = ttk.Button(toolbar_frame_overview, text="Reset Spectrum Zoom", command=self._reset_freq_zoom)
        btn_reset_zoom.pack(side=tk.RIGHT, padx=6, pady=2)

        self.toolbar_overview = NavigationToolbar2Tk(self.canvas_overview, toolbar_frame_overview)
        self.toolbar_overview.config(background='#1e222a')
        self.toolbar_overview.update()

        # Tab 2: Waterfall / Spectrogram & Residual Histogram
        tab_waterfall = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(tab_waterfall, text="Waterfall & Residual Analysis")

        self.fig_waterfall = Figure(figsize=(8, 5.5), dpi=100, facecolor='#1e222a')
        self.ax_waterfall = self.fig_waterfall.add_subplot(211, facecolor='#21252b')
        self.ax_hist = self.fig_waterfall.add_subplot(212, facecolor='#21252b')
        self.fig_waterfall.tight_layout(pad=2.5)

        self.canvas_waterfall = FigureCanvasTkAgg(self.fig_waterfall, master=tab_waterfall)
        self.canvas_waterfall.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar_waterfall = NavigationToolbar2Tk(self.canvas_waterfall, tab_waterfall)
        self.toolbar_waterfall.config(background='#1e222a')
        self.toolbar_waterfall.update()

        # Tab 3: WAV Spectrogram & Interactive Playback
        tab_wav = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(tab_wav, text="WAV Spectrogram & Playback")

        wav_ctrl_bar = tk.Frame(tab_wav, bg='#1e222a')
        wav_ctrl_bar.pack(fill=tk.X, side=tk.TOP, padx=5, pady=3)

        self.btn_wav_play = ttk.Button(wav_ctrl_bar, text="▶ Play", command=self._toggle_wav_playback)
        self.btn_wav_play.pack(side=tk.LEFT, padx=3)

        self.btn_wav_stop = ttk.Button(wav_ctrl_bar, text="⏹ Stop", command=self._stop_wav_playback)
        self.btn_wav_stop.pack(side=tk.LEFT, padx=3)

        self.btn_wav_rewind = ttk.Button(wav_ctrl_bar, text="⏮ Rewind", command=self._rewind_wav_playback)
        self.btn_wav_rewind.pack(side=tk.LEFT, padx=3)

        ttk.Label(wav_ctrl_bar, text="Position:", background='#1e222a', foreground='#abb2bf').pack(side=tk.LEFT, padx=(10, 2))
        self.wav_time_slider = ttk.Scale(wav_ctrl_bar, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=self._on_wav_scrub)
        self.wav_time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.wav_time_label = ttk.Label(wav_ctrl_bar, text="0.00 / 0.00 s", background='#1e222a', foreground='#61afef', font=('Monospace', 9, 'bold'))
        self.wav_time_label.pack(side=tk.LEFT, padx=6)

        ttk.Label(wav_ctrl_bar, text="Scale:", background='#1e222a', foreground='#abb2bf').pack(side=tk.LEFT, padx=(6, 2))
        self.wav_scale_cb = ttk.Combobox(wav_ctrl_bar, textvariable=self.wav_scale_mode, values=["linear", "log"], state="readonly", width=7)
        self.wav_scale_cb.pack(side=tk.LEFT, padx=2)
        self.wav_scale_cb.bind("<<ComboboxSelected>>", self._on_wav_scale_mode_changed)

        ttk.Button(wav_ctrl_bar, text="🔍+", width=4, command=lambda: self._zoom_wav_view(0.5)).pack(side=tk.LEFT, padx=1)
        ttk.Button(wav_ctrl_bar, text="🔍-", width=4, command=lambda: self._zoom_wav_view(2.0)).pack(side=tk.LEFT, padx=1)
        ttk.Button(wav_ctrl_bar, text="Reset View", command=self._reset_wav_zoom).pack(side=tk.LEFT, padx=3)
        ttk.Button(wav_ctrl_bar, text="🔄 Recache", command=self._precompute_wav_cache).pack(side=tk.LEFT, padx=3)

        self.fig_wav = Figure(figsize=(8, 5.5), dpi=100, facecolor='#1e222a')
        self.ax_wav_spec = self.fig_wav.add_subplot(211, facecolor='#21252b')
        self.ax_wav_slice = self.fig_wav.add_subplot(212, facecolor='#21252b')
        self.fig_wav.tight_layout(pad=2.5)

        self.canvas_wav = FigureCanvasTkAgg(self.fig_wav, master=tab_wav)
        self.canvas_wav.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_wav.mpl_connect("button_press_event", self._on_wav_canvas_click)

        self.toolbar_wav = NavigationToolbar2Tk(self.canvas_wav, tab_wav)
        self.toolbar_wav.config(background='#1e222a')
        self.toolbar_wav.update()

        # Tab 4: Harmonic Analysis & Estimator Consensus
        tab_fusion = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(tab_fusion, text="Consensus & Harmonics")

        self.fusion_text = tk.Text(tab_fusion, bg='#21252b', fg='#61afef', font=('Monospace', 10), wrap=tk.WORD)
        self.fusion_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Bottom Results Table & Status Bar
        results_frame = ttk.LabelFrame(right_frame, text="Decomposition Results Table", padding=5)
        results_frame.pack(fill=tk.X, padx=5, pady=5)

        cols = ("idx", "bank", "freq_bin", "freq_hz", "peak_amp", "wiener_gain")
        self.results_tree = ttk.Treeview(results_frame, columns=cols, show="headings", height=4)
        self.results_tree.heading("idx", text="#")
        self.results_tree.column("idx", width=30, anchor="center")
        self.results_tree.heading("bank", text="Grid Mode")
        self.results_tree.column("bank", width=100, anchor="center")
        self.results_tree.heading("freq_bin", text="Bin (Q16)")
        self.results_tree.column("freq_bin", width=100, anchor="center")
        self.results_tree.heading("freq_hz", text="Freq (Hz)")
        self.results_tree.column("freq_hz", width=110, anchor="center")
        self.results_tree.heading("peak_amp", text="Peak Amp")
        self.results_tree.column("peak_amp", width=100, anchor="center")
        self.results_tree.heading("wiener_gain", text="Wiener Gain")
        self.results_tree.column("wiener_gain", width=100, anchor="center")
        self.results_tree.pack(fill=tk.X, side=tk.LEFT, expand=True)

        self.status_label = ttk.Label(results_frame, text="Ready.", font=("Monospace", 9, "bold"), foreground="#98c379")
        self.status_label.pack(fill=tk.Y, side=tk.RIGHT, padx=10)

    def _build_source_tab(self, parent):
        mode_frame = ttk.LabelFrame(parent, text="Input Source Mode", padding=5)
        mode_frame.pack(fill=tk.X, pady=4)

        ttk.Radiobutton(mode_frame, text="Synthetic Generator", variable=self.input_mode, value="generated", command=self._on_params_changed).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="WAV Audio File", variable=self.input_mode, value="file", command=self._on_params_changed).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Live / Continuous Stream", variable=self.input_mode, value="live", command=self._on_params_changed).pack(anchor=tk.W)

        file_frame = ttk.LabelFrame(parent, text="Audio File Controls", padding=5)
        file_frame.pack(fill=tk.X, pady=4)

        self.file_label = ttk.Label(file_frame, text="No WAV file loaded", wraplength=350)
        self.file_label.pack(fill=tk.X, pady=2)

        file_btns = ttk.Frame(file_frame)
        file_btns.pack(fill=tk.X)
        ttk.Button(file_btns, text="Browse WAV...", command=self._browse_wav_file).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_play_file = ttk.Button(file_btns, text="Play Continuous", command=self._toggle_file_playback)
        self.btn_play_file.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        gen_frame = ttk.LabelFrame(parent, text="Synthetic Waveform Synthesizer", padding=5)
        gen_frame.pack(fill=tk.X, pady=4)

        ttk.Label(gen_frame, text="Waveform:").grid(row=0, column=0, sticky=tk.W)
        wv_cb = ttk.Combobox(gen_frame, textvariable=self.waveform_type, values=["Sine Multi-Tone", "Square Wave", "Sawtooth", "Chirp Sweep"], state="readonly", width=15)
        wv_cb.grid(row=0, column=1, sticky=tk.W, columnspan=2)
        wv_cb.bind("<<ComboboxSelected>>", lambda e: self._on_params_changed())

        ttk.Label(gen_frame, text="DC Offset:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(gen_frame, textvariable=self.dc_offset, width=10).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(gen_frame, text="Noise StdDev:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(gen_frame, textvariable=self.noise_std, width=10).grid(row=2, column=1, sticky=tk.W)

        for idx, t_cfg in enumerate(self.tones_config):
            r = 3 + idx * 2
            ttk.Checkbutton(gen_frame, text=f"Tone {idx+1}", variable=t_cfg["enable"], command=self._on_params_changed).grid(row=r, column=0, sticky=tk.W, pady=(4,0))
            ttk.Label(gen_frame, text="  Freq (Hz):").grid(row=r+1, column=0, sticky=tk.W)
            ttk.Entry(gen_frame, textvariable=t_cfg["freq"], width=8).grid(row=r+1, column=1, sticky=tk.W)
            ttk.Label(gen_frame, text=" Amp:").grid(row=r+1, column=2, sticky=tk.W)
            ttk.Entry(gen_frame, textvariable=t_cfg["amp"], width=8).grid(row=r+1, column=3, sticky=tk.W)

        live_frame = ttk.LabelFrame(parent, text="Live Input / Streaming", padding=5)
        live_frame.pack(fill=tk.X, pady=4)

        self.stream_btn = ttk.Button(live_frame, text="Start Live Streaming", command=self._toggle_live_streaming)
        self.stream_btn.pack(fill=tk.X)

        ttk.Button(parent, text="Run / Refresh Analysis", command=self._on_params_changed).pack(fill=tk.X, pady=8)

    def _build_dsp_tab(self, parent):
        grid_frame = ttk.LabelFrame(parent, text="Frequency Search Grid & Mode", padding=5)
        grid_frame.pack(fill=tk.X, pady=5)

        ttk.Label(grid_frame, text="Grid Mode:").grid(row=0, column=0, sticky=tk.W)
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

        algo_frame = ttk.LabelFrame(parent, text="Decomposition & Stopping", padding=5)
        algo_frame.pack(fill=tk.X, pady=5)

        ttk.Label(algo_frame, text="Max Tones (1-16):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(algo_frame, textvariable=self.max_tones, width=12).grid(row=0, column=1)

        ttk.Label(algo_frame, text="Min Amplitude:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(algo_frame, textvariable=self.min_amplitude, width=12).grid(row=1, column=1)

        wiener_frame = ttk.LabelFrame(parent, text="Wiener Shrinkage Filter", padding=5)
        wiener_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(wiener_frame, text="Enable Wiener Shrinkage", variable=self.use_wiener, command=self._on_params_changed).pack(anchor=tk.W)
        ttk.Label(wiener_frame, text="Noise Power Scale:").pack(anchor=tk.W)
        ttk.Entry(wiener_frame, textvariable=self.wiener_noise_scale, width=12).pack(anchor=tk.W)

    def _build_viz_tab(self, parent):
        preset_frame = ttk.LabelFrame(parent, text="DSP Test Preset Scenarios", padding=5)
        preset_frame.pack(fill=tk.X, pady=5)

        ttk.Label(preset_frame, text="Scenario:").grid(row=0, column=0, sticky=tk.W, pady=2)
        presets = ["Default Multi-Tone", "DTMF Key 5 (770 + 1336 Hz)", "Concert Pitch A440 Harmonics", "Low-SNR Chirp Sweep", "Two-Tone IMD Test"]
        preset_cb = ttk.Combobox(preset_frame, textvariable=self.preset_scenario, values=presets, state="readonly", width=22)
        preset_cb.grid(row=0, column=1, sticky=tk.W, pady=2)
        preset_cb.bind("<<ComboboxSelected>>", self._load_preset_scenario)

        viz_frame = ttk.LabelFrame(parent, text="Plot Customization Options", padding=5)
        viz_frame.pack(fill=tk.X, pady=5)

        ttk.Label(viz_frame, text="Freq Axis Scale:").grid(row=0, column=0, sticky=tk.W, pady=2)
        scale_cb = ttk.Combobox(viz_frame, textvariable=self.freq_scale_mode, values=["linear", "log"], state="readonly", width=12)
        scale_cb.grid(row=0, column=1, sticky=tk.W, pady=2)
        scale_cb.bind("<<ComboboxSelected>>", self._on_freq_scale_mode_changed)

        ttk.Label(viz_frame, text="Waterfall Colormap:").grid(row=1, column=0, sticky=tk.W, pady=2)
        cmap_cb = ttk.Combobox(viz_frame, textvariable=self.waterfall_cmap, values=["magma", "viridis", "plasma", "inferno", "coolwarm"], state="readonly", width=12)
        cmap_cb.grid(row=1, column=1, sticky=tk.W, pady=2)
        cmap_cb.bind("<<ComboboxSelected>>", lambda e: self._render_active_tab(self.sample_rate.get(), self.block_size.get()))

        ttk.Checkbutton(viz_frame, text="Show Peak Frequency Markers", variable=self.show_peak_markers,
                        command=lambda: self._render_active_tab(self.sample_rate.get(), self.block_size.get())).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Checkbutton(viz_frame, text="Show Peak Frequency Text Labels", variable=self.show_peak_labels,
                        command=lambda: self._render_active_tab(self.sample_rate.get(), self.block_size.get())).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        btn_clr_wf = ttk.Button(viz_frame, text="Clear Waterfall History", command=self._clear_waterfall)
        btn_clr_wf.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=6)

        btn_rst_zoom = ttk.Button(viz_frame, text="Reset Spectrum Zoom / Limits", command=self._reset_freq_zoom)
        btn_rst_zoom.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=4)

    def _init_persistent_plots(self):
        Fs = self.sample_rate.get()
        # Initialize ax_time static artists
        self.line_time_in, = self.ax_time.plot([], [], label="Input Signal", color="#61afef", alpha=0.7)
        self.line_time_mod, = self.ax_time.plot([], [], label="Decomposed Model", color="#98c379", linewidth=1.5)
        self.line_time_res, = self.ax_time.plot([], [], label="Residual Error", color="#e06c75", linestyle="--", alpha=0.6)
        self.ax_time.set_title("Time Domain Signal, Fitted Model & Residual", color="#abb2bf")
        self.ax_time.set_xlabel("Time (ms)", color="#abb2bf")
        self.ax_time.set_ylabel("Amplitude", color="#abb2bf")
        self.ax_time.tick_params(colors='#abb2bf')
        self.ax_time.legend(loc="upper right", fontsize=8, facecolor='#21252b', edgecolor='#3b4048', labelcolor='#abb2bf')
        self.ax_time.grid(True, linestyle=":", alpha=0.4)

        # Initialize ax_freq static artists
        self.line_fft, = self.ax_freq.plot([], [], label="Reference FFT Spectrum", color="#abb2bf", alpha=0.4)
        self.line_stem, = self.ax_freq.plot([], [], color="#e06c75", linewidth=2, label="IntSpectrum Tones")
        self.pts_stem, = self.ax_freq.plot([], [], 'o', color="#e06c75", zorder=5)
        self.ax_freq.set_title("Frequency Domain Spectrum (FFT vs Integer Grid Tones)", color="#abb2bf")
        self.ax_freq.set_xlabel("Frequency (Hz)", color="#abb2bf")
        self.ax_freq.set_ylabel("Peak Amplitude", color="#abb2bf")
        self.ax_freq.set_xlim(0, Fs / 2.0)
        self.ax_freq.set_ylim(0, 32768)
        self.ax_freq.tick_params(colors='#abb2bf')
        self.ax_freq.legend(loc="upper right", fontsize=8, facecolor='#21252b', edgecolor='#3b4048', labelcolor='#abb2bf')
        self.ax_freq.grid(True, linestyle=":", alpha=0.4)

        # Initialize ax_wav_spec static artists
        self.im_wav_spec = self.ax_wav_spec.imshow([[0]], aspect='auto', origin='lower', extent=[0, 1, 0, Fs/2], cmap='magma')
        self.pts_wav_tones = self.ax_wav_spec.scatter([], [], c='#98c379', s=25, alpha=0.8, edgecolors='none', label='IntSpectrum Tones', zorder=4)
        self.cursor_wav = self.ax_wav_spec.axvline(x=0.0, color='#e5c07b', linewidth=1.8, linestyle='--', zorder=5)
        self.ax_wav_spec.set_title("Full Audio File Spectrogram & Detected Tone Trajectory", color="#abb2bf")
        self.ax_wav_spec.set_xlabel("Time (seconds)", color="#abb2bf")
        self.ax_wav_spec.set_ylabel("Frequency (Hz)", color="#abb2bf")
        self.ax_wav_spec.set_xlim(0, 1.0)
        self.ax_wav_spec.set_ylim(0, Fs / 2.0)
        self.ax_wav_spec.tick_params(colors='#abb2bf')
        self.ax_wav_spec.legend(loc="upper right", fontsize=8, facecolor='#21252b', edgecolor='#3b4048', labelcolor='#abb2bf')
        self.ax_wav_spec.grid(True, linestyle=":", alpha=0.3)

        # Initialize ax_wav_slice static artists
        self.line_wav_slice_fft, = self.ax_wav_slice.plot([], [], label="Instantaneous FFT Spectrum", color="#61afef", alpha=0.7)
        self.line_wav_slice_stems, = self.ax_wav_slice.plot([], [], color="#98c379", linewidth=2, label="IntSpectrum Tones")
        self.pts_wav_slice_stems, = self.ax_wav_slice.plot([], [], 'o', color="#98c379", zorder=5)
        self.ax_wav_slice.set_title("Current Playhead Frequency Spectrum", color="#abb2bf")
        self.ax_wav_slice.set_xlabel("Frequency (Hz)", color="#abb2bf")
        self.ax_wav_slice.set_ylabel("Peak Amplitude", color="#abb2bf")
        self.ax_wav_slice.set_xlim(0, Fs / 2.0)
        self.ax_wav_slice.set_ylim(0, 32768)
        self.ax_wav_slice.tick_params(colors='#abb2bf')
        self.ax_wav_slice.legend(loc="upper right", fontsize=8, facecolor='#21252b', edgecolor='#3b4048', labelcolor='#abb2bf')
        self.ax_wav_slice.grid(True, linestyle=":", alpha=0.3)

        self.last_plotted_fs = Fs
        self.last_scale_mode = self.freq_scale_mode.get()

    def _reset_freq_zoom(self):
        Fs = self.sample_rate.get()
        if self.freq_scale_mode.get() == "log":
            self.ax_freq.set_xlim(20.0, Fs / 2.0)
        else:
            self.ax_freq.set_xlim(0, Fs / 2.0)
        self.ax_freq.set_ylim(0, 32768)
        if hasattr(self, 'toolbar_overview'):
            self.toolbar_overview.update()
        self.canvas_overview.draw_idle()

    def _on_freq_scale_mode_changed(self, event=None):
        mode = self.freq_scale_mode.get()
        Fs = self.sample_rate.get()
        if mode == "log":
            self.ax_freq.set_xscale("log")
            self.ax_freq.set_xlim(20.0, Fs / 2.0)
        else:
            self.ax_freq.set_xscale("linear")
            self.ax_freq.set_xlim(0, Fs / 2.0)
        self.last_scale_mode = mode
        self._render_active_tab(Fs, self.block_size.get())

    def _load_preset_scenario(self, event=None):
        preset = self.preset_scenario.get()
        if preset == "Default Multi-Tone":
            self.waveform_type.set("Sine Multi-Tone")
            self.tones_config[0]["freq"].set(222.65625); self.tones_config[0]["amp"].set(12000.0); self.tones_config[0]["enable"].set(True)
            self.tones_config[1]["freq"].set(432.03125); self.tones_config[1]["amp"].set(6500.0);  self.tones_config[1]["enable"].set(True)
            self.tones_config[2]["freq"].set(517.1875);  self.tones_config[2]["amp"].set(3500.0);  self.tones_config[2]["enable"].set(True)
            self.tones_config[3]["enable"].set(False); self.tones_config[4]["enable"].set(False)
            self.noise_std.set(0.0)
            self.grid_min_hz.set(100.0); self.grid_max_hz.set(700.0)
        elif preset == "DTMF Key 5 (770 + 1336 Hz)":
            self.waveform_type.set("Sine Multi-Tone")
            self.tones_config[0]["freq"].set(770.0);  self.tones_config[0]["amp"].set(10000.0); self.tones_config[0]["enable"].set(True)
            self.tones_config[1]["freq"].set(1336.0); self.tones_config[1]["amp"].set(10000.0); self.tones_config[1]["enable"].set(True)
            self.tones_config[2]["enable"].set(False); self.tones_config[3]["enable"].set(False); self.tones_config[4]["enable"].set(False)
            self.grid_min_hz.set(500.0); self.grid_max_hz.set(1600.0)
            self.noise_std.set(200.0)
        elif preset == "Concert Pitch A440 Harmonics":
            self.waveform_type.set("Sine Multi-Tone")
            self.tones_config[0]["freq"].set(440.0);  self.tones_config[0]["amp"].set(12000.0); self.tones_config[0]["enable"].set(True)
            self.tones_config[1]["freq"].set(880.0);  self.tones_config[1]["amp"].set(6000.0);  self.tones_config[1]["enable"].set(True)
            self.tones_config[2]["freq"].set(1320.0); self.tones_config[2]["amp"].set(3000.0);  self.tones_config[2]["enable"].set(True)
            self.tones_config[3]["enable"].set(False); self.tones_config[4]["enable"].set(False)
            self.grid_min_hz.set(300.0); self.grid_max_hz.set(1600.0)
        elif preset == "Low-SNR Chirp Sweep":
            self.waveform_type.set("Chirp Sweep")
            self.tones_config[0]["amp"].set(8000.0)
            self.grid_min_hz.set(150.0); self.grid_max_hz.set(900.0)
            self.noise_std.set(1200.0)
            self.use_wiener.set(True)
        elif preset == "Two-Tone IMD Test":
            self.waveform_type.set("Sine Multi-Tone")
            self.tones_config[0]["freq"].set(300.0); self.tones_config[0]["amp"].set(10000.0); self.tones_config[0]["enable"].set(True)
            self.tones_config[1]["freq"].set(340.0); self.tones_config[1]["amp"].set(10000.0); self.tones_config[1]["enable"].set(True)
            self.tones_config[2]["enable"].set(False); self.tones_config[3]["enable"].set(False); self.tones_config[4]["enable"].set(False)
            self.grid_min_hz.set(200.0); self.grid_max_hz.set(500.0)

        self._on_params_changed()

    def _clear_waterfall(self):
        self.waterfall_history.clear()
        self._render_active_tab(self.sample_rate.get(), self.block_size.get())

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
                        samples = samples[::nchannels]

                    self.audio_file_path = filename
                    self.audio_samples = samples
                    self.sample_rate.set(float(framerate))
                    self.file_read_pos = 0
                    self.file_label.config(text=f"{os.path.basename(filename)}\n({len(samples)} samples, {framerate} Hz)")
                    self.input_mode.set("file")
                    self._on_params_changed()
                    self._precompute_wav_cache()
            except Exception as e:
                messagebox.showerror("File Error", f"Error opening WAV file: {e}")

    def _precompute_wav_cache(self):
        if self.audio_samples is None or len(self.audio_samples) == 0:
            return

        Fs = float(self.sample_rate.get())
        N = int(self.block_size.get())
        hop = max(64, N // 4)
        total_len = len(self.audio_samples)
        duration = total_len / Fs

        frames = []
        frame_times = []
        for i in range(0, total_len - N + 1, hop):
            frames.append(self.audio_samples[i : i + N])
            frame_times.append((i + N / 2.0) / Fs)

        if not frames:
            frames.append(np.pad(self.audio_samples, (0, max(0, N - total_len)), mode='constant'))
            frame_times.append(0.5 * N / Fs)

        frames_arr = np.array(frames, dtype=np.float32)
        window = np.hanning(N)
        windowed = frames_arr * window
        fft_complex = np.fft.rfft(windowed, axis=1)
        mags = np.abs(fft_complex) * (2.0 / np.sum(window))
        mags_db = 20.0 * np.log10(np.maximum(mags, 1e-4))
        # Spectrogram matrix format: freqs on vertical axis, time on horizontal
        spec_2d = mags_db.T
        freq_axis = np.fft.rfftfreq(N, d=1.0 / Fs)

        # Multi-frame tone detection across the audio file using int_spectrum
        detected_tones_all = []
        grid_pts = max(2, int(self.grid_points.get()))
        min_amp = max(0, int(self.min_amplitude.get()))
        max_t = max(1, min(16, int(self.max_tones.get())))

        # Grid scanning up to Nyquist
        f_min = max(10.0, float(self.grid_min_hz.get()))
        f_max = max(float(self.grid_max_hz.get()), Fs / 2.0 - 50.0)
        grid = (ctypes.c_uint32 * grid_pts)()
        self.api.lib.int_spectrum_make_linear_grid(N, ctypes.c_double(f_min), ctypes.c_double(f_max), ctypes.c_double(Fs), grid, grid_pts)

        c_tones = (IntSpectrumTone * 16)()
        c_tone_count = ctypes.c_size_t(0)
        c_res = (ctypes.c_int64 * N)()
        c_dc = ctypes.c_int32(0)

        # Stride analysis frames if WAV is long to keep computation under 200ms
        stride = max(1, len(frames) // 100)
        for idx in range(0, len(frames), stride):
            t_center = frame_times[idx]
            sub_chunk = np.ascontiguousarray(frames[idx], dtype=np.int16)
            c_chunk = sub_chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
            ok = self.api.lib.int_spectrum_decompose_grid(
                N, c_chunk, grid, grid_pts, min_amp, max_t, c_tones, ctypes.byref(c_tone_count), c_res, ctypes.byref(c_dc), None, None
            )
            if ok:
                for k in range(min(c_tone_count.value, max_t)):
                    hz = self.api.lib.int_spectrum_bin_q16_to_hz(N, c_tones[k].frequencyQ16, ctypes.c_double(Fs))
                    amp = c_tones[k].peakAmplitude
                    detected_tones_all.append((t_center, hz, amp))

        self.wav_spec_cache = {
            "S_db": spec_2d,
            "times": np.array(frame_times),
            "freqs": freq_axis,
            "detected_tones": detected_tones_all,
            "Fs": Fs,
            "duration": duration,
            "N": N,
            "hop": hop
        }

        # Initialize / update full spectrogram plot
        self.wav_playhead_sec = 0.0
        self.wav_time_slider.config(to=max(0.01, duration))
        self.wav_time_slider.set(0.0)
        self.wav_time_label.config(text=f"0.00 / {duration:.2f} s")

        self.im_wav_spec.set_data(spec_2d)
        vmin = np.percentile(spec_2d, 5)
        vmax = np.percentile(spec_2d, 99.5)
        self.im_wav_spec.set_clim(vmin=vmin, vmax=max(vmax, vmin + 10))

        is_log = (self.wav_scale_mode.get() == "log")
        y_bot = 20.0 if is_log else 0.0
        self.im_wav_spec.set_extent([0, duration, y_bot, Fs / 2.0])

        if detected_tones_all:
            t_pts = [item[0] for item in detected_tones_all]
            f_pts = [item[1] for item in detected_tones_all]
            self.pts_wav_tones.set_offsets(np.column_stack([t_pts, f_pts]))
        else:
            self.pts_wav_tones.set_offsets(np.empty((0, 2)))

        self.cursor_wav.set_xdata([0.0, 0.0])
        self._reset_wav_zoom()
        self._update_wav_slice_view()
        self.canvas_wav.draw_idle()

    def _reset_wav_zoom(self):
        if self.wav_spec_cache is None:
            return
        duration = self.wav_spec_cache["duration"]
        Fs = self.wav_spec_cache["Fs"]
        is_log = (self.wav_scale_mode.get() == "log")

        self.ax_wav_spec.set_xlim(0, max(0.01, duration))
        if is_log:
            self.ax_wav_spec.set_yscale("log")
            self.ax_wav_spec.set_ylim(20.0, Fs / 2.0)
            self.ax_wav_slice.set_xscale("log")
            self.ax_wav_slice.set_xlim(20.0, Fs / 2.0)
        else:
            self.ax_wav_spec.set_yscale("linear")
            self.ax_wav_spec.set_ylim(0, Fs / 2.0)
            self.ax_wav_slice.set_xscale("linear")
            self.ax_wav_slice.set_xlim(0, Fs / 2.0)

        self.ax_wav_slice.set_ylim(0, 32768)
        if hasattr(self, 'toolbar_wav'):
            self.toolbar_wav.update()
        self.canvas_wav.draw_idle()

    def _on_wav_scale_mode_changed(self, event=None):
        self._reset_wav_zoom()

    def _zoom_wav_view(self, factor):
        if self.wav_spec_cache is None:
            return

        x0, x1 = self.ax_wav_spec.get_xlim()
        cx = 0.5 * (x0 + x1)
        new_hx = 0.5 * (x1 - x0) * factor
        self.ax_wav_spec.set_xlim(max(0, cx - new_hx), min(self.wav_spec_cache["duration"], cx + new_hx))

        y0, y1 = self.ax_wav_spec.get_ylim()
        Fs = self.wav_spec_cache["Fs"]
        if self.wav_scale_mode.get() == "log":
            y0 = max(20.0, y0)
            cy = np.sqrt(y0 * y1)
            new_y0 = max(20.0, cy * ((y0 / cy) ** factor))
            new_y1 = min(Fs / 2.0, cy * ((y1 / cy) ** factor))
            self.ax_wav_spec.set_ylim(new_y0, new_y1)
            self.ax_wav_slice.set_xlim(new_y0, new_y1)
        else:
            cy = 0.5 * (y0 + y1)
            new_hy = 0.5 * (y1 - y0) * factor
            new_y0 = max(0, cy - new_hy)
            new_y1 = min(Fs / 2.0, cy + new_hy)
            self.ax_wav_spec.set_ylim(new_y0, new_y1)
            self.ax_wav_slice.set_xlim(new_y0, new_y1)

        if hasattr(self, 'toolbar_wav'):
            self.toolbar_wav.update()
        self.canvas_wav.draw_idle()

    def _on_wav_canvas_click(self, event):
        if event.inaxes == self.ax_wav_spec and event.xdata is not None and self.wav_spec_cache:
            t_click = max(0.0, min(self.wav_spec_cache["duration"], float(event.xdata)))
            self.wav_playhead_sec = t_click
            self.wav_time_slider.set(t_click)
            self.wav_time_label.config(text=f"{t_click:.2f} / {self.wav_spec_cache['duration']:.2f} s")
            self.cursor_wav.set_xdata([t_click, t_click])
            self._update_wav_slice_view()
            self.canvas_wav.draw_idle()
            if self.wav_is_playing:
                self._start_wav_audio_playback(seek_t=t_click)

    def _on_wav_scrub(self, val):
        if self.wav_spec_cache:
            t = float(val)
            self.wav_playhead_sec = t
            self.wav_time_label.config(text=f"{t:.2f} / {self.wav_spec_cache['duration']:.2f} s")
            self.cursor_wav.set_xdata([t, t])
            self._update_wav_slice_view()
            self.canvas_wav.draw_idle()
            if self.wav_is_playing:
                self._start_wav_audio_playback(seek_t=t)

    def _toggle_wav_playback(self):
        if self.wav_is_playing:
            self.wav_is_playing = False
            self.btn_wav_play.config(text="▶ Play")
            self._stop_wav_audio_playback()
        else:
            if self.wav_spec_cache is None:
                messagebox.showwarning("File Required", "Please load a WAV file first.")
                return
            self.wav_is_playing = True
            self.btn_wav_play.config(text="⏸ Pause")
            self._start_wav_audio_playback(seek_t=self.wav_playhead_sec)

    def _stop_wav_playback(self):
        self.wav_is_playing = False
        self.btn_wav_play.config(text="▶ Play")
        self.wav_playhead_sec = 0.0
        self._stop_wav_audio_playback()
        if self.wav_spec_cache:
            self.wav_time_slider.set(0.0)
            self.wav_time_label.config(text=f"0.00 / {self.wav_spec_cache['duration']:.2f} s")
            self.cursor_wav.set_xdata([0.0, 0.0])
            self._update_wav_slice_view()
            self.canvas_wav.draw_idle()
        if self.wav_is_playing:
            self._start_wav_audio_playback(seek_t=0.0)

    def _stop_wav_audio_playback(self):
        # 1. Stop sounddevice stream if active
        if self.wav_audio_stream is not None:
            try:
                self.wav_audio_stream.stop()
                self.wav_audio_stream.close()
            except Exception:
                pass
            self.wav_audio_stream = None

        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        # 2. Terminate background aplay/paplay process if running
        if self.wav_playback_proc is not None:
            try:
                self.wav_playback_proc.terminate()
                self.wav_playback_proc.wait(timeout=0.2)
            except Exception:
                try:
                    self.wav_playback_proc.kill()
                except Exception:
                    pass
            self.wav_playback_proc = None

    def _start_wav_audio_playback(self, seek_t=0.0):
        self._stop_wav_audio_playback()
        if self.audio_samples is None or len(self.audio_samples) == 0:
            return

        Fs = float(self.sample_rate.get())
        start_idx = max(0, min(len(self.audio_samples), int(round(seek_t * Fs))))
        samples_to_play = self.audio_samples[start_idx:]
        if len(samples_to_play) == 0:
            return

        # Try real-time streaming via sounddevice callback stream
        played_stream = False
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            has_output_device = any(d.get('max_output_channels', 0) > 0 for d in devs) if isinstance(devs, list) else False
            if has_output_device:
                current_sample_idx = [start_idx]
                total_samples = len(self.audio_samples)

                def audio_out_callback(outdata, frames, time_info, status):
                    c_idx = current_sample_idx[0]
                    remaining = total_samples - c_idx
                    if remaining <= 0:
                        outdata.fill(0)
                        raise sd.CallbackStop()
                    take = min(frames, remaining)
                    chunk = self.audio_samples[c_idx : c_idx + take].astype(np.float32) / 32768.0
                    outdata[:take, 0] = chunk
                    if take < frames:
                        outdata[take:, 0] = 0
                    current_sample_idx[0] += take
                    self.wav_playhead_sec = current_sample_idx[0] / Fs

                stream = sd.OutputStream(samplerate=int(Fs), channels=1, callback=audio_out_callback)
                stream.start()
                self.wav_audio_stream = stream
                played_stream = True
        except Exception:
            played_stream = False

        if not played_stream:
            # Fallback 1: Try sd.play on slice
            try:
                import sounddevice as sd
                devs = sd.query_devices()
                has_output_device = any(d.get('max_output_channels', 0) > 0 for d in devs) if isinstance(devs, list) else False
                if has_output_device:
                    samples_float = samples_to_play.astype(np.float32) / 32768.0
                    sd.play(samples_float, int(Fs))
                    played_stream = True
            except Exception:
                played_stream = False

        if not played_stream:
            # Fallback 2: Spawn background aplay process
            import shutil
            import subprocess
            playback_cmd = shutil.which("aplay") or shutil.which("paplay")
            if playback_cmd:
                try:
                    tmp_wav = "/tmp/int_spectrum_wav_playback.wav"
                    with wave.open(tmp_wav, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(int(Fs))
                        wf.writeframes(samples_to_play.tobytes())
                    self.wav_playback_proc = subprocess.Popen([playback_cmd, tmp_wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass

    def _rewind_wav_playback(self):
        self.wav_playhead_sec = 0.0
        if self.wav_spec_cache:
            self.wav_time_slider.set(0.0)
            self.wav_time_label.config(text=f"0.00 / {self.wav_spec_cache['duration']:.2f} s")
            self.cursor_wav.set_xdata([0.0, 0.0])
            self._update_wav_slice_view()
            self.canvas_wav.draw_idle()

    def _update_wav_slice_view(self):
        if self.wav_spec_cache is None:
            return

        Fs = self.wav_spec_cache["Fs"]
        N = self.wav_spec_cache["N"]
        t = self.wav_playhead_sec
        idx_sample = int(round(t * Fs))
        total_len = len(self.audio_samples)

        if idx_sample + N <= total_len:
            slice_samples = self.audio_samples[idx_sample : idx_sample + N]
        else:
            pad_n = max(0, idx_sample + N - total_len)
            slice_samples = np.pad(self.audio_samples[max(0, idx_sample) :], (0, pad_n), mode='constant')

        # Compute instant slice FFT
        fft_mags = np.abs(np.fft.rfft(slice_samples)) * (2.0 / N)
        fft_freqs = np.fft.rfftfreq(N, d=1.0 / Fs)
        self.line_wav_slice_fft.set_data(fft_freqs, fft_mags)

        # Decompose current slice using int_spectrum
        grid_pts = max(2, int(self.grid_points.get()))
        min_amp = max(0, int(self.min_amplitude.get()))
        max_t = max(1, min(16, int(self.max_tones.get())))

        f_min = max(10.0, float(self.grid_min_hz.get()))
        f_max = max(float(self.grid_max_hz.get()), Fs / 2.0 - 50.0)
        grid = (ctypes.c_uint32 * grid_pts)()
        self.api.lib.int_spectrum_make_linear_grid(N, ctypes.c_double(f_min), ctypes.c_double(f_max), ctypes.c_double(Fs), grid, grid_pts)

        c_tones = (IntSpectrumTone * 16)()
        c_tone_count = ctypes.c_size_t(0)
        c_res = (ctypes.c_int64 * N)()
        c_dc = ctypes.c_int32(0)

        chunk_arr = np.ascontiguousarray(slice_samples, dtype=np.int16)
        c_chunk = chunk_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        ok = self.api.lib.int_spectrum_decompose_grid(
            N, c_chunk, grid, grid_pts, min_amp, max_t, c_tones, ctypes.byref(c_tone_count), c_res, ctypes.byref(c_dc), None, None
        )

        stem_xs = []
        stem_ys = []
        det_f = []
        det_a = []
        if ok:
            for k in range(min(c_tone_count.value, max_t)):
                hz = self.api.lib.int_spectrum_bin_q16_to_hz(N, c_tones[k].frequencyQ16, ctypes.c_double(Fs))
                amp = c_tones[k].peakAmplitude
                stem_xs.extend([hz, hz, np.nan])
                stem_ys.extend([0, amp, np.nan])
                det_f.append(hz)
                det_a.append(amp)

        self.line_wav_slice_stems.set_data(stem_xs, stem_ys)
        self.pts_wav_slice_stems.set_data(det_f, det_a)
        self.ax_wav_slice.set_title(f"Instantaneous Spectrum at t = {t:.2f} s ({len(det_f)} Tones Detected)", color="#abb2bf")

    def _toggle_file_playback(self):
        if self.is_file_playing:
            self.is_file_playing = False
            self.btn_play_file.config(text="Play Continuous")
        else:
            if self.audio_samples is None:
                messagebox.showwarning("File Required", "Please load a WAV file first.")
                return
            self.is_file_playing = True
            self.input_mode.set("file")
            self.btn_play_file.config(text="Pause Playback")

    def _toggle_live_streaming(self):
        if self.is_streaming:
            self.is_streaming = False
            self.stream_btn.config(text="Start Live Streaming")
        else:
            self.is_streaming = True
            self.input_mode.set("live")
            self.stream_btn.config(text="Stop Live Streaming")
            self.stream_thread = threading.Thread(target=self._live_audio_worker, daemon=True)
            self.stream_thread.start()

    def _live_audio_worker(self):
        if HAS_SOUNDDEVICE:
            try:
                def audio_callback(indata, frames, time_info, status):
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
                print(f"Hardware audio device fallback: {e}")

        t_accum = 0.0
        while self.is_streaming:
            N = self.block_size.get()
            Fs = self.sample_rate.get()
            f1 = 200.0 + 120.0 * math.sin(2.0 * math.pi * 0.25 * t_accum)
            f2 = 450.0 + 80.0 * math.cos(2.0 * math.pi * 0.15 * t_accum)
            t = np.arange(N) / Fs + t_accum
            t_accum += N / Fs

            sig = 10000.0 * np.sin(2.0 * np.pi * f1 * t) + 6000.0 * np.sin(2.0 * np.pi * f2 * t) + 1200.0
            sig += np.random.normal(0, 300, N)
            int_samples = np.clip(np.round(sig), -32768, 32767).astype(np.int16)

            if not self.audio_queue.full():
                self.audio_queue.put(int_samples)
            time.sleep(N / Fs)

    def _play_current_audio(self):
        if self.last_input_samples is None:
            messagebox.showwarning("Audio Error", "No active signal buffer to play.")
            return

        fs = int(self.sample_rate.get())
        played_hardware = False

        # 1. Try playing via sounddevice if an output device is available
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            has_output_device = any(d.get('max_output_channels', 0) > 0 for d in devs) if isinstance(devs, list) else False
            if has_output_device:
                samples_float = self.last_input_samples.astype(np.float32) / 32768.0
                sd.play(samples_float, fs)
                played_hardware = True
        except Exception:
            pass

        # 2. Try playing via ALSA aplay or PulseAudio paplay if sounddevice was not used
        if not played_hardware:
            import shutil
            import subprocess
            playback_cmd = shutil.which("aplay") or shutil.which("paplay")
            if playback_cmd:
                try:
                    tmp_wav = "/tmp/int_spectrum_playback.wav"
                    with wave.open(tmp_wav, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(fs)
                        wf.writeframes(self.last_input_samples.tobytes())
                    proc = subprocess.run([playback_cmd, tmp_wav], capture_output=True, timeout=1.0)
                    if proc.returncode == 0:
                        played_hardware = True
                except Exception:
                    pass

        # 3. If hardware soundcard is not present (e.g. headless Linux container)
        if not played_hardware:
            tmp_wav = "/tmp/int_spectrum_playback.wav"
            try:
                with wave.open(tmp_wav, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(fs)
                    wf.writeframes(self.last_input_samples.tobytes())
                self.status_label.config(
                    text=f"Audio saved to {tmp_wav} (no soundcard)", foreground="#e5c07b"
                )
                messagebox.showinfo(
                    "Audio Playback",
                    f"No physical soundcard detected in this environment.\n\n"
                    f"The active signal ({len(self.last_input_samples)} samples @ {fs} Hz) has been saved to:\n"
                    f"{tmp_wav}\n\nYou can download or play this WAV file externally."
                )
            except Exception as e:
                messagebox.showerror("Audio Error", f"Failed to write audio playback file: {e}")

    def _save_audio_wav(self):
        if self.last_input_samples is None:
            messagebox.showwarning("Save Error", "No active signal buffer to save.")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if filename:
            try:
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(int(self.sample_rate.get()))
                    wf.writeframes(self.last_input_samples.tobytes())
                messagebox.showinfo("Save Success", f"Saved audio block to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save WAV: {e}")

    def _gui_update_loop(self):
        if self.is_streaming and not self.audio_queue.empty():
            try:
                samples = self.audio_queue.get_nowait()
                self._run_analysis_on_samples(samples)
            except queue.Empty:
                pass

        elif self.is_file_playing and self.audio_samples is not None:
            N = self.block_size.get()
            if self.file_read_pos + N > len(self.audio_samples):
                self.file_read_pos = 0
            samples = self.audio_samples[self.file_read_pos : self.file_read_pos + N]
            self.file_read_pos += N // 2
            self._run_analysis_on_samples(samples)

        if self.wav_is_playing and self.wav_spec_cache is not None:
            # If sounddevice callback stream is not driving wav_playhead_sec, advance it by timer tick
            if self.wav_audio_stream is None:
                dt = 0.040 # 40ms tick
                duration = self.wav_spec_cache["duration"]
                self.wav_playhead_sec += dt
                if self.wav_playhead_sec > duration:
                    self.wav_playhead_sec = 0.0
                    self._start_wav_audio_playback(seek_t=0.0)

            duration = self.wav_spec_cache["duration"]
            t_curr = min(duration, max(0.0, self.wav_playhead_sec))
            self.wav_time_slider.set(t_curr)
            self.wav_time_label.config(text=f"{t_curr:.2f} / {duration:.2f} s")
            self.cursor_wav.set_xdata([t_curr, t_curr])
            self._update_wav_slice_view()
            self.canvas_wav.draw_idle()

        self.after(40, self._gui_update_loop)

    def _on_params_changed(self):
        if self.is_streaming or self.is_file_playing:
            return

        mode = self.input_mode.get()
        N = self.block_size.get()
        Fs = self.sample_rate.get()

        if mode == "generated":
            t = np.arange(N) / Fs
            sig = np.full(N, self.dc_offset.get())
            w_type = self.waveform_type.get()

            if w_type == "Sine Multi-Tone":
                for t_cfg in self.tones_config:
                    if t_cfg["enable"].get():
                        sig += t_cfg["amp"].get() * np.sin(2.0 * np.pi * t_cfg["freq"].get() * t)
            elif w_type == "Square Wave":
                f = self.tones_config[0]["freq"].get()
                a = self.tones_config[0]["amp"].get()
                sig += a * np.sign(np.sin(2.0 * np.pi * f * t))
            elif w_type == "Sawtooth":
                f = self.tones_config[0]["freq"].get()
                a = self.tones_config[0]["amp"].get()
                sig += a * (2.0 * (f * t - np.floor(0.5 + f * t)))
            elif w_type == "Chirp Sweep":
                a = self.tones_config[0]["amp"].get()
                f0, f1 = self.grid_min_hz.get(), self.grid_max_hz.get()
                t_max = N / Fs
                sig += a * np.sin(2.0 * np.pi * (f0 + (f1 - f0) * t / (2 * t_max)) * t)

            if self.noise_std.get() > 0:
                sig += np.random.normal(0, self.noise_std.get(), N)

            samples = np.clip(np.round(sig), -32768, 32767).astype(np.int16)
            self._run_analysis_on_samples(samples)

        elif mode == "file":
            if self.audio_samples is not None and len(self.audio_samples) >= N:
                samples = self.audio_samples[:N]
                self._run_analysis_on_samples(samples)

    def _run_analysis_on_samples(self, samples):
        N = len(samples)
        try:
            Fs = float(self.sample_rate.get())
        except Exception:
            Fs = 8000.0

        grid_mode = self.grid_type.get()

        try:
            pts = max(2, int(self.grid_points.get()))
        except Exception:
            pts = 129

        try:
            raw_max_t = int(self.max_tones.get())
            max_t = max(1, min(16, raw_max_t))
        except Exception:
            max_t = 5

        try:
            min_amp = max(0, int(self.min_amplitude.get()))
        except Exception:
            min_amp = 300

        c_samples = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        c_tones = (IntSpectrumTone * 16)()
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
            self.status_label.config(text="Analysis failed: parameters out of bounds.", foreground="#e06c75")
            return

        tone_count = min(c_tone_count.value, max_t, 16)
        residual_arr = np.ctypeslib.as_array(c_residual)
        res_rms = self.api.lib.int_spectrum_residual_rms_q8(N, c_residual) / 256.0

        # Reconstruct fitted tone signal
        denoised = (ctypes.c_int16 * N)()
        self.api.lib.int_spectrum_reconstruct_denoised(N, c_samples, c_residual, denoised)
        denoised_samples = np.ctypeslib.as_array(denoised)

        # Run multi-estimator fusion diagnostic on top tone
        fusion_res = IntSpectrumFusionResult()
        has_fusion = False
        if tone_count > 0:
            top_coarse_bin = int(round(c_tones[0].frequencyQ16 / 65536.0))
            if top_coarse_bin > 0 and top_coarse_bin < N // 2:
                has_fusion = bool(self.api.lib.int_spectrum_fuse_estimates(N, c_samples, top_coarse_bin, ctypes.byref(fusion_res)))

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

        self.status_label.config(
            text=f"Time: {t_elapsed:.2f}ms | DC: {c_dc.value} | Tones: {tone_count} | Res RMS: {res_rms:.2f} samples",
            foreground="#98c379"
        )

        # Save state for rendering
        self.last_input_samples = samples
        self.last_model_samples = denoised_samples
        self.last_residual_samples = residual_arr / 256.0
        self.last_detected_tones = detected_tones_list
        self.last_fusion_result = fusion_res if has_fusion else None

        # Update Waterfall history
        fft_mags = np.abs(np.fft.rfft(samples)) * (2.0 / N)
        self.waterfall_history.append(fft_mags)
        if len(self.waterfall_history) > self.max_waterfall_history:
            self.waterfall_history.pop(0)

        # Render active tab plots
        self._render_active_tab(Fs, N)

    def _render_active_tab(self, Fs, N):
        curr_tab = self.plot_notebook.index(self.plot_notebook.select())

        if curr_tab == 0:
            # Overview Tab
            time_axis = np.arange(N) / Fs * 1000.0

            # Update time domain lines without clearing to prevent UI flashing
            self.line_time_in.set_data(time_axis, self.last_input_samples)
            self.line_time_mod.set_data(time_axis, self.last_model_samples)
            self.line_time_res.set_data(time_axis, self.last_residual_samples)
            self.ax_time.set_xlim(0, np.max(time_axis))
            self.ax_time.set_ylim(-35000, 35000)

            # Update frequency domain data
            fft_mags = np.abs(np.fft.rfft(self.last_input_samples)) * (2.0 / N)
            fft_freqs = np.fft.rfftfreq(N, d=1.0/Fs)
            self.line_fft.set_data(fft_freqs, fft_mags)

            # Update detected tone stems and markers
            if self.show_peak_markers.get() and self.last_detected_tones:
                stem_xs = []
                stem_ys = []
                det_freqs = []
                det_amps = []
                for hz, amp in self.last_detected_tones:
                    stem_xs.extend([hz, hz, np.nan])
                    stem_ys.extend([0, amp, np.nan])
                    det_freqs.append(hz)
                    det_amps.append(amp)
                self.line_stem.set_data(stem_xs, stem_ys)
                self.pts_stem.set_data(det_freqs, det_amps)
            else:
                self.line_stem.set_data([], [])
                self.pts_stem.set_data([], [])

            # Update text annotations
            for ann in self.spectrum_annotations:
                ann.remove()
            self.spectrum_annotations.clear()

            if self.show_peak_markers.get() and self.show_peak_labels.get() and self.last_detected_tones:
                for hz, amp in self.last_detected_tones:
                    ann = self.ax_freq.annotate(
                        f"{hz:.1f}Hz", xy=(hz, amp), xytext=(0, 5), textcoords="offset points",
                        fontsize=8, color="#e5c07b", ha="center"
                    )
                    self.spectrum_annotations.append(ann)

            # Only adjust xlim when sampling rate changes or scale mode toggles, NOT on continuous frames!
            if self.last_plotted_fs != Fs:
                if self.freq_scale_mode.get() == "linear":
                    self.ax_freq.set_xlim(0, Fs / 2.0)
                else:
                    self.ax_freq.set_xlim(20.0, Fs / 2.0)
                self.last_plotted_fs = Fs

            self.canvas_overview.draw_idle()

        elif curr_tab == 1:
            # Waterfall & Residual Tab
            self.ax_waterfall.clear()
            if self.waterfall_history:
                wf_mat = np.array(self.waterfall_history)
                self.ax_waterfall.imshow(wf_mat, aspect='auto', origin='lower',
                                         extent=[0, Fs/2, 0, len(wf_mat)], cmap=self.waterfall_cmap.get())
            self.ax_waterfall.set_title("Spectral Waterfall / Spectrogram History", color="#abb2bf")
            self.ax_waterfall.set_xlabel("Frequency (Hz)", color="#abb2bf")
            self.ax_waterfall.set_ylabel("Block Frame", color="#abb2bf")
            self.ax_waterfall.tick_params(colors='#abb2bf')

            self.ax_hist.clear()
            if self.last_residual_samples is not None:
                self.ax_hist.hist(self.last_residual_samples, bins=30, color="#98c379", alpha=0.7, edgecolor="#1e222a")
            self.ax_hist.set_title("Residual Error Amplitude Histogram", color="#abb2bf")
            self.ax_hist.set_xlabel("Residual Error (samples)", color="#abb2bf")
            self.ax_hist.set_ylabel("Count", color="#abb2bf")
            self.ax_hist.tick_params(colors='#abb2bf')
            self.ax_hist.grid(True, linestyle=":", alpha=0.4)

            self.fig_waterfall.tight_layout(pad=1.5)
            self.canvas_waterfall.draw()

        elif curr_tab == 2:
            # WAV Spectrogram & Playback Tab
            self._update_wav_slice_view()
            self.canvas_wav.draw_idle()

        elif curr_tab == 3:
            # Estimator Fusion & Harmonic Analysis Tab
            self.fusion_text.delete("1.0", tk.END)
            report_str = f"===============================================================\n"
            report_str += f"         MULTI-ESTIMATOR CONSENSUS & HARMONIC ANALYSIS         \n"
            report_str += f"===============================================================\n\n"

            if self.last_fusion_result is not None:
                fr = self.last_fusion_result
                estimator_names = {0: "Half-Block Phase", 1: "First-Difference", 2: "Lag Recurrence", 3: "Parabolic Spectrum"}
                selected_name = estimator_names.get(fr.selectedEstimator, f"Estimator #{fr.selectedEstimator}")

                report_str += f"Target Coarse Bin       : {fr.coarseBin} (~{fr.coarseBin * Fs / N:.2f} Hz)\n"
                report_str += f"Fused Frequency (Q16)   : {fr.frequencyQ16} (~{fr.frequencyQ16 * Fs / (N * 65536.0):.3f} Hz)\n"
                report_str += f"Refined Offset (Q16)    : {fr.offsetQ16} ({fr.offsetQ16 / 65536.0:+.4f} bins)\n"
                report_str += f"Half-Block Phase Offset : {fr.phaseOffsetQ16} ({fr.phaseOffsetQ16 / 65536.0:+.4f} bins)\n"
                report_str += f"Fitted Amplitude        : {fr.peakAmplitude}\n"
                report_str += f"Detector Strength       : {fr.strength}\n\n"

                report_str += f"--- Consensus Metrics ---\n"
                report_str += f"Selected Winner         : {selected_name}\n"
                report_str += f"Consensus Confidence    : {fr.consensusConfidenceQ15 / 32767.0 * 100.0:.1f}% (Q15: {fr.consensusConfidenceQ15})\n"
                report_str += f"Consensus Spread (Q16)  : {fr.consensusSpreadQ16} ({fr.consensusSpreadQ16 / 65536.0:.4f} bins)\n"
                report_str += f"Valid Estimator Mask    : 0x{fr.estimatorValidMask:02X}\n"
                report_str += f"Inlier Estimator Mask   : 0x{fr.estimatorInlierMask:02X}\n\n"

                report_str += f"--- Active Estimators Summary ---\n"
                for code, name in estimator_names.items():
                    valid = bool(fr.estimatorValidMask & (1 << code))
                    inlier = bool(fr.estimatorInlierMask & (1 << code))
                    status = "INLIER" if inlier else ("VALID" if valid else "DISABLED")
                    report_str += f"  - [{status:<8}] {name}\n"
            else:
                report_str += "No tone active for consensus fusion diagnostics.\n\n"

            # Harmonic Analysis Section
            report_str += f"\n===============================================================\n"
            report_str += f"                     HARMONIC SPECTRUM SUMMARY                 \n"
            report_str += f"===============================================================\n"

            if self.last_detected_tones:
                f0_hz, f0_amp = self.last_detected_tones[0]
                report_str += f"Fundamental F0 Candidate: {f0_hz:.2f} Hz (Peak Amp: {f0_amp})\n"
                harm_amps = []
                for idx, (hz, amp) in enumerate(self.last_detected_tones):
                    ratio = hz / f0_hz if f0_hz > 0 else 0
                    report_str += f"  Tone #{idx+1}: {hz:>8.2f} Hz | Amp: {amp:>7} | Ratio to F0: {ratio:.2f}x\n"
                    if idx > 0:
                        harm_amps.append(amp)

                if harm_amps and f0_amp > 0:
                    thd_est = (math.sqrt(sum(a**2 for a in harm_amps)) / f0_amp) * 100.0
                    report_str += f"\nEstimated Total Harmonic Ratio (THD): {thd_est:.2f}%\n"
            else:
                report_str += "No tones extracted in current frame.\n"

            self.fusion_text.insert(tk.END, report_str)

    def _export_csv(self):
        if not self.last_detected_tones:
            messagebox.showwarning("Export Empty", "No detected tones to export.")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Index", "Frequency_Hz", "Peak_Amplitude"])
                    for i, (hz, amp) in enumerate(self.last_detected_tones):
                        writer.writerow([i+1, hz, amp])
                messagebox.showinfo("Export Success", f"Exported results to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save CSV: {e}")

    def _export_png(self):
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG images", "*.png")])
        if filename:
            try:
                curr_tab = self.plot_notebook.index(self.plot_notebook.select())
                if curr_tab == 0:
                    self.fig_overview.savefig(filename, dpi=200)
                elif curr_tab == 1:
                    self.fig_waterfall.savefig(filename, dpi=200)
                elif curr_tab == 2:
                    self.fig_wav.savefig(filename, dpi=200)
                messagebox.showinfo("Export Success", f"Saved plot to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save image: {e}")


if __name__ == "__main__":
    app = IntSpectrumGUI()
    app.mainloop()
