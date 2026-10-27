#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
#define HAS_RDTSC 1
#else
#define HAS_RDTSC 0
#endif

#include "int_spectrum/c_api.h"

// Inline CPU cycle counter
static inline uint64_t read_cycles() {
#if HAS_RDTSC
    return __rdtsc();
#else
    auto now = std::chrono::high_resolution_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
#endif
}

// Simple floating-point Cooley-Tukey FFT implementation for cycle-count baseline comparison
static void radix2_fft(std::vector<double>& real, std::vector<double>& imag) {
    size_t n = real.size();
    if (n <= 1) return;

    // Bit reversal permutation
    size_t j = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
        size_t k = n >> 1;
        while (k >= 1 && j >= k) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    // Cooley-Tukey computation
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / static_cast<double>(len);
        double wlen_r = std::cos(ang);
        double wlen_i = std::sin(ang);
        for (size_t i = 0; i < n; i += len) {
            double wr = 1.0;
            double wi = 0.0;
            for (size_t k = 0; k < len / 2; ++k) {
                size_t u = i + k;
                size_t v = i + k + len / 2;
                double tr = wr * real[v] - wi * imag[v];
                double ti = wr * imag[v] + wi * real[v];
                real[v] = real[u] - tr;
                imag[v] = imag[u] - ti;
                real[u] += tr;
                imag[u] += ti;
                double next_wr = wr * wlen_r - wi * wlen_i;
                double next_wi = wr * wlen_i + wi * wlen_r;
                wr = next_wr;
                wi = next_wi;
            }
        }
    }
}

void benchmark_block_size(size_t N) {
    constexpr double Fs = 8000.0;
    std::vector<int16_t> samples(N);
    for (size_t i = 0; i < N; ++i) {
        double t = static_cast<double>(i) / Fs;
        samples[i] = static_cast<int16_t>(10000.0 * std::cos(2.0 * M_PI * 234.5 * t));
    }

    // Prepare IntSpectrum Grid (129 points logarithmic)
    size_t grid_size = (N < 129) ? N : 129;
    std::vector<uint32_t> gridQ16(grid_size);
    int_spectrum_make_log_grid(N, 100.0, 1000.0, Fs, gridQ16.data(), grid_size);

    IntSpectrumTone tones[5];
    size_t toneCount = 0;
    std::vector<int64_t> residualQ8(N);
    int32_t removedDc = 0;

    // Warmup
    for (int w = 0; w < 20; ++w) {
        int_spectrum_decompose_grid(
            N, samples.data(), gridQ16.data(), grid_size, 300, 3, tones, &toneCount, residualQ8.data(), &removedDc, NULL, NULL
        );
    }

    // Measure IntSpectrum Grid CPU Cycles
    constexpr int ITERS = 500;
    uint64_t start_cycles = read_cycles();
    for (int i = 0; i < ITERS; ++i) {
        int_spectrum_decompose_grid(
            N, samples.data(), gridQ16.data(), grid_size, 300, 3, tones, &toneCount, residualQ8.data(), &removedDc, NULL, NULL
        );
    }
    uint64_t end_cycles = read_cycles();
    uint64_t int_cycles_per_block = (end_cycles - start_cycles) / ITERS;
    double int_cycles_per_sample = static_cast<double>(int_cycles_per_block) / static_cast<double>(N);

    // Prepare FFT Float buffers
    std::vector<double> fft_real(N);
    std::vector<double> fft_imag(N, 0.0);
    for (size_t i = 0; i < N; ++i) fft_real[i] = samples[i];

    // Warmup FFT
    for (int w = 0; w < 20; ++w) {
        radix2_fft(fft_real, fft_imag);
    }

    // Measure Float Radix-2 FFT Cycles
    start_cycles = read_cycles();
    for (int i = 0; i < ITERS; ++i) {
        for (size_t k = 0; k < N; ++k) { fft_real[k] = samples[k]; fft_imag[k] = 0.0; }
        radix2_fft(fft_real, fft_imag);
    }
    end_cycles = read_cycles();
    uint64_t fft_cycles_per_block = (end_cycles - start_cycles) / ITERS;
    double fft_cycles_per_sample = static_cast<double>(fft_cycles_per_block) / static_cast<double>(N);

    std::cout << std::setw(12) << N
              << std::setw(22) << int_cycles_per_block
              << std::setw(20) << std::fixed << std::setprecision(1) << int_cycles_per_sample
              << std::setw(22) << fft_cycles_per_block
              << std::setw(20) << fft_cycles_per_sample
              << "\n";
}

int main() {
    std::cout << "=========================================================================================\n";
    std::cout << "           NATIVE C++ CPU CLOCK CYCLE BENCHMARK (RDTSC / CPU CYCLES)\n";
    std::cout << "=========================================================================================\n";
    std::cout << "Units: Measured CPU cycles per execution block and per input sample.\n\n";
    std::cout << std::setw(12) << "Block Size N"
              << std::setw(22) << "IntSpectrum (Cycles)"
              << std::setw(20) << "IntSpectrum (Cycles/Smp)"
              << std::setw(22) << "Radix-2 FFT (Cycles)"
              << std::setw(20) << "Radix-2 FFT (Cycles/Smp)"
              << "\n";
    std::cout << "-----------------------------------------------------------------------------------------\n";

    benchmark_block_size(64);
    benchmark_block_size(128);
    benchmark_block_size(256);
    benchmark_block_size(512);
    benchmark_block_size(1024);
    benchmark_block_size(2048);
    benchmark_block_size(4096);

    std::cout << "=========================================================================================\n";
    return 0;
}
