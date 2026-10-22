#include "int_spectrum/bresenham_spectrum.hpp"
#include "int_spectrum/multi_tone_comb.hpp"
#include "int_spectrum/c_api.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>

using namespace int_spectrum;

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (cond) { \
            g_tests_passed++; \
        } else { \
            std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            g_tests_failed++; \
        } \
    } while(0)

void test_single_tone_bresenham() {
    constexpr size_t N = 256;
    constexpr size_t DETECTORS = 11;
    constexpr double Fs = 8000.0;
    constexpr double target_bin = 14.25;
    constexpr double target_amp = 10000.0;

    using Analyzer = BresenhamSpectrum<N, DETECTORS>;
    Analyzer::prepare();

    Analyzer::InputBlock samples{};
    for (size_t n = 0; n < N; ++n) {
        double angle = 2.0 * M_PI * target_bin * static_cast<double>(n) / N;
        samples[n] = static_cast<int16_t>(std::lround(target_amp * std::sin(angle)));
    }

    Analyzer analyzer;
    Analyzer::Results results{};
    Analyzer::DifferentialResult refined{};

    bool ok = analyzer.analyzeWithDifferentialRefinement(samples, 10, 20, results, refined);
    TEST_ASSERT(ok, "Single tone analysis returned true");

    double detected_bin = static_cast<double>(refined.frequency_q16) / 65536.0;
    double bin_error = std::abs(detected_bin - target_bin);
    TEST_ASSERT(bin_error < 0.05, "Single tone frequency accuracy within 0.05 bin");

    double amp_error = std::abs(static_cast<double>(refined.peak_amplitude) - target_amp) / target_amp;
    TEST_ASSERT(amp_error < 0.05, "Single tone amplitude accuracy within 5%");
}

void test_multi_tone_comb() {
    constexpr size_t N = 512;
    constexpr size_t MAX_TONES = 5;
    constexpr double Fs = 8000.0;
    constexpr size_t GRID_POINTS = 129;

    using Analyzer = MultiToneComb<N, MAX_TONES, 12>;
    Analyzer::prepare();

    std::array<uint32_t, GRID_POINTS> grid{};
    bool grid_ok = Analyzer::makeLogGridHz(100.0, 700.0, Fs, grid);
    TEST_ASSERT(grid_ok, "Log grid generation succeeded");

    Analyzer::InputBlock samples{};
    struct ToneDef { double bin; double amp; };
    ToneDef input_tones[3] = {{14.25, 12000.0}, {27.65, 6500.0}, {33.10, 3500.0}};

    double sample_sum = 0;
    for (size_t n = 0; n < N; ++n) {
        double val = 500.0; // DC offset
        for (const auto& t : input_tones) {
            val += t.amp * std::sin(2.0 * M_PI * t.bin * static_cast<double>(n) / N);
        }
        samples[n] = static_cast<int16_t>(std::lround(val));
        sample_sum += samples[n];
    }
    int expected_dc = std::lround(sample_sum / N);

    Analyzer analyzer;
    Analyzer::ToneList tones{};
    Analyzer::ResidualBlock residual{};
    size_t toneCount = 0;
    int32_t removedDc = 0;

    bool decomp_ok = analyzer.decomposeGrid(samples, grid, 500, tones, toneCount, residual, removedDc);
    TEST_ASSERT(decomp_ok, "Multi-tone grid decomposition succeeded");
    TEST_ASSERT(toneCount == 3, "Detected exactly 3 tones");
    TEST_ASSERT(std::abs(removedDc - expected_dc) <= 2, "Removed DC correctly matches sample mean");

    double res_rms = static_cast<double>(Analyzer::residualRmsQ8(residual)) / 256.0;
    TEST_ASSERT(res_rms < 5.0, "Residual RMS is very low (< 5.0 samples)");
}

void test_c_api() {
    constexpr size_t N = 512;
    constexpr double Fs = 8000.0;
    constexpr size_t GRID_POINTS = 129;

    std::vector<uint32_t> grid(GRID_POINTS);
    int grid_ok = int_spectrum_make_log_grid(N, 100.0, 700.0, Fs, grid.data(), GRID_POINTS);
    TEST_ASSERT(grid_ok == 1, "C API make_log_grid returned 1");

    std::vector<int16_t> samples(N);
    for (size_t n = 0; n < N; ++n) {
        double val = 8000.0 * std::sin(2.0 * M_PI * 15.5 * static_cast<double>(n) / N);
        samples[n] = static_cast<int16_t>(std::lround(val));
    }

    IntSpectrumTone tones[5];
    size_t toneCount = 0;
    std::vector<int64_t> residual(N);
    int32_t dc = 0;

    int res = int_spectrum_decompose_grid(N, samples.data(), grid.data(), GRID_POINTS, 500, 5, tones, &toneCount, residual.data(), &dc, NULL, NULL);
    TEST_ASSERT(res == 1, "C API decompose_grid succeeded");
    TEST_ASSERT(toneCount >= 1, "C API detected at least 1 tone");

    double hz = int_spectrum_bin_q16_to_hz(N, tones[0].frequencyQ16, Fs);
    double expected_hz = 15.5 * Fs / N;
    TEST_ASSERT(std::abs(hz - expected_hz) < 2.0, "C API detected frequency matches ground truth");
}

int main() {
    std::cout << "=== Running C++ IntSpectrum Unit Tests ===\n";
    test_single_tone_bresenham();
    test_multi_tone_comb();
    test_c_api();

    std::cout << "Results: " << g_tests_passed << " passed, " << g_tests_failed << " failed.\n";
    return g_tests_failed == 0 ? 0 : 1;
}
