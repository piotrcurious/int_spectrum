#include "int_spectrum/multi_tone_comb.hpp"
#include <iostream>

using namespace int_spectrum;

int main() {
    constexpr std::size_t N = 512;
    constexpr std::size_t MAX_TONES = 5;
    constexpr double SAMPLE_RATE_HZ = 8000.0;
    constexpr std::size_t GRID_POINTS = 129;
    constexpr double GRID_MIN_HZ = 100.0;
    constexpr double GRID_MAX_HZ = 700.0;
    constexpr double PI = 3.1415926535897932384626433832795;

    struct TestTone { double bin; double amplitude; double phase; };
    constexpr std::array<TestTone, 3> inputTones{{
        {14.25, 12000.0, 0.20},
        {27.65,  6500.0, 1.10},
        {33.10,  3500.0, 2.00}
    }};

    using Analyzer = MultiToneComb<N, MAX_TONES, 12>;
    Analyzer::prepare();
    std::array<std::uint32_t, GRID_POINTS> frequencyGridQ16{};
    if (!Analyzer::makeLogGridHz(GRID_MIN_HZ, GRID_MAX_HZ,
                                 SAMPLE_RATE_HZ, frequencyGridQ16)) {
        std::cerr << "Could not create strictly increasing logarithmic frequency grid.\n";
        return 1;
    }
    Analyzer::InputBlock samples{};
    for (std::size_t n = 0; n < N; ++n) {
        double x = 1800.0; // deliberate DC offset
        for (const auto& tone : inputTones) {
            const double angle = 2.0 * PI * tone.bin * static_cast<double>(n) / N + tone.phase;
            x += tone.amplitude * std::sin(angle);
        }
        const long rounded = std::lround(x);
        samples[n] = static_cast<std::int16_t>(
            std::max(-32768L, std::min(32767L, rounded)));
    }

    Analyzer analyzer;
    Analyzer::ToneList tones{};
    Analyzer::ResidualBlock residual{};
    std::size_t toneCount = 0;
    std::int32_t removedDc = 0;
    constexpr std::uint32_t MINIMUM_AMPLITUDE = 500;
    if (!analyzer.decomposeGrid(samples, frequencyGridQ16,
                                MINIMUM_AMPLITUDE, tones, toneCount,
                                residual, removedDc)) {
        std::cerr << "Invalid frequency range or decomposition parameters.\n";
        return 1;
    }

    const std::uint64_t rmsQ8 = Analyzer::residualRmsQ8(residual);
    std::cout << "Log-frequency detector grid / joint multi-tone least-squares decomposition\n";
    std::cout << "N=" << N << ", Fs=" << SAMPLE_RATE_HZ
              << " Hz, log grid=" << GRID_MIN_HZ << ".." << GRID_MAX_HZ
              << " Hz (" << GRID_POINTS << " points), removed DC="
              << removedDc << " samples\n";
    std::cout << "Detected " << toneCount << " tones; residual RMS ~"
              << (static_cast<double>(rmsQ8) / 256.0) << " samples\n\n";
    std::cout << "#\tbank\tfrequency_bin\tfrequency_Hz\tpeak_amplitude\n";
    for (std::size_t i = 0; i < toneCount; ++i) {
        const auto& t = tones[i];
        const double bin = static_cast<double>(t.frequencyQ16) / 65536.0;
        const char* bankName = (t.discoveryBank == 2) ? "custom-grid" :
                               ((t.discoveryBank == 0) ? "integer" : "half-bin");
        std::cout << i << '\t' << bankName
                  << '\t' << bin << '\t' << (bin * SAMPLE_RATE_HZ / N)
                  << '\t' << t.peakAmplitude << '\n';
    }

    return 0;
}
