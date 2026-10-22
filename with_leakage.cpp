#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

// Integer DFT-bin detector bank.
// Fixed memory; no allocation in the analysis loop.
template <std::size_t N, std::size_t DETECTORS,
          unsigned LUT_BITS = 10>
class BresenhamSpectrum {
    static_assert(N >= 2 && N <= 32768,
                  "N must be in [2, 32768]");
    static_assert(DETECTORS >= 1,
                  "At least one detector is required");
    static_assert(LUT_BITS >= 8 && LUT_BITS <= 12,
                  "LUT_BITS must be in [8, 12]");

    static constexpr std::size_t LUT_SIZE =
        std::size_t{1} << LUT_BITS;

    static constexpr std::uint32_t PHASE_INDEX_SHIFT =
        32U - LUT_BITS;

    static constexpr std::int64_t Q15_ONE = 32767;

public:
    struct Result {
        std::uint32_t bin = 0;
        std::uint32_t peak_amplitude = 0;
        std::uint64_t strength = 0;
    };

    using InputBlock = std::array<std::int16_t, N>;
    using Results = std::array<Result, DETECTORS>;

    // Call during initialization, outside the real-time loop.
    static void prepare() {
        (void)sineTable();
    }

    // Select DETECTORS bins evenly across the inclusive range.
    bool analyze(const InputBlock& samples,
                 std::uint32_t firstBin,
                 std::uint32_t lastBin,
                 Results& out) const {
        constexpr std::uint32_t NYQUIST =
            static_cast<std::uint32_t>(N / 2);

        if (firstBin > lastBin || lastBin > NYQUIST)
            return false;

        if constexpr (DETECTORS > 1) {
            if (lastBin - firstBin < DETECTORS - 1)
                return false; // Avoid duplicate bin locations.
        }

        const auto& lut = sineTable();

        std::uint32_t bin = firstBin;
        std::uint32_t binStep = 0;
        std::uint32_t binRemainder = 0;
        std::uint64_t binError = 0;

        if constexpr (DETECTORS > 1) {
            const std::uint32_t intervals =
                static_cast<std::uint32_t>(DETECTORS - 1);
            const std::uint32_t span = lastBin - firstBin;

            binStep = span / intervals;
            binRemainder = span % intervals;
        } else {
            bin = firstBin + (lastBin - firstBin) / 2;
        }

        for (std::size_t d = 0; d < DETECTORS; ++d) {
            out[d] = analyzeBin(samples, bin, lut);

            if constexpr (DETECTORS > 1) {
                if (d + 1 < DETECTORS) {
                    const std::uint32_t intervals =
                        static_cast<std::uint32_t>(DETECTORS - 1);

                    bin += binStep;
                    binError += binRemainder;

                    if (binError >= intervals) {
                        ++bin;
                        binError -= intervals;
                    }
                }
            }
        }

        return true;
    }

private:
    // Q15 sine table, generated once.
    // No floating-point operations in the sample loop.
    static const std::array<std::int16_t, LUT_SIZE>&
    sineTable() {
        static const std::array<std::int16_t, LUT_SIZE> table = [] {
            std::array<std::int16_t, LUT_SIZE> t{};
            constexpr double PI =
                3.1415926535897932384626433832795;

            for (std::size_t i = 0; i < LUT_SIZE; ++i) {
                const double angle =
                    2.0 * PI * static_cast<double>(i) / LUT_SIZE;

                t[i] = static_cast<std::int16_t>(
                    std::lround(std::sin(angle) * 32767.0));
            }

            return t;
        }();

        return table;
    }

    static std::int64_t divideRounded(
        std::int64_t x, std::int64_t divisor) {
        if (x >= 0)
            return (x + divisor / 2) / divisor;

        return -((-x + divisor / 2) / divisor);
    }

    // Bounded integer square root: at most 32 iterations.
    static std::uint64_t integerSqrt(std::uint64_t x) {
        std::uint64_t root = 0;
        std::uint64_t bit = std::uint64_t{1} << 62;

        while (bit > x)
            bit >>= 2;

        while (bit != 0) {
            if (x >= root + bit) {
                x -= root + bit;
                root = (root >> 1) + bit;
            } else {
                root >>= 1;
            }

            bit >>= 2;
        }

        return root;
    }

    static Result analyzeBin(
        const InputBlock& samples,
        std::uint32_t bin,
        const std::array<std::int16_t, LUT_SIZE>& lut) {

        // One full turn = 2^32 phase units.
        // Therefore pi = 2^31 and pi/2 = 2^30.
        //
        // The rational phase increment is:
        // bin * 2^32 / N.
        const std::uint64_t numerator =
            static_cast<std::uint64_t>(bin) *
            (std::uint64_t{1} << 32);

        const std::uint64_t stepQ32 = numerator / N;
        const std::uint64_t remainder = numerator % N;

        std::uint64_t phaseError = 0;
        std::uint32_t phase = 0;

        std::int64_t realAccQ15 = 0;
        std::int64_t imagAccQ15 = 0;

        constexpr std::size_t MASK = LUT_SIZE - 1;
        constexpr std::size_t QUARTER_TURN = LUT_SIZE / 4;

        for (std::size_t n = 0; n < N; ++n) {
            // High phase bits address the lookup table.
            const std::uint32_t index =
                phase >> PHASE_INDEX_SHIFT;

            const std::int32_t sinQ15 = lut[index];
            const std::int32_t cosQ15 =
                lut[(index + QUARTER_TURN) & MASK];

            const std::int64_t sample = samples[n];

            // Integer quadrature correlations.
            realAccQ15 += sample * cosQ15;
            imagAccQ15 += sample * sinQ15;

            // Bresenham phase step.
            // uint32_t wraparound is intentional.
            phase += static_cast<std::uint32_t>(stepQ32);

            phaseError += remainder;
            if (phaseError >= N) {
                ++phase;
                phaseError -= N;
            }
        }

        // Convert accumulated Q15 projections to sample units.
        const std::int64_t real =
            divideRounded(realAccQ15, Q15_ONE);
        const std::int64_t imag =
            divideRounded(imagAccQ15, Q15_ONE);

        const std::uint64_t magnitudeSquared =
            static_cast<std::uint64_t>(real * real) +
            static_cast<std::uint64_t>(imag * imag);

        const std::uint64_t magnitude =
            integerSqrt(magnitudeSquared);

        // Interior real-DFT bins use 2/N amplitude scaling.
        // DC and Nyquist use 1/N.
        const bool endpoint =
            (bin == 0) ||
            ((N % 2 == 0) && (bin == N / 2));

        const std::uint64_t scale = endpoint ? 1U : 2U;
        const std::uint64_t strength = scale * magnitude;

        const auto amplitude = static_cast<std::uint32_t>(
            (strength + N / 2) / N);

        return Result{bin, amplitude, strength};
    }
};

int main() {
    constexpr std::size_t N = 256;
    constexpr std::size_t DETECTORS = 11;
    constexpr double SAMPLE_RATE_HZ = 8000.0;
    constexpr std::uint32_t SIGNAL_BIN = 14;

    using Analyzer = BresenhamSpectrum<N, DETECTORS>;

    Analyzer::prepare();

    Analyzer::InputBlock samples{};

    // Synthetic test signal only; replace with ADC samples on hardware.
    constexpr double PI =
        3.1415926535897932384626433832795;

    for (std::size_t n = 0; n < N; ++n) {
        const double angle =
            2.0 * PI * SIGNAL_BIN * static_cast<double>(n) / N;

        samples[n] = static_cast<std::int16_t>(
            std::lround(12000.0 * std::sin(angle)));
    }

    Analyzer analyzer;
    Analyzer::Results results{};

    if (!analyzer.analyze(samples, 10, 20, results)) {
        std::cerr << "Invalid detector range\n";
        return 1;
    }

    const auto best = std::max_element(
        results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.strength < b.strength;
        });

    std::cout << "Strongest bin: " << best->bin
              << " (~"
              << (best->bin * SAMPLE_RATE_HZ / N)
              << " Hz), estimated peak amplitude: "
              << best->peak_amplitude << "\n\n";

    std::cout << "bin\tfrequency_Hz\tamplitude\tstrength\n";

    for (const auto& r : results) {
        std::cout << r.bin << '\t'
                  << (r.bin * SAMPLE_RATE_HZ / N) << '\t'
                  << r.peak_amplitude << '\t'
                  << r.strength << '\n';
    }
}
