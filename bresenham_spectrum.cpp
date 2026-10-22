#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

// Fixed-size, allocation-free detector bank for an N-sample block.
// Differential phase refinement is intended for an isolated dominant tone
// whose nearest integer DFT bin is known (or has been found by a dense scan).
template <std::size_t N, std::size_t DETECTORS, unsigned LUT_BITS = 10>
class BresenhamSpectrum {
    static_assert(N >= 4, "N must be at least 4");
    static_assert(N <= 32768, "N is bounded to keep signed correlation products safe");
    static_assert(DETECTORS >= 1, "At least one detector is required");
    static_assert(LUT_BITS >= 8 && LUT_BITS <= 12, "LUT_BITS should be in [8, 12]");

    static constexpr std::size_t LUT_SIZE = std::size_t{1} << LUT_BITS;
    static constexpr std::uint32_t PHASE_INDEX_SHIFT = 32U - LUT_BITS;
    static constexpr std::int64_t Q15_ONE = 32767;
    static constexpr std::int64_t PI_Q30 = 3373259426LL;

public:
    struct Result {
        std::uint32_t bin = 0;
        std::uint32_t peak_amplitude = 0;
        std::uint64_t strength = 0;
    };

    // offset_q16 is the estimated frequency correction in DFT-bin units:
    // +16384 means +0.25 bin; frequency_q16 is absolute bin position in Q16.
    struct DifferentialResult {
        std::uint32_t coarse_bin = 0;
        std::int32_t offset_q16 = 0;       // final, optionally de-biased offset
        std::int32_t phase_offset_q16 = 0; // raw half-block differential-phase estimate
        std::uint32_t frequency_q16 = 0;
        std::uint32_t peak_amplitude = 0;
        std::uint64_t strength = 0;
        std::int64_t phase_difference_q30 = 0;
    };

    using InputBlock = std::array<std::int16_t, N>;
    using Results = std::array<Result, DETECTORS>;

    // Call once during setup if first-use table initialization must not occur
    // in a time-critical path. Floating-point math is used only here.
    static void prepare() { (void)sineTable(); }

    // Analyze inclusive integer DFT-bin range [firstBin, lastBin]. Detector
    // locations are distributed with Bresenham; gaps differ by at most one.
    bool analyze(const InputBlock& samples,
                 std::uint32_t firstBin,
                 std::uint32_t lastBin,
                 Results& out) const {
        constexpr std::uint32_t NYQUIST = static_cast<std::uint32_t>(N / 2);
        if (firstBin > lastBin || lastBin > NYQUIST) return false;
        if constexpr (DETECTORS > 1) {
            if ((lastBin - firstBin) < (DETECTORS - 1)) return false;
        }

        const auto& lut = sineTable();
        std::uint32_t bin = firstBin;
        std::uint64_t binError = 0;
        std::uint32_t binStep = 0;
        std::uint32_t binRemainder = 0;
        if constexpr (DETECTORS > 1) {
            const std::uint32_t intervals = static_cast<std::uint32_t>(DETECTORS - 1);
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
                    const std::uint32_t intervals = static_cast<std::uint32_t>(DETECTORS - 1);
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

    // Refine one dominant sinusoid using differential phase between the two
    // halves of the block. For a tone at (k + delta) bins, heterodyning by k
    // makes the second-half phasor rotate by approximately pi*delta radians.
    // Therefore delta = phase(C_second * conj(C_first)) / pi.
    //
    // Preconditions: even N; coarseBin is an interior bin and is the nearest
    // integer bin to the dominant tone (normally |delta| <= 0.5). This is a
    // single-tone estimator, not a universal multi-tone leakage remover.
    bool refineSingleTone(const InputBlock& samples,
                          std::uint32_t coarseBin,
                          DifferentialResult& out) const {
        if constexpr ((N % 2) != 0) return false;

        constexpr std::uint32_t NYQUIST = static_cast<std::uint32_t>(N / 2);
        if (coarseBin == 0 || coarseBin >= NYQUIST) return false;

        const auto& lut = sineTable();
        const std::uint64_t coarseFrequencyQ16 =
            static_cast<std::uint64_t>(coarseBin) * 65536ULL;
        ComplexPair halves[2]{};
        correlateHalves(samples, coarseFrequencyQ16, lut, halves);
        const ComplexPair& first = halves[0];
        const ComplexPair& second = halves[1];

        if ((first.re == 0 && first.im == 0) ||
            (second.re == 0 && second.im == 0)) return false;

        // Differential phase = arg(second * conj(first)).
        // Values are normalized to sample units before products to keep
        // cross/dot intermediates within int64_t for the class's N bound.
        const std::int64_t dot = first.re * second.re + first.im * second.im;
        const std::int64_t cross = first.re * second.im - first.im * second.re;
        if (dot == 0 && cross == 0) return false;

        const std::int64_t angleQ30 = atan2Q30(cross, dot);
        const std::int64_t phaseOffsetQ16 =
            divideRounded(angleQ30 * 65536LL, PI_Q30);

        // Differential phase is only unambiguous to +/- half a bin when the
        // coarse bin is the nearest bin. Reject rather than silently aliasing.
        if (phaseOffsetQ16 < -32768LL || phaseOffsetQ16 > 32768LL) return false;

        // Half-window phasors contain a small negative-frequency image, which
        // biases their phase difference for real off-bin tones. A real-valued
        // second-difference recurrence gives an independent frequency estimate
        // and cancels constant ADC offset. When both estimates agree within
        // 1/32 bin, use the recurrence to de-bias the phase estimate.
        std::int64_t offsetQ16 = phaseOffsetQ16;
        std::uint32_t recurrenceFrequencyQ16 = 0;
        if (estimateFrequencyFromDifferences(samples, recurrenceFrequencyQ16)) {
            const std::int64_t recurrenceOffset =
                static_cast<std::int64_t>(recurrenceFrequencyQ16) -
                static_cast<std::int64_t>(coarseFrequencyQ16);
            const std::int64_t disagreement = recurrenceOffset - phaseOffsetQ16;
            const std::int64_t absDisagreement =
                (disagreement < 0) ? -disagreement : disagreement;
            if (recurrenceOffset >= -32768LL && recurrenceOffset <= 32768LL &&
                absDisagreement <= 2048LL) {
                offsetQ16 = recurrenceOffset;
            }
        }

        const std::int64_t refinedFrequencyQ16 =
            static_cast<std::int64_t>(coarseFrequencyQ16) + offsetQ16;
        const std::int64_t maxFrequencyQ16 =
            static_cast<std::int64_t>(NYQUIST) * 65536LL;
        if (refinedFrequencyQ16 <= 0 || refinedFrequencyQ16 >= maxFrequencyQ16)
            return false;

        std::uint32_t amplitude = 0;
        if (!fitSinusoidAmplitude(samples,
                                  static_cast<std::uint64_t>(refinedFrequencyQ16),
                                  lut, amplitude)) return false;
        // Keep the same scale as the coarse detector strength (approximately A*N).
        const std::uint64_t strength = static_cast<std::uint64_t>(amplitude) * N;

        out.coarse_bin = coarseBin;
        out.offset_q16 = static_cast<std::int32_t>(offsetQ16);
        out.phase_offset_q16 = static_cast<std::int32_t>(phaseOffsetQ16);
        out.frequency_q16 = static_cast<std::uint32_t>(refinedFrequencyQ16);
        out.peak_amplitude = amplitude;
        out.strength = strength;
        out.phase_difference_q30 = angleQ30;
        return true;
    }

    // Convenience API: do the coarse scan, select the strongest detector, and
    // refine that peak. For this to guarantee a nearest-bin candidate, scan
    // consecutive bins (span == DETECTORS-1) and avoid DC/Nyquist tones.
    bool analyzeWithDifferentialRefinement(const InputBlock& samples,
                                           std::uint32_t firstBin,
                                           std::uint32_t lastBin,
                                           Results& out,
                                           DifferentialResult& refined) const {
        if constexpr (DETECTORS > 1) {
            if (lastBin < firstBin || (lastBin - firstBin) != DETECTORS - 1)
                return false;
        }
        if (!analyze(samples, firstBin, lastBin, out)) return false;
        const auto best = std::max_element(
            out.begin(), out.end(),
            [](const Result& a, const Result& b) { return a.strength < b.strength; });
        return refineSingleTone(samples, best->bin, refined);
    }

private:
    struct ComplexPair {
        std::int64_t re = 0;
        std::int64_t im = 0;
    };

    static const std::array<std::int16_t, LUT_SIZE>& sineTable() {
        static const std::array<std::int16_t, LUT_SIZE> table = [] {
            std::array<std::int16_t, LUT_SIZE> t{};
            constexpr double PI = 3.1415926535897932384626433832795;
            for (std::size_t i = 0; i < LUT_SIZE; ++i) {
                const double angle = (2.0 * PI * static_cast<double>(i)) /
                                     static_cast<double>(LUT_SIZE);
                t[i] = static_cast<std::int16_t>(std::lround(std::sin(angle) * 32767.0));
            }
            return t;
        }();
        return table;
    }

    static std::int64_t divideRounded(std::int64_t x, std::int64_t divisor) {
        if (x >= 0) return (x + divisor / 2) / divisor;
        return -((-x + divisor / 2) / divisor);
    }

    static std::uint64_t integerSqrt(std::uint64_t x) {
        std::uint64_t root = 0;
        std::uint64_t bit = std::uint64_t{1} << 62;
        while (bit > x) bit >>= 2;
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

    // Defined floor division by 2^shift, avoiding implementation-defined
    // signed right-shift behavior for negative CORDIC intermediates.
    static std::int64_t shiftRightFloor(std::int64_t value, unsigned shift) {
        if (shift == 0) return value;
        const std::int64_t divisor = std::int64_t{1} << shift;
        if (value >= 0) return value / divisor;
        return -((-value + divisor - 1) / divisor);
    }

    // atan2(y,x) in radians scaled by 2^30, using integer CORDIC vectoring.
    static std::int64_t atan2Q30(std::int64_t y, std::int64_t x) {
        static constexpr std::array<std::int64_t, 28> ATAN = {
            843314857LL, 497837829LL, 263043837LL, 133525159LL,
             67021687LL,  33543516LL,  16775851LL,   8388437LL,
              4194283LL,   2097149LL,   1048576LL,    524288LL,
               262144LL,    131072LL,     65536LL,     32768LL,
                16384LL,      8192LL,      4096LL,      2048LL,
                 1024LL,       512LL,       256LL,       128LL,
                   64LL,        32LL,        16LL,         8LL
        };
        if (x == 0) {
            if (y > 0) return PI_Q30 / 2;
            if (y < 0) return -PI_Q30 / 2;
            return 0;
        }

        std::int64_t z = 0;
        if (x < 0) {
            if (y >= 0) z = PI_Q30;
            else z = -PI_Q30;
            x = -x;
            y = -y;
        }

        for (unsigned i = 0; i < ATAN.size(); ++i) {
            const std::int64_t oldX = x;
            const std::int64_t oldY = y;
            if (oldY > 0) {
                x = oldX + shiftRightFloor(oldY, i);
                y = oldY - shiftRightFloor(oldX, i);
                z += ATAN[i];
            } else if (oldY < 0) {
                x = oldX - shiftRightFloor(oldY, i);
                y = oldY + shiftRightFloor(oldX, i);
                z -= ATAN[i];
            }
        }
        return z;
    }

    // Convert a Q16-bin frequency to a phase step represented as quotient plus
    // Bresenham remainder. denominator = N * 2^16; wrap of uint32_t phase is
    // intentional modulo-2^32 arithmetic.
    static void phaseStep(std::uint64_t frequencyQ16,
                          std::uint64_t& stepBase,
                          std::uint64_t& stepRemainder,
                          std::uint64_t& denominator) {
        denominator = static_cast<std::uint64_t>(N) * 65536ULL;
        const std::uint64_t numerator = frequencyQ16 * (std::uint64_t{1} << 32);
        stepBase = numerator / denominator;
        stepRemainder = numerator % denominator;
    }

    static ComplexPair correlateAll(
        const InputBlock& samples,
        std::uint64_t frequencyQ16,
        const std::array<std::int16_t, LUT_SIZE>& lut) {
        std::uint64_t stepBase = 0, stepRemainder = 0, denominator = 0;
        phaseStep(frequencyQ16, stepBase, stepRemainder, denominator);
        std::uint64_t phaseError = 0;
        std::uint32_t phase = 0;
        std::int64_t realAccQ15 = 0;
        std::int64_t imagAccQ15 = 0;
        constexpr std::size_t MASK = LUT_SIZE - 1;
        constexpr std::size_t QUARTER_TURN = LUT_SIZE / 4;

        for (std::size_t n = 0; n < N; ++n) {
            const std::uint32_t index = phase >> PHASE_INDEX_SHIFT;
            const std::int32_t sinQ15 = lut[index];
            const std::int32_t cosQ15 = lut[(index + QUARTER_TURN) & MASK];
            const std::int64_t sample = samples[n];
            realAccQ15 += sample * cosQ15;
            imagAccQ15 -= sample * sinQ15; // e^(-j phase): positive offset gives positive phase slope

            phase += static_cast<std::uint32_t>(stepBase);
            phaseError += stepRemainder;
            if (phaseError >= denominator) {
                ++phase;
                phaseError -= denominator;
            }
        }
        return {divideRounded(realAccQ15, Q15_ONE),
                divideRounded(imagAccQ15, Q15_ONE)};
    }

    static void correlateHalves(
        const InputBlock& samples,
        std::uint64_t frequencyQ16,
        const std::array<std::int16_t, LUT_SIZE>& lut,
        ComplexPair (&halves)[2]) {
        std::uint64_t stepBase = 0, stepRemainder = 0, denominator = 0;
        phaseStep(frequencyQ16, stepBase, stepRemainder, denominator);
        std::uint64_t phaseError = 0;
        std::uint32_t phase = 0;
        std::int64_t realAccQ15[2] = {0, 0};
        std::int64_t imagAccQ15[2] = {0, 0};
        constexpr std::size_t MASK = LUT_SIZE - 1;
        constexpr std::size_t QUARTER_TURN = LUT_SIZE / 4;
        constexpr std::size_t HALF = N / 2;

        for (std::size_t n = 0; n < N; ++n) {
            const std::size_t h = (n < HALF) ? 0 : 1;
            const std::uint32_t index = phase >> PHASE_INDEX_SHIFT;
            const std::int32_t sinQ15 = lut[index];
            const std::int32_t cosQ15 = lut[(index + QUARTER_TURN) & MASK];
            const std::int64_t sample = samples[n];
            realAccQ15[h] += sample * cosQ15;
            imagAccQ15[h] -= sample * sinQ15;

            phase += static_cast<std::uint32_t>(stepBase);
            phaseError += stepRemainder;
            if (phaseError >= denominator) {
                ++phase;
                phaseError -= denominator;
            }
        }

        halves[0] = {divideRounded(realAccQ15[0], Q15_ONE),
                     divideRounded(imagAccQ15[0], Q15_ONE)};
        halves[1] = {divideRounded(realAccQ15[1], Q15_ONE),
                     divideRounded(imagAccQ15[1], Q15_ONE)};
    }

    // Q30 fixed-point ratio num/den using binary long division. The remainder
    // is kept below den, so doubling it is safe for the bounded sums here.
    static std::int64_t ratioQ30(std::int64_t num, std::int64_t den) {
        if (den <= 0) return 0;
        const bool negative = num < 0;
        std::uint64_t magnitude = static_cast<std::uint64_t>(negative ? -num : num);
        const std::uint64_t denominator = static_cast<std::uint64_t>(den);
        constexpr std::uint64_t ONE_Q30 = std::uint64_t{1} << 30;
        if (magnitude >= denominator)
            return negative ? -static_cast<std::int64_t>(ONE_Q30)
                            :  static_cast<std::int64_t>(ONE_Q30);

        std::uint64_t quotient = 0;
        for (unsigned i = 0; i < 30; ++i) {
            magnitude <<= 1;
            quotient <<= 1;
            if (magnitude >= denominator) {
                magnitude -= denominator;
                quotient |= 1U;
            }
        }
        if (magnitude * 2U >= denominator) ++quotient;
        const std::int64_t signedQuotient = static_cast<std::int64_t>(quotient);
        return negative ? -signedQuotient : signedQuotient;
    }

    // Estimate angular frequency from the sinusoid recurrence
    // y[n-1] + y[n+1] = 2*cos(w)*y[n], where y is first-differenced input.
    // Differencing suppresses a constant ADC bias. This estimator is intended
    // as a single-dominant-tone correction; it is not suitable for dense mixtures.
    static bool estimateFrequencyFromDifferences(
        const InputBlock& samples, std::uint32_t& frequencyQ16) {
        std::int64_t numerator = 0;
        std::int64_t energy = 0;
        for (std::size_t n = 1; n + 2 < N; ++n) {
            const std::int64_t previous =
                static_cast<std::int64_t>(samples[n]) - samples[n - 1];
            const std::int64_t current =
                static_cast<std::int64_t>(samples[n + 1]) - samples[n];
            const std::int64_t next =
                static_cast<std::int64_t>(samples[n + 2]) - samples[n + 1];
            numerator += current * (previous + next);
            energy += current * current;
        }
        if (energy == 0) return false;

        const std::int64_t cosineQ30 = ratioQ30(numerator, 2 * energy);
        constexpr std::int64_t ONE_Q30 = std::int64_t{1} << 30;
        const std::int64_t boundedCosine =
            std::max(-ONE_Q30, std::min(ONE_Q30, cosineQ30));
        const std::uint64_t sineSquared = static_cast<std::uint64_t>(
            ONE_Q30 * ONE_Q30 - boundedCosine * boundedCosine);
        const std::int64_t sineQ30 = static_cast<std::int64_t>(integerSqrt(sineSquared));
        const std::int64_t omegaQ30 = atan2Q30(sineQ30, boundedCosine);
        const std::int64_t binQ16 = divideRounded(
            omegaQ30 * static_cast<std::int64_t>(N) * 65536LL,
            2 * PI_Q30);
        const std::int64_t maxBinQ16 =
            static_cast<std::int64_t>(N / 2) * 65536LL;
        if (binQ16 <= 0 || binQ16 >= maxBinQ16) return false;
        frequencyQ16 = static_cast<std::uint32_t>(binQ16);
        return true;
    }

    // Real-sinusoid least-squares fit at the refined frequency. A raw complex
    // correlation is biased for off-bin real tones because cos/sin are not
    // exactly orthogonal over a finite, non-integer number of cycles: the
    // negative-frequency image leaks into the positive-frequency estimate.
    // Solving the 2x2 Gram system removes that single-tone image contribution.
    static bool fitSinusoidAmplitude(
        const InputBlock& samples,
        std::uint64_t frequencyQ16,
        const std::array<std::int16_t, LUT_SIZE>& lut,
        std::uint32_t& amplitude) {
        std::uint64_t stepBase = 0, stepRemainder = 0, denominator = 0;
        phaseStep(frequencyQ16, stepBase, stepRemainder, denominator);
        std::uint64_t phaseError = 0;
        std::uint32_t phase = 0;
        std::int64_t xCosQ15 = 0, xSinQ15 = 0, sampleSum = 0;
        std::int64_t cosSumQ15 = 0, sinSumQ15 = 0;
        std::int64_t cosCosQ30 = 0, sinSinQ30 = 0, cosSinQ30 = 0;
        constexpr std::size_t MASK = LUT_SIZE - 1;
        constexpr std::size_t QUARTER_TURN = LUT_SIZE / 4;

        for (std::size_t n = 0; n < N; ++n) {
            const std::uint32_t index = phase >> PHASE_INDEX_SHIFT;
            const std::int64_t sinQ15 = lut[index];
            const std::int64_t cosQ15 = lut[(index + QUARTER_TURN) & MASK];
            const std::int64_t sample = samples[n];

            xCosQ15 += sample * cosQ15;
            xSinQ15 += sample * sinQ15;
            sampleSum += sample;
            cosSumQ15 += cosQ15;
            sinSumQ15 += sinQ15;
            cosCosQ30 += cosQ15 * cosQ15;
            sinSinQ30 += sinQ15 * sinQ15;
            cosSinQ30 += cosQ15 * sinQ15;

            phase += static_cast<std::uint32_t>(stepBase);
            phaseError += stepRemainder;
            if (phaseError >= denominator) {
                ++phase;
                phaseError -= denominator;
            }
        }

        // Correlations become sample units. Gram entries use Q16 scaling,
        // preserving sub-unit precision without overflowing int64_t.
        constexpr std::int64_t Q15_SQUARED = Q15_ONE * Q15_ONE;
        const std::int64_t sumCos = divideRounded(cosSumQ15, Q15_ONE);
        const std::int64_t sumSin = divideRounded(sinSumQ15, Q15_ONE);

        // Center x, cos and sin jointly, equivalent to fitting an additional
        // constant offset term. This prevents a DC-biased ADC from leaking
        // into an off-bin sinusoid's amplitude estimate.
        const std::int64_t rCos =
            divideRounded(xCosQ15, Q15_ONE) -
            divideRounded(sampleSum * sumCos, static_cast<std::int64_t>(N));
        const std::int64_t rSin =
            divideRounded(xSinQ15, Q15_ONE) -
            divideRounded(sampleSum * sumSin, static_cast<std::int64_t>(N));
        const std::int64_t gCC =
            divideRounded(cosCosQ30 * 65536LL, Q15_SQUARED) -
            divideRounded(sumCos * sumCos * 65536LL, static_cast<std::int64_t>(N));
        const std::int64_t gSS =
            divideRounded(sinSinQ30 * 65536LL, Q15_SQUARED) -
            divideRounded(sumSin * sumSin * 65536LL, static_cast<std::int64_t>(N));
        const std::int64_t gCS =
            divideRounded(cosSinQ30 * 65536LL, Q15_SQUARED) -
            divideRounded(sumCos * sumSin * 65536LL, static_cast<std::int64_t>(N));
        const std::int64_t determinant = gCC * gSS - gCS * gCS;
        const std::int64_t scaledDeterminant = determinant / 65536LL;
        if (scaledDeterminant <= 0) return false;

        // Solve for x[n] = a*cos(w*n) + b*sin(w*n).
        const std::int64_t numeratorA = rCos * gSS - rSin * gCS;
        const std::int64_t numeratorB = rSin * gCC - rCos * gCS;
        const std::int64_t a = divideRounded(numeratorA, scaledDeterminant);
        const std::int64_t b = divideRounded(numeratorB, scaledDeterminant);
        const std::uint64_t coefficientMagnitudeSquared =
            static_cast<std::uint64_t>(a * a) +
            static_cast<std::uint64_t>(b * b);
        const std::uint64_t coefficientMagnitude = integerSqrt(coefficientMagnitudeSquared);
        amplitude = static_cast<std::uint32_t>(coefficientMagnitude);
        return true;
    }

    static Result analyzeBin(const InputBlock& samples,
                             std::uint32_t bin,
                             const std::array<std::int16_t, LUT_SIZE>& lut) {
        const std::uint64_t frequencyQ16 = static_cast<std::uint64_t>(bin) * 65536ULL;
        const ComplexPair c = correlateAll(samples, frequencyQ16, lut);
        const std::uint64_t magnitudeSquared =
            static_cast<std::uint64_t>(c.re * c.re) +
            static_cast<std::uint64_t>(c.im * c.im);
        const std::uint64_t magnitude = integerSqrt(magnitudeSquared);
        const bool endpoint = (bin == 0) || ((N % 2 == 0) && (bin == N / 2));
        const std::uint64_t strength = (endpoint ? 1ULL : 2ULL) * magnitude;
        const std::uint32_t amplitude = static_cast<std::uint32_t>(
            (strength + N / 2) / N);
        return Result{bin, amplitude, strength};
    }
};


// Multi-tone extension: two interleaved detector banks (integer bins and
// half-bin offsets) form a uniform 0.5-bin comb. Each detector computes the
// quadrature cross-correlation of the current residual against a Q15 sine /
// cosine reference. The strongest response is least-squares fitted, its time-
// domain waveform is subtracted, and the scan repeats (matching pursuit).
//
// All sample-loop phase scheduling is integer-only. The 32-bit phase wraps
// modulo 2^32 intentionally. Floating point is used only by the test signal
// generator and one-time LUT initialization.
template <std::size_t N, std::size_t MAX_TONES, unsigned LUT_BITS = 12>
class MultiToneComb {
    static_assert(N >= 64 && N <= 4096,
                  "N must be in [64, 4096] for bounded fixed-point products");
    static_assert((N % 2) == 0, "N must be even");
    static_assert(MAX_TONES >= 1 && MAX_TONES <= 16,
                  "MAX_TONES must be in [1, 16]");
    static_assert(LUT_BITS >= 8 && LUT_BITS <= 12,
                  "LUT_BITS must be in [8, 12]");

    static constexpr std::size_t LUT_SIZE = std::size_t{1} << LUT_BITS;
    static constexpr std::size_t LUT_MASK = LUT_SIZE - 1;
    static constexpr unsigned PHASE_SHIFT = 32U - LUT_BITS;
    static constexpr std::int64_t Q8_ONE = 256;
    static constexpr std::int64_t Q15_ONE = 32768;
    static constexpr std::int64_t GRAM_SCALE = std::int64_t{1} << 23;
    static constexpr std::size_t MAX_GRID_POINTS = 4096;
    static constexpr std::size_t MAX_COEFFICIENTS = MAX_TONES * 2;
    static constexpr std::int64_t ONE_Q30 = std::int64_t{1} << 30;

public:
    using InputBlock = std::array<std::int16_t, N>;
    using ResidualBlock = std::array<std::int64_t, N>; // Q8 sample units

    struct Tone {
        std::uint32_t frequencyQ16 = 0; // DFT-bin position, Q16 (65536 == one bin)
        std::uint32_t peakAmplitude = 0; // original sample units
        std::int32_t cosineQ8 = 0; // fitted coefficient, Q8 sample units
        std::int32_t sineQ8 = 0;   // fitted coefficient, Q8 sample units
        std::uint8_t discoveryBank = 0; // 0=integer; 1=half-bin; 2=caller-defined grid
        std::uint16_t wienerGainQ15 = 32768; // 1.0 before optional Wiener shrinkage
    };

    using ToneList = std::array<Tone, MAX_TONES>;
    using DenoisedBlock = InputBlock;

    // Empirical-Bayes/Wiener shrinkage of fitted sinusoid coefficients.
    // noiseSigmaQ8==0 estimates sigma from the post-fit residual RMS;
    // otherwise it supplies an external noise RMS in Q8 sample units.
    struct WienerOptions {
        std::uint32_t noiseSigmaQ8 = 0;
        std::uint16_t noisePowerScaleQ8 = 256; // 256 == 1.0; allowed 1..4096
        std::uint16_t minimumGainQ15 = 0;      // optional gain floor, 0..32768
        std::uint16_t suppressBelowGainQ15 = 0; // drop tones below gain; 0 disables threshold
    };

    struct WienerReport {
        std::uint32_t noiseSigmaQ8 = 0;
        std::uint64_t noiseCoefficientPowerQ16 = 0;
        std::uint16_t tonesAttenuated = 0;
        std::uint16_t tonesSuppressed = 0;
    };

    static void prepare() { (void)sineTable(); }

    // Legacy API: inspect the integer/half-bin comb over an inclusive bin range.
    // Frequencies are normalized DFT-bin positions; 65536 == one bin.
    bool decompose(const InputBlock& samples,
                   std::uint32_t firstBin,
                   std::uint32_t lastBin,
                   std::uint32_t minimumAmplitude,
                   ToneList& tones,
                   std::size_t& toneCount,
                   ResidualBlock& residual,
                   std::int32_t& removedDc) const {
        return decomposeImpl(samples, firstBin, lastBin, nullptr, 0,
                             minimumAmplitude, tones, toneCount,
                             residual, removedDc);
    }

    // Arbitrary frequency-grid API. The caller owns the sorted grid, so custom
    // scales (logarithmic, mel, Bark, ERB, octave bands, measured frequencies,
    // or application-specific points) need no allocation inside this class.
    // Values are Q16 DFT bins, not Hz: frequencyQ16 = f_Hz * N/Fs * 65536.
    bool decomposeFrequencies(const InputBlock& samples,
                             const std::uint32_t* frequenciesQ16,
                             std::size_t frequencyCount,
                             std::uint32_t minimumAmplitude,
                             ToneList& tones,
                             std::size_t& toneCount,
                             ResidualBlock& residual,
                             std::int32_t& removedDc) const {
        if (frequenciesQ16 == nullptr || frequencyCount < 2 ||
            frequencyCount > MAX_GRID_POINTS) return false;
        constexpr std::uint32_t NYQUIST_Q16 =
            static_cast<std::uint32_t>(N / 2) * 65536U;
        if (frequenciesQ16[0] == 0 || frequenciesQ16[frequencyCount - 1] >= NYQUIST_Q16)
            return false;
        for (std::size_t i = 1; i < frequencyCount; ++i) {
            if (frequenciesQ16[i] <= frequenciesQ16[i - 1]) return false;
        }
        return decomposeImpl(samples, 0, 0, frequenciesQ16, frequencyCount,
                             minimumAmplitude, tones, toneCount,
                             residual, removedDc);
    }

    template <std::size_t GRID_POINTS>
    bool decomposeGrid(const InputBlock& samples,
                       const std::array<std::uint32_t, GRID_POINTS>& frequenciesQ16,
                       std::uint32_t minimumAmplitude,
                       ToneList& tones,
                       std::size_t& toneCount,
                       ResidualBlock& residual,
                       std::int32_t& removedDc) const {
        static_assert(GRID_POINTS >= 2, "A custom grid needs at least two points");
        return decomposeFrequencies(samples, frequenciesQ16.data(), GRID_POINTS,
                                    minimumAmplitude, tones, toneCount,
                                    residual, removedDc);
    }

    // Wiener-enabled built-in half-bin-comb decomposition. This runs the same
    // multi-tone estimator, estimates residual noise (or uses a supplied noise
    // RMS), then shrinks the fitted cosine/sine coefficients with a Wiener-like
    // power gain. Existing decompose() remains unchanged for compatibility.
    bool decomposeWiener(const InputBlock& samples,
                         std::uint32_t firstBin,
                         std::uint32_t lastBin,
                         std::uint32_t minimumAmplitude,
                         const WienerOptions& wiener,
                         ToneList& tones,
                         std::size_t& toneCount,
                         ResidualBlock& residual,
                         std::int32_t& removedDc,
                         WienerReport& report) const {
        if (!decompose(samples, firstBin, lastBin, minimumAmplitude,
                       tones, toneCount, residual, removedDc)) return false;
        return applyWiener(residual, tones, toneCount, wiener, report);
    }

    // Wiener-enabled caller-supplied frequency grid. Grid values remain Q16
    // DFT-bin positions; they may be logarithmic, linear, or arbitrary.
    bool decomposeFrequenciesWiener(const InputBlock& samples,
                                    const std::uint32_t* frequenciesQ16,
                                    std::size_t frequencyCount,
                                    std::uint32_t minimumAmplitude,
                                    const WienerOptions& wiener,
                                    ToneList& tones,
                                    std::size_t& toneCount,
                                    ResidualBlock& residual,
                                    std::int32_t& removedDc,
                                    WienerReport& report) const {
        if (!decomposeFrequencies(samples, frequenciesQ16, frequencyCount,
                                  minimumAmplitude, tones, toneCount,
                                  residual, removedDc)) return false;
        return applyWiener(residual, tones, toneCount, wiener, report);
    }

    template <std::size_t GRID_POINTS>
    bool decomposeGridWiener(const InputBlock& samples,
                             const std::array<std::uint32_t, GRID_POINTS>& frequenciesQ16,
                             std::uint32_t minimumAmplitude,
                             const WienerOptions& wiener,
                             ToneList& tones,
                             std::size_t& toneCount,
                             ResidualBlock& residual,
                             std::int32_t& removedDc,
                             WienerReport& report) const {
        return decomposeFrequenciesWiener(samples, frequenciesQ16.data(), GRID_POINTS,
                                          minimumAmplitude, wiener, tones, toneCount,
                                          residual, removedDc, report);
    }

    // Reconstruct the estimated clean signal as input minus residual. The
    // residual is in Q8 units; result is rounded and saturated to int16_t.
    // With Wiener-enabled decomposition, this is the denoised tone model plus
    // the estimated DC component; broadband residual noise is not re-added.
    static void reconstructDenoised(const InputBlock& original,
                                    const ResidualBlock& residual,
                                    DenoisedBlock& denoised) {
        for (std::size_t n = 0; n < N; ++n) {
            const std::int64_t modelQ8 =
                static_cast<std::int64_t>(original[n]) * Q8_ONE - residual[n];
            std::int64_t sample = divideRounded(modelQ8, Q8_ONE);
            if (sample > std::numeric_limits<std::int16_t>::max())
                sample = std::numeric_limits<std::int16_t>::max();
            if (sample < std::numeric_limits<std::int16_t>::min())
                sample = std::numeric_limits<std::int16_t>::min();
            denoised[n] = static_cast<std::int16_t>(sample);
        }
    }

    // Construct evenly spaced frequencies with exact quotient/remainder
    // distribution (Bresenham) in the Q16-bin domain.
    // Convert a caller-chosen frequency to this block's Q16-bin domain.
    // Example: frequencyHzToBinQ16(123.4, 8000.0, fQ16) before adding fQ16
    // to a custom std::array<uint32_t, ...> and calling decomposeGrid().
    static bool frequencyHzToBinQ16(double frequencyHz,
                                    double sampleRateHz,
                                    std::uint32_t& outFrequencyQ16) {
        if (!(sampleRateHz > 0.0) || !(frequencyHz > 0.0) ||
            !(frequencyHz < sampleRateHz * 0.5)) return false;
        const double value = frequencyHz * static_cast<double>(N) * 65536.0 /
                             sampleRateHz;
        if (!(value > 0.0) ||
            value >= static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
            return false;
        outFrequencyQ16 = static_cast<std::uint32_t>(std::llround(value));
        return outFrequencyQ16 != 0;
    }

    static double frequencyQ16ToHz(std::uint32_t frequencyQ16,
                                   double sampleRateHz) {
        return static_cast<double>(frequencyQ16) * sampleRateHz /
               (static_cast<double>(N) * 65536.0);
    }

    template <std::size_t GRID_POINTS>
    static bool makeLinearGridHz(double minHz,
                                 double maxHz,
                                 double sampleRateHz,
                                 std::array<std::uint32_t, GRID_POINTS>& out) {
        static_assert(GRID_POINTS >= 2, "A grid needs at least two points");
        if (!(sampleRateHz > 0.0) || !(minHz > 0.0) || !(maxHz > minHz) ||
            !(maxHz < sampleRateHz * 0.5)) return false;
        const double scale = static_cast<double>(N) * 65536.0 / sampleRateHz;
        const double firstD = minHz * scale;
        const double lastD = maxHz * scale;
        if (lastD >= static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
            return false;
        const std::uint64_t first = static_cast<std::uint64_t>(std::llround(firstD));
        const std::uint64_t last = static_cast<std::uint64_t>(std::llround(lastD));
        if (last <= first || last - first < GRID_POINTS - 1) return false;
        const std::uint64_t intervals = GRID_POINTS - 1;
        const std::uint64_t span = last - first;
        const std::uint64_t step = span / intervals;
        const std::uint64_t remainder = span % intervals;
        std::uint64_t f = first;
        std::uint64_t error = 0;
        for (std::size_t i = 0; i < GRID_POINTS; ++i) {
            out[i] = static_cast<std::uint32_t>(f);
            if (i + 1 < GRID_POINTS) {
                f += step;
                error += remainder;
                if (error >= intervals) {
                    ++f;
                    error -= intervals;
                }
            }
        }
        return true;
    }

    // Construct a logarithmic frequency scale. The logarithm/exponential are
    // used only during grid setup; all detector correlations and fitting remain
    // integer/fixed-point. Equal spacing is in log(Hz), giving more absolute
    // resolution at low frequencies. The Q16-bin grid must remain strictly
    // increasing after quantization or this function returns false.
    template <std::size_t GRID_POINTS>
    static bool makeLogGridHz(double minHz,
                              double maxHz,
                              double sampleRateHz,
                              std::array<std::uint32_t, GRID_POINTS>& out) {
        static_assert(GRID_POINTS >= 2, "A grid needs at least two points");
        if (!(sampleRateHz > 0.0) || !(minHz > 0.0) || !(maxHz > minHz) ||
            !(maxHz < sampleRateHz * 0.5)) return false;
        const double logMin = std::log(minHz);
        const double logSpan = std::log(maxHz) - logMin;
        const double scale = static_cast<double>(N) * 65536.0 / sampleRateHz;
        std::uint32_t previous = 0;
        for (std::size_t i = 0; i < GRID_POINTS; ++i) {
            const double fraction = static_cast<double>(i) /
                                    static_cast<double>(GRID_POINTS - 1);
            const double hz = std::exp(logMin + fraction * logSpan);
            const double binQ16D = hz * scale;
            if (!(binQ16D > 0.0) ||
                binQ16D >= static_cast<double>(std::numeric_limits<std::uint32_t>::max()))
                return false;
            const auto binQ16 = static_cast<std::uint32_t>(std::llround(binQ16D));
            if (i != 0 && binQ16 <= previous) return false;
            out[i] = binQ16;
            previous = binQ16;
        }
        return true;
    }

    static std::uint64_t residualRmsQ8(const ResidualBlock& residual) {
        std::uint64_t energy = 0;
        for (const std::int64_t x : residual) {
            const std::uint64_t mag = static_cast<std::uint64_t>(x < 0 ? -x : x);
            energy += mag * mag;
        }
        return integerSqrt(energy / N);
    }

private:
    static std::uint64_t coefficientPowerQ16(const Tone& tone) {
        const std::int64_t c = tone.cosineQ8;
        const std::int64_t s = tone.sineQ8;
        return static_cast<std::uint64_t>(c * c) +
               static_cast<std::uint64_t>(s * s);
    }

    static std::uint16_t ratioQ15(std::uint64_t numerator,
                                  std::uint64_t denominator) {
        if (denominator == 0 || numerator == 0) return 0;
        if (numerator >= denominator) return 32768;
        // Scale both operands down together before multiplying by 2^15.
        // This avoids overflow even if a badly conditioned fit approaches the
        // signed 32-bit coefficient limit.
        while (denominator > (std::numeric_limits<std::uint64_t>::max() / 32768ULL)) {
            denominator >>= 1;
            numerator >>= 1;
            if (denominator == 0) return 0;
        }
        const std::uint64_t q =
            (numerator * 32768ULL + denominator / 2ULL) / denominator;
        return static_cast<std::uint16_t>(std::min<std::uint64_t>(q, 32768ULL));
    }

    bool applyWiener(ResidualBlock& residual,
                     ToneList& tones,
                     std::size_t& toneCount,
                     const WienerOptions& options,
                     WienerReport& report) const {
        report = WienerReport{};
        if (toneCount > MAX_TONES || options.noisePowerScaleQ8 == 0 ||
            options.noisePowerScaleQ8 > 4096 ||
            options.minimumGainQ15 > 32768 ||
            options.suppressBelowGainQ15 > 32768 ||
            options.noiseSigmaQ8 > 32768U * 256U) return false;

        const std::uint64_t estimatedSigmaQ8 =
            options.noiseSigmaQ8 == 0 ? residualRmsQ8(residual) : options.noiseSigmaQ8;
        report.noiseSigmaQ8 = static_cast<std::uint32_t>(estimatedSigmaQ8);

        // For white residual noise with sample-domain variance sigma^2, the
        // variance of each fitted cosine/sine coefficient is approximately
        // 2*sigma^2/N. Thus E[a^2+b^2] ~= 4*sigma^2/N. This is the expected
        // noise power of one fitted sinusoid's Q8 coefficient vector.
        const std::uint64_t sigmaSquared = estimatedSigmaQ8 * estimatedSigmaQ8;
        const std::uint64_t scaledNoisePower =
            sigmaSquared * 4ULL * options.noisePowerScaleQ8;
        const std::uint64_t denominator = static_cast<std::uint64_t>(N) * 256ULL;
        const std::uint64_t noiseCoefficientPower =
            (scaledNoisePower + denominator / 2ULL) / denominator;
        report.noiseCoefficientPowerQ16 = noiseCoefficientPower;

        ToneList keptTones{};
        std::size_t keptCount = 0;
        for (std::size_t i = 0; i < toneCount; ++i) {
            Tone tone = tones[i];
            const std::uint64_t observedPower = coefficientPowerQ16(tone);
            const std::uint64_t cleanPower = observedPower > noiseCoefficientPower
                ? observedPower - noiseCoefficientPower : 0ULL;
            std::uint16_t gain = ratioQ15(cleanPower, observedPower);

            const bool suppress = (gain == 0) ||
                (options.suppressBelowGainQ15 != 0 &&
                 gain < options.suppressBelowGainQ15);

            // Undo this component in the current residual before replacing its
            // coefficients. This keeps residual == input - current tone model.
            applyTone(residual, tone, +1);
            if (suppress) {
                ++report.tonesSuppressed;
                continue;
            }

            gain = std::max(gain, options.minimumGainQ15);
            tone.cosineQ8 = clampI32(divideRounded(
                static_cast<std::int64_t>(tone.cosineQ8) * gain, 32768));
            tone.sineQ8 = clampI32(divideRounded(
                static_cast<std::int64_t>(tone.sineQ8) * gain, 32768));
            tone.wienerGainQ15 = gain;
            const std::uint64_t filteredPower = coefficientPowerQ16(tone);
            const std::uint64_t filteredAmplitudeQ8 = integerSqrt(filteredPower);
            tone.peakAmplitude = static_cast<std::uint32_t>(
                (filteredAmplitudeQ8 + Q8_ONE / 2) / Q8_ONE);
            if (gain < 32768) ++report.tonesAttenuated;

            applyTone(residual, tone, -1);
            keptTones[keptCount++] = tone;
        }

        tones = keptTones;
        toneCount = keptCount;

        // Wiener gains can change amplitude ranking; preserve the public
        // descending-amplitude ordering used by the unfiltered API.
        for (std::size_t i = 1; i < toneCount; ++i) {
            const Tone key = tones[i];
            std::size_t j = i;
            while (j > 0 && tones[j - 1].peakAmplitude < key.peakAmplitude) {
                tones[j] = tones[j - 1];
                --j;
            }
            tones[j] = key;
        }
        return true;
    }

    bool decomposeImpl(const InputBlock& samples,
                       std::uint32_t firstBin,
                       std::uint32_t lastBin,
                       const std::uint32_t* gridQ16,
                       std::size_t gridCount,
                       std::uint32_t minimumAmplitude,
                       ToneList& tones,
                       std::size_t& toneCount,
                       ResidualBlock& residual,
                       std::int32_t& removedDc) const {
        constexpr std::uint32_t NYQUIST = static_cast<std::uint32_t>(N / 2);
        if (gridQ16 == nullptr) {
            if (firstBin < 1 || firstBin >= lastBin || lastBin >= NYQUIST)
                return false;
        } else {
            if (gridCount < 2 || gridCount > MAX_GRID_POINTS) return false;
            const std::uint32_t nyquistQ16 = NYQUIST * 65536U;
            if (gridQ16[0] == 0 || gridQ16[gridCount - 1] >= nyquistQ16)
                return false;
            for (std::size_t i = 1; i < gridCount; ++i) {
                if (gridQ16[i] <= gridQ16[i - 1]) return false;
            }
        }

        std::int64_t sampleSum = 0;
        for (const std::int16_t x : samples) sampleSum += x;
        const std::int64_t dcQ8 = divideRounded(
            sampleSum * Q8_ONE, static_cast<std::int64_t>(N));
        removedDc = static_cast<std::int32_t>(divideRounded(dcQ8, Q8_ONE));
        for (std::size_t n = 0; n < N; ++n) {
            residual[n] = static_cast<std::int64_t>(samples[n]) * Q8_ONE - dcQ8;
        }

        toneCount = 0;
        tones.fill(Tone{});
        const std::uint64_t minimumAmplitudeQ8 =
            static_cast<std::uint64_t>(minimumAmplitude) * Q8_ONE;

        while (toneCount < MAX_TONES) {
            Candidate best{};
            bool found = false;
            if (gridQ16 != nullptr) {
                found = findStrongestGridResponse(
                    residual, gridQ16, gridCount, best);
            } else {
                found = findStrongestCombResponse(
                    residual, firstBin, lastBin, best);
            }
            if (!found || best.fit.amplitudeQ8 < minimumAmplitudeQ8) break;

            Candidate refined{};
            if (gridQ16 != nullptr) {
                refined = refineGridCell(
                    residual, gridQ16, gridCount, best.gridIndex);
            } else {
                refined = refineCandidate(
                    residual, best.frequencyQ16, firstBin, lastBin, 16384, 256);
            }
            if (!refined.fit.valid ||
                refined.fit.amplitudeQ8 < minimumAmplitudeQ8) break;

            Tone t{};
            t.frequencyQ16 = refined.frequencyQ16;
            t.cosineQ8 = clampI32(refined.fit.cosineQ8);
            t.sineQ8 = clampI32(refined.fit.sineQ8);
            t.peakAmplitude = static_cast<std::uint32_t>(
                (refined.fit.amplitudeQ8 + Q8_ONE / 2) / Q8_ONE);
            t.discoveryBank = (gridQ16 != nullptr) ? 2U :
                (((best.frequencyQ16 & 0xFFFFU) == 32768U) ? 1U : 0U);

            tones[toneCount++] = t;
            applyTone(residual, t, -1);
        }

        // Coordinate frequency refinement against the residual of all other
        // tones. On a custom scale the search is confined to the selected grid
        // tooth's Voronoi cell; grid spacing can vary arbitrarily.
        for (unsigned pass = 0; pass < 5; ++pass) {
            for (std::size_t i = 0; i < toneCount; ++i) {
                const Tone old = tones[i];
                applyTone(residual, old, +1);
                Candidate update{};
                if (gridQ16 != nullptr) {
                    update = refineGridNeighborhood(
                        residual, gridQ16, gridCount, old.frequencyQ16, 4);
                } else {
                    update = refineCandidate(
                        residual, old.frequencyQ16, firstBin, lastBin,
                        8192, 256);
                }
                if (update.fit.valid) {
                    tones[i].frequencyQ16 = update.frequencyQ16;
                    tones[i].cosineQ8 = clampI32(update.fit.cosineQ8);
                    tones[i].sineQ8 = clampI32(update.fit.sineQ8);
                    tones[i].peakAmplitude = static_cast<std::uint32_t>(
                        (update.fit.amplitudeQ8 + Q8_ONE / 2) / Q8_ONE);
                    applyTone(residual, tones[i], -1);
                } else {
                    applyTone(residual, old, -1);
                }
            }
        }

        if (gridQ16 != nullptr) {
            (void)jointCoordinateFrequencyRefine(samples, gridQ16, gridCount,
                                                 dcQ8, tones, toneCount, residual);
        }

        // Jointly refit all cosine/sine coefficients by solving the full
        // centered Gram system. This accounts for pairwise detector correlation.
        // Components that collapse below minimumAmplitude after the joint fit
        // are pruned, then the remaining coefficients are solved once more.
        bool jointSolved = (toneCount == 0);
        if (toneCount != 0) jointSolved = jointRefit(samples, tones, toneCount);
        if (jointSolved && toneCount != 0) {
            ToneList pruned{};
            std::size_t kept = 0;
            for (std::size_t i = 0; i < toneCount; ++i) {
                if (tones[i].peakAmplitude >= minimumAmplitude)
                    pruned[kept++] = tones[i];
            }
            if (kept < toneCount) {
                if (kept == 0) {
                    tones.fill(Tone{});
                    toneCount = 0;
                } else if (jointRefit(samples, pruned, kept)) {
                    tones = pruned;
                    toneCount = kept;
                }
            }
        }
        // Always rebuild the output residual from the actual final component
        // list; this also keeps it consistent if an ill-conditioned joint solve
        // is deliberately rejected.
        for (std::size_t n = 0; n < N; ++n) {
            residual[n] = static_cast<std::int64_t>(samples[n]) * Q8_ONE - dcQ8;
        }
        for (std::size_t i = 0; i < toneCount; ++i)
            applyTone(residual, tones[i], -1);

        // Keep strongest components first using a bounded insertion sort.
        for (std::size_t i = 1; i < toneCount; ++i) {
            const Tone key = tones[i];
            std::size_t j = i;
            while (j > 0 && tones[j - 1].peakAmplitude < key.peakAmplitude) {
                tones[j] = tones[j - 1];
                --j;
            }
            tones[j] = key;
        }
        return true;
    }

    struct Fit {
        std::int64_t cosineQ8 = 0;
        std::int64_t sineQ8 = 0;
        std::uint64_t amplitudeQ8 = 0;
        bool valid = false;
        std::uint64_t residualError = std::numeric_limits<std::uint64_t>::max();
    };

    struct Candidate {
        std::uint32_t frequencyQ16 = 0;
        Fit fit{};
        std::size_t gridIndex = 0;
    };

    static const std::array<std::int16_t, LUT_SIZE>& sineTable() {
        static const std::array<std::int16_t, LUT_SIZE> table = [] {
            std::array<std::int16_t, LUT_SIZE> out{};
            constexpr double PI = 3.1415926535897932384626433832795;
            for (std::size_t i = 0; i < LUT_SIZE; ++i) {
                const double angle = (2.0 * PI * static_cast<double>(i)) /
                                     static_cast<double>(LUT_SIZE);
                out[i] = static_cast<std::int16_t>(
                    std::lround(std::sin(angle) * 32767.0));
            }
            return out;
        }();
        return table;
    }

    static std::int64_t divideRounded(std::int64_t x, std::int64_t d) {
        if (d <= 0) return 0;
        if (x >= 0) return (x + d / 2) / d;
        return -((-x + d / 2) / d);
    }

    static std::uint64_t integerSqrt(std::uint64_t x) {
        std::uint64_t root = 0;
        std::uint64_t bit = std::uint64_t{1} << 62;
        while (bit > x) bit >>= 2;
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

    static std::int32_t clampI32(std::int64_t x) {
        if (x > std::numeric_limits<std::int32_t>::max())
            return std::numeric_limits<std::int32_t>::max();
        if (x < std::numeric_limits<std::int32_t>::min())
            return std::numeric_limits<std::int32_t>::min();
        return static_cast<std::int32_t>(x);
    }

    static void phaseStep(std::uint64_t frequencyQ16,
                          std::uint64_t& stepBase,
                          std::uint64_t& stepRemainder,
                          std::uint64_t& denominator) {
        denominator = static_cast<std::uint64_t>(N) * 65536ULL;
        const std::uint64_t numerator =
            frequencyQ16 * (std::uint64_t{1} << 32);
        stepBase = numerator / denominator;
        stepRemainder = numerator % denominator;
    }

    // Cross-correlate residual against cos/sin references and solve the 2x2
    // centered Gram system. All correlations and Gram sums are integer.
    static Fit fitAtFrequency(const ResidualBlock& residual,
                              std::uint32_t frequencyQ16) {
        const auto& lut = sineTable();
        std::uint64_t stepBase = 0, stepRemainder = 0, denominator = 0;
        phaseStep(frequencyQ16, stepBase, stepRemainder, denominator);
        std::uint64_t phaseError = 0;
        std::uint32_t phase = 0;

        std::int64_t sumR = 0, sumC = 0, sumS = 0;
        std::int64_t sumRC = 0, sumRS = 0;
        std::int64_t sumCC = 0, sumSS = 0, sumCS = 0;

        for (std::size_t n = 0; n < N; ++n) {
            const std::int64_t s = basisValue(phase, true, lut);
            const std::int64_t c = basisValue(phase, false, lut);
            const std::int64_t r = residual[n];
            sumR += r;
            sumC += c;
            sumS += s;
            sumRC += r * c;
            sumRS += r * s;
            sumCC += c * c;
            sumSS += s * s;
            sumCS += c * s;

            phase += static_cast<std::uint32_t>(stepBase); // intentional Q32 wrap
            phaseError += stepRemainder;
            if (phaseError >= denominator) {
                ++phase;
                phaseError -= denominator;
            }
        }

        // Remove the constant component from correlations and the Gram matrix.
        const std::int64_t xC = divideRounded(
            sumRC - divideRounded(sumR * sumC, static_cast<std::int64_t>(N)), 256);
        const std::int64_t xS = divideRounded(
            sumRS - divideRounded(sumR * sumS, static_cast<std::int64_t>(N)), 256);
        const std::int64_t gCC = divideRounded(
            sumCC - divideRounded(sumC * sumC, static_cast<std::int64_t>(N)), GRAM_SCALE);
        const std::int64_t gSS = divideRounded(
            sumSS - divideRounded(sumS * sumS, static_cast<std::int64_t>(N)), GRAM_SCALE);
        const std::int64_t gCS = divideRounded(
            sumCS - divideRounded(sumC * sumS, static_cast<std::int64_t>(N)), GRAM_SCALE);

        const std::int64_t determinant = gCC * gSS - gCS * gCS;
        if (determinant <= 16 || gCC <= 0 || gSS <= 0) return {};

        const std::int64_t a = divideRounded(xC * gSS - xS * gCS, determinant);
        const std::int64_t b = divideRounded(xS * gCC - xC * gCS, determinant);
        const std::uint64_t aa = static_cast<std::uint64_t>(a < 0 ? -a : a);
        const std::uint64_t bb = static_cast<std::uint64_t>(b < 0 ? -b : b);
        if (aa > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
            bb > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) return {};
        const std::uint64_t amplitude = integerSqrt(aa * aa + bb * bb);

        // Optimize frequency against actual residual energy, not merely the
        // fitted coefficient magnitude. Coefficient magnitude can be slightly
        // biased by the non-orthogonal finite-window sine/cosine Gram matrix;
        // SSE gives a better-defined objective for local frequency refinement.
        const std::int64_t meanC = divideRounded(sumC, static_cast<std::int64_t>(N));
        const std::int64_t meanS = divideRounded(sumS, static_cast<std::int64_t>(N));
        phase = 0;
        phaseError = 0;
        std::uint64_t errorEnergy = 0;
        for (std::size_t n = 0; n < N; ++n) {
            const std::int64_t c = basisValue(phase, false, lut) - meanC;
            const std::int64_t s = basisValue(phase, true, lut) - meanS;
            const std::int64_t modelQ8 = divideRounded(a * c + b * s, Q15_ONE);
            const std::int64_t error = residual[n] - modelQ8;
            const std::uint64_t magnitude = static_cast<std::uint64_t>(
                error < 0 ? -error : error);
            if (magnitude != 0 && magnitude >
                (std::numeric_limits<std::uint64_t>::max() - errorEnergy) / magnitude) {
                errorEnergy = std::numeric_limits<std::uint64_t>::max();
                break;
            }
            errorEnergy += magnitude * magnitude;

            phase += static_cast<std::uint32_t>(stepBase);
            phaseError += stepRemainder;
            if (phaseError >= denominator) {
                ++phase;
                phaseError -= denominator;
            }
        }
        return Fit{a, b, amplitude, true, errorEnergy};
    }

    struct PhaseStepper {
        std::uint64_t base = 0;
        std::uint64_t remainder = 0;
        std::uint64_t denominator = 1;
        std::uint64_t error = 0;
        std::uint32_t phase = 0;

        explicit PhaseStepper(std::uint32_t frequencyQ16) {
            phaseStep(frequencyQ16, base, remainder, denominator);
        }

        void advance() {
            phase += static_cast<std::uint32_t>(base); // intentional modulo-2^32 wrap
            error += remainder;
            if (error >= denominator) {
                ++phase;
                error -= denominator;
            }
        }
    };

    struct BasisRef {
        std::uint32_t frequencyQ16 = 0;
        bool sine = false;
    };

    static std::int64_t basisValue(std::uint32_t phase,
                                   bool sine,
                                   const std::array<std::int16_t, LUT_SIZE>& lut) {
        // Interpolate between adjacent Q15 table entries using the low phase
        // bits. Without this, high-resolution frequency search sees a stair-
        // stepped objective and can overfit lookup-table quantization.
        const std::uint32_t lookupPhase = sine ? phase : phase + 0x40000000U;
        const std::size_t idx = lookupPhase >> PHASE_SHIFT;
        const std::uint32_t fractionMask = (std::uint32_t{1} << PHASE_SHIFT) - 1U;
        const std::int64_t fraction = lookupPhase & fractionMask;
        const std::int64_t first = lut[idx];
        const std::int64_t next = lut[(idx + 1U) & LUT_MASK];
        const std::int64_t interpolatedDelta = divideRounded(
            (next - first) * fraction,
            static_cast<std::int64_t>(std::uint64_t{1} << PHASE_SHIFT));
        return first + interpolatedDelta;
    }

    static void basisStatistics(const InputBlock& samples,
                                const BasisRef& basis,
                                std::int64_t& sumBasis,
                                std::int64_t& sumSampleTimesBasis) {
        const auto& lut = sineTable();
        PhaseStepper phase(basis.frequencyQ16);
        sumBasis = 0;
        sumSampleTimesBasis = 0;
        for (std::size_t n = 0; n < N; ++n) {
            const std::int64_t b = basisValue(phase.phase, basis.sine, lut);
            const std::int64_t yQ8 = static_cast<std::int64_t>(samples[n]) * Q8_ONE;
            sumBasis += b;
            sumSampleTimesBasis += yQ8 * b;
            phase.advance();
        }
    }

    static std::int64_t basisProductSum(const BasisRef& a,
                                        const BasisRef& b) {
        const auto& lut = sineTable();
        PhaseStepper phaseA(a.frequencyQ16);
        PhaseStepper phaseB(b.frequencyQ16);
        std::int64_t sum = 0;
        for (std::size_t n = 0; n < N; ++n) {
            sum += basisValue(phaseA.phase, a.sine, lut) *
                   basisValue(phaseB.phase, b.sine, lut);
            phaseA.advance();
            phaseB.advance();
        }
        return sum;
    }

    // Joint least-squares solve over all selected tone frequencies. A centered
    // Gram matrix includes every pairwise sin/sin, sin/cos and cos/cos product.
    // Matrix entries are normalized Q30; coefficients/RHS are sample Q8 units.
    // Partial-pivoted Gaussian elimination is fully integer and bounded to at
    // most 32 unknowns. Nearly collinear frequencies are rejected cleanly.
    static bool jointRefit(const InputBlock& samples,
                           ToneList& tones,
                           std::size_t toneCount) {
        if (toneCount == 0 || toneCount > MAX_TONES) return false;
        const std::size_t dimensions = toneCount * 2;
        std::array<BasisRef, MAX_COEFFICIENTS> refs{};
        std::array<std::int64_t, MAX_COEFFICIENTS> sumBasis{};
        std::array<std::int64_t, MAX_COEFFICIENTS> sumYBasis{};
        std::array<std::array<std::int64_t, MAX_COEFFICIENTS>, MAX_COEFFICIENTS> matrix{};
        std::array<std::int64_t, MAX_COEFFICIENTS> rhs{};
        std::array<std::int64_t, MAX_COEFFICIENTS> solution{};

        std::int64_t sumY = 0;
        for (const std::int16_t sample : samples)
            sumY += static_cast<std::int64_t>(sample) * Q8_ONE;

        for (std::size_t i = 0; i < toneCount; ++i) {
            refs[2 * i] = BasisRef{tones[i].frequencyQ16, false};
            refs[2 * i + 1] = BasisRef{tones[i].frequencyQ16, true};
        }

        for (std::size_t i = 0; i < dimensions; ++i) {
            basisStatistics(samples, refs[i], sumBasis[i], sumYBasis[i]);
            // Remove the DC/constant column analytically by centering each
            // basis against the mean of the input and the other basis columns.
            const std::int64_t centeredY = sumYBasis[i] - divideRounded(
                sumY * sumBasis[i], static_cast<std::int64_t>(N));
            const std::int64_t meanCenteredY = divideRounded(
                centeredY, static_cast<std::int64_t>(N));
            rhs[i] = divideRounded(meanCenteredY, 32767);
        }

        constexpr std::int64_t BASIS_SQUARED = 32767LL * 32767LL;
        for (std::size_t i = 0; i < dimensions; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                const std::int64_t raw = basisProductSum(refs[i], refs[j]);
                const std::int64_t centered = raw - divideRounded(
                    sumBasis[i] * sumBasis[j], static_cast<std::int64_t>(N));
                const std::int64_t meanCovariance = divideRounded(
                    centered, static_cast<std::int64_t>(N));
                const std::int64_t normalized = divideRounded(
                    meanCovariance * ONE_Q30, BASIS_SQUARED);
                matrix[i][j] = normalized;
                matrix[j][i] = normalized;
            }
        }

        constexpr std::int64_t MIN_PIVOT_Q30 = 2048;
        for (std::size_t p = 0; p < dimensions; ++p) {
            std::size_t pivotRow = p;
            std::int64_t pivotMagnitude = 0;
            for (std::size_t r = p; r < dimensions; ++r) {
                const std::int64_t v = matrix[r][p];
                const std::int64_t magnitude = (v < 0) ? -v : v;
                if (magnitude > pivotMagnitude) {
                    pivotMagnitude = magnitude;
                    pivotRow = r;
                }
            }
            if (pivotMagnitude < MIN_PIVOT_Q30) return false;
            if (pivotRow != p) {
                std::swap(matrix[pivotRow], matrix[p]);
                std::swap(rhs[pivotRow], rhs[p]);
            }

            const std::int64_t pivot = matrix[p][p];
            for (std::size_t r = p + 1; r < dimensions; ++r) {
                if (matrix[r][p] == 0) continue;
                const std::int64_t factorQ30 = divideRounded(
                    matrix[r][p] * ONE_Q30, pivot);
                for (std::size_t c = p + 1; c < dimensions; ++c) {
                    matrix[r][c] -= divideRounded(
                        factorQ30 * matrix[p][c], ONE_Q30);
                }
                rhs[r] -= divideRounded(factorQ30 * rhs[p], ONE_Q30);
                matrix[r][p] = 0;
            }
        }

        for (std::size_t reverse = dimensions; reverse > 0; --reverse) {
            const std::size_t i = reverse - 1;
            std::int64_t value = rhs[i];
            for (std::size_t j = i + 1; j < dimensions; ++j) {
                value -= divideRounded(matrix[i][j] * solution[j], ONE_Q30);
            }
            const std::int64_t pivot = matrix[i][i];
            const std::int64_t pivotMagnitude = (pivot < 0) ? -pivot : pivot;
            if (pivotMagnitude < MIN_PIVOT_Q30 ||
                value > std::numeric_limits<std::int64_t>::max() / ONE_Q30 ||
                value < std::numeric_limits<std::int64_t>::min() / ONE_Q30)
                return false;
            solution[i] = divideRounded(value * ONE_Q30, pivot);
            if (solution[i] > std::numeric_limits<std::int32_t>::max() ||
                solution[i] < std::numeric_limits<std::int32_t>::min()) return false;
        }

        ToneList updated = tones;
        for (std::size_t i = 0; i < toneCount; ++i) {
            const std::int64_t a = solution[2 * i];
            const std::int64_t b = solution[2 * i + 1];
            const std::uint64_t aa = static_cast<std::uint64_t>(a < 0 ? -a : a);
            const std::uint64_t bb = static_cast<std::uint64_t>(b < 0 ? -b : b);
            const std::uint64_t ampQ8 = integerSqrt(aa * aa + bb * bb);
            updated[i].cosineQ8 = clampI32(a);
            updated[i].sineQ8 = clampI32(b);
            updated[i].peakAmplitude = static_cast<std::uint32_t>(
                (ampQ8 + Q8_ONE / 2) / Q8_ONE);
        }
        tones = updated;
        return true;
    }

    static bool betterFit(const Fit& candidate, const Fit& current) {
        if (!candidate.valid) return false;
        if (!current.valid) return true;
        if (candidate.residualError != current.residualError)
            return candidate.residualError < current.residualError;
        return candidate.amplitudeQ8 > current.amplitudeQ8;
    }

    static std::uint64_t residualEnergy(const ResidualBlock& residual) {
        std::uint64_t energy = 0;
        for (const std::int64_t value : residual) {
            const std::uint64_t magnitude = static_cast<std::uint64_t>(value < 0 ? -value : value);
            if (magnitude != 0 && magnitude >
                (std::numeric_limits<std::uint64_t>::max() - energy) / magnitude)
                return std::numeric_limits<std::uint64_t>::max();
            energy += magnitude * magnitude;
        }
        return energy;
    }

    static bool evaluateJointModel(const InputBlock& samples,
                                   std::int64_t dcQ8,
                                   const ToneList& candidateTones,
                                   std::size_t toneCount,
                                   ToneList& fittedTones,
                                   ResidualBlock& workResidual,
                                   std::uint64_t& error) {
        fittedTones = candidateTones;
        if (!jointRefit(samples, fittedTones, toneCount)) return false;
        for (std::size_t n = 0; n < N; ++n)
            workResidual[n] = static_cast<std::int64_t>(samples[n]) * Q8_ONE - dcQ8;
        for (std::size_t i = 0; i < toneCount; ++i)
            applyTone(workResidual, fittedTones[i], -1);
        error = residualEnergy(workResidual);
        return true;
    }

    static bool jointCoordinateFrequencyRefine(const InputBlock& samples,
                                               const std::uint32_t* grid,
                                               std::size_t gridCount,
                                               std::int64_t dcQ8,
                                               ToneList& tones,
                                               std::size_t toneCount,
                                               ResidualBlock& workResidual) {
        if (grid == nullptr || gridCount < 2 || toneCount == 0) return false;
        // Variable-projection frequency search is more expensive than the
        // coefficient solve. Use the full search budget on small/medium blocks;
        // for larger matrices the bounded single-tone coordinate pass plus the
        // joint coefficient solve remains available without multiplying the
        // O(N*T^2) Gram cost by every frequency trial.
        const bool detailedSearch = (N <= 1024 && toneCount <= 4);
        const bool mediumSearch = (N <= 2048 && toneCount <= 6);
        if (!detailedSearch && !mediumSearch) return false;
        const std::size_t searchRadiusPoints = detailedSearch ? 8U : 4U;
        const unsigned passes = detailedSearch ? 5U : 2U;
        const unsigned goldenIterations = detailedSearch ? 20U : 12U;
        constexpr std::uint64_t GOLDEN_LO_Q16 = 25033;
        constexpr std::uint64_t GOLDEN_HI_Q16 = 40503;

        struct Point {
            std::uint32_t frequencyQ16 = 0;
            std::uint64_t error = std::numeric_limits<std::uint64_t>::max();
            bool valid = false;
            ToneList fitted{};
        };

        ToneList fitted{};
        std::uint64_t currentError = 0;
        if (!evaluateJointModel(samples, dcQ8, tones, toneCount,
                                fitted, workResidual, currentError)) return false;
        tones = fitted;

        for (unsigned pass = 0; pass < passes; ++pass) {
            for (std::size_t toneIndex = 0; toneIndex < toneCount; ++toneIndex) {
                const ToneList baseTones = tones;
                Point best{};
                best.frequencyQ16 = baseTones[toneIndex].frequencyQ16;
                best.error = currentError;
                best.valid = true;
                best.fitted = baseTones;

                const std::size_t anchor = nearestGridIndex(
                    grid, gridCount, baseTones[toneIndex].frequencyQ16);
                const std::size_t begin = (anchor > searchRadiusPoints)
                    ? anchor - searchRadiusPoints : 0;
                const std::size_t end = std::min(
                    gridCount - 1, anchor + searchRadiusPoints);
                std::size_t bestAnchor = anchor;

                for (std::size_t g = begin; g <= end; ++g) {
                    ToneList trial = baseTones;
                    trial[toneIndex].frequencyQ16 = grid[g];
                    ToneList trialFit{};
                    std::uint64_t trialError = 0;
                    if (!evaluateJointModel(samples, dcQ8, trial, toneCount,
                                            trialFit, workResidual, trialError)) continue;
                    if (trialError < best.error) {
                        best.frequencyQ16 = grid[g];
                        best.error = trialError;
                        best.valid = true;
                        best.fitted = trialFit;
                        bestAnchor = g;
                    }
                }

                std::uint64_t left = (bestAnchor == 0) ? grid[0] :
                    (static_cast<std::uint64_t>(grid[bestAnchor - 1]) + grid[bestAnchor]) / 2U;
                std::uint64_t right = (bestAnchor + 1 == gridCount) ? grid[gridCount - 1] :
                    (static_cast<std::uint64_t>(grid[bestAnchor]) + grid[bestAnchor + 1]) / 2U;

                auto evaluatePoint = [&](std::uint64_t f) {
                    Point point{};
                    point.frequencyQ16 = static_cast<std::uint32_t>(f);
                    ToneList trial = baseTones;
                    trial[toneIndex].frequencyQ16 = point.frequencyQ16;
                    point.valid = evaluateJointModel(samples, dcQ8, trial, toneCount,
                                                     point.fitted, workResidual, point.error);
                    return point;
                };

                if (right > left + 2U) {
                    std::uint64_t x1 = left + ((right - left) * GOLDEN_LO_Q16 >> 16);
                    std::uint64_t x2 = left + ((right - left) * GOLDEN_HI_Q16 >> 16);
                    if (x1 == x2) ++x2;
                    Point p1 = evaluatePoint(x1);
                    Point p2 = evaluatePoint(x2);
                    auto consider = [&best](const Point& p) {
                        if (p.valid && p.error < best.error) best = p;
                    };
                    consider(p1);
                    consider(p2);
                    for (unsigned it = 0; it < goldenIterations && right > left + 2U; ++it) {
                        if (p1.valid && (!p2.valid || p1.error <= p2.error)) {
                            right = x2;
                            x2 = x1;
                            p2 = p1;
                            x1 = left + ((right - left) * GOLDEN_LO_Q16 >> 16);
                            p1 = evaluatePoint(x1);
                            consider(p1);
                        } else {
                            left = x1;
                            x1 = x2;
                            p1 = p2;
                            x2 = left + ((right - left) * GOLDEN_HI_Q16 >> 16);
                            p2 = evaluatePoint(x2);
                            consider(p2);
                        }
                        if (x1 >= x2) break;
                    }
                    consider(evaluatePoint(left + (right - left) / 2U));
                }

                if (best.valid) {
                    tones = best.fitted;
                    currentError = best.error;
                }
            }
        }

        ToneList finalFit{};
        if (!evaluateJointModel(samples, dcQ8, tones, toneCount,
                                finalFit, workResidual, currentError)) return false;
        tones = finalFit;
        return true;
    }

    static std::size_t nearestGridIndex(const std::uint32_t* grid,
                                        std::size_t count,
                                        std::uint32_t frequencyQ16) {
        std::size_t low = 0;
        std::size_t high = count;
        while (low < high) {
            const std::size_t mid = low + (high - low) / 2;
            if (grid[mid] < frequencyQ16) low = mid + 1;
            else high = mid;
        }
        if (low == 0) return 0;
        if (low >= count) return count - 1;
        const std::uint64_t lowerDistance = frequencyQ16 - grid[low - 1];
        const std::uint64_t upperDistance = grid[low] - frequencyQ16;
        return (lowerDistance <= upperDistance) ? low - 1 : low;
    }

    // Let coordinate refinement move across several neighboring grid cells.
    // This is important when greedy peeling initially models two close tones as
    // one biased peak plus a side residual: pinning each tone to its original
    // Voronoi cell would trap the coordinate solver at a false local solution.
    static Candidate refineGridNeighborhood(const ResidualBlock& residual,
                                            const std::uint32_t* grid,
                                            std::size_t count,
                                            std::uint32_t centerQ16,
                                            std::size_t radiusPoints) {
        if (grid == nullptr || count < 2) return {};
        const std::size_t anchor = nearestGridIndex(grid, count, centerQ16);
        const std::size_t begin = (anchor > radiusPoints) ? anchor - radiusPoints : 0;
        const std::size_t end = std::min(count - 1, anchor + radiusPoints);
        Candidate best{};
        for (std::size_t i = begin; i <= end; ++i) {
            const Fit fit = fitAtFrequency(residual, grid[i]);
            if (betterFit(fit, best.fit)) best = Candidate{grid[i], fit, i};
        }
        if (!best.fit.valid) return {};
        return refineGridCell(residual, grid, count, best.gridIndex);
    }

    static Candidate refineGridCell(const ResidualBlock& residual,
                                    const std::uint32_t* grid,
                                    std::size_t count,
                                    std::size_t index) {
        if (grid == nullptr || count < 2 || index >= count) return {};
        const std::uint64_t center = grid[index];
        const std::uint64_t lo = (index == 0) ? center :
            (static_cast<std::uint64_t>(grid[index - 1]) + center) / 2U;
        const std::uint64_t hi = (index + 1 == count) ? center :
            (center + static_cast<std::uint64_t>(grid[index + 1])) / 2U;
        if (hi <= lo) return {};

        // Golden-section search in this grid point's local cell. Because the
        // cell width follows the selected scale, this works for logarithmic and
        // arbitrary nonuniform grids without imposing a fixed bin-sized radius.
        constexpr std::uint64_t GOLDEN_LO_Q16 = 25033; // 0.381966 * 2^16
        constexpr std::uint64_t GOLDEN_HI_Q16 = 40503; // 0.618034 * 2^16
        std::uint64_t left = lo;
        std::uint64_t right = hi;
        std::uint64_t x1 = left + ((right - left) * GOLDEN_LO_Q16 >> 16);
        std::uint64_t x2 = left + ((right - left) * GOLDEN_HI_Q16 >> 16);
        if (x1 == x2) ++x2;
        Candidate c1{static_cast<std::uint32_t>(x1),
                     fitAtFrequency(residual, static_cast<std::uint32_t>(x1)), index};
        Candidate c2{static_cast<std::uint32_t>(x2),
                     fitAtFrequency(residual, static_cast<std::uint32_t>(x2)), index};
        Candidate best{static_cast<std::uint32_t>(center),
                       fitAtFrequency(residual, static_cast<std::uint32_t>(center)), index};
        auto consider = [&best](const Candidate& c) {
            if (betterFit(c.fit, best.fit)) best = c;
        };
        consider(c1);
        consider(c2);

        for (unsigned iteration = 0; iteration < 24 && right > left + 2; ++iteration) {
            const bool firstWins = c1.fit.valid &&
                (!c2.fit.valid || c1.fit.residualError <= c2.fit.residualError);
            if (firstWins) {
                right = x2;
                x2 = x1;
                c2 = c1;
                x1 = left + ((right - left) * GOLDEN_LO_Q16 >> 16);
                c1 = Candidate{static_cast<std::uint32_t>(x1),
                               fitAtFrequency(residual, static_cast<std::uint32_t>(x1)), index};
                consider(c1);
            } else {
                left = x1;
                x1 = x2;
                c1 = c2;
                x2 = left + ((right - left) * GOLDEN_HI_Q16 >> 16);
                c2 = Candidate{static_cast<std::uint32_t>(x2),
                               fitAtFrequency(residual, static_cast<std::uint32_t>(x2)), index};
                consider(c2);
            }
            if (x1 >= x2) break;
        }
        const std::uint32_t middle = static_cast<std::uint32_t>(left + (right - left) / 2U);
        consider(Candidate{middle, fitAtFrequency(residual, middle), index});
        return best;
    }

    // Return the strongest projection over an arbitrary ordered frequency list.
    static bool findStrongestGridResponse(const ResidualBlock& residual,
                                          const std::uint32_t* grid,
                                          std::size_t count,
                                          Candidate& best) {
        bool found = false;
        for (std::size_t i = 0; i < count; ++i) {
            const Fit fit = fitAtFrequency(residual, grid[i]);
            if (fit.valid && (!found || betterFit(fit, best.fit))) {
                best = Candidate{grid[i], fit, i};
                found = true;
            }
        }
        return found;
    }

    // Return the highest-amplitude match over both banks: integer frequencies
    // (bank A) and half-bin-offset frequencies (bank B). These interleave into
    // one uniform half-bin comb of residual cross-correlations.
    static bool findStrongestCombResponse(const ResidualBlock& residual,
                                          std::uint32_t firstBin,
                                          std::uint32_t lastBin,
                                          Candidate& best) {
        bool found = false;
        for (std::uint32_t k = firstBin; k <= lastBin; ++k) {
            const std::uint32_t f = k * 65536U;
            const Fit fit = fitAtFrequency(residual, f); // bank A
            if (fit.valid && (!found || betterFit(fit, best.fit))) {
                best = Candidate{f, fit};
                found = true;
            }
        }
        for (std::uint32_t k = firstBin; k < lastBin; ++k) {
            const std::uint32_t f = k * 65536U + 32768U;
            const Fit fit = fitAtFrequency(residual, f); // bank B, offset by 1/2 bin
            if (fit.valid && (!found || betterFit(fit, best.fit))) {
                best = Candidate{f, fit};
                found = true;
            }
        }
        return found;
    }

    static Candidate refineCandidate(const ResidualBlock& residual,
                                     std::uint32_t centerQ16,
                                     std::uint32_t firstBin,
                                     std::uint32_t lastBin,
                                     std::int32_t radiusQ16,
                                     std::int32_t stepQ16) {
        Candidate best{};
        const std::int64_t low = static_cast<std::int64_t>(firstBin) * 65536LL;
        const std::int64_t high = static_cast<std::int64_t>(lastBin) * 65536LL;
        const std::int64_t center = centerQ16;
        for (std::int64_t offset = -radiusQ16;
             offset <= radiusQ16; offset += stepQ16) {
            const std::int64_t candidateFrequency = center + offset;
            if (candidateFrequency < low || candidateFrequency > high) continue;
            const auto f = static_cast<std::uint32_t>(candidateFrequency);
            const Fit fit = fitAtFrequency(residual, f);
            if (betterFit(fit, best.fit)) {
                best = Candidate{f, fit};
            }
        }
        return best;
    }

    // Add or subtract one fitted component from residual. The cosine and sine
    // basis is centered over the finite block so the residual stays DC-free.
    static void applyTone(ResidualBlock& residual, const Tone& tone, int sign) {
        const auto& lut = sineTable();
        std::uint64_t stepBase = 0, stepRemainder = 0, denominator = 0;
        phaseStep(tone.frequencyQ16, stepBase, stepRemainder, denominator);
        std::uint64_t phaseError = 0;
        std::uint32_t phase = 0;
        std::int64_t sumC = 0, sumS = 0;

        for (std::size_t n = 0; n < N; ++n) {
            sumS += basisValue(phase, true, lut);
            sumC += basisValue(phase, false, lut);
            phase += static_cast<std::uint32_t>(stepBase);
            phaseError += stepRemainder;
            if (phaseError >= denominator) {
                ++phase;
                phaseError -= denominator;
            }
        }
        const std::int64_t meanC = divideRounded(sumC, static_cast<std::int64_t>(N));
        const std::int64_t meanS = divideRounded(sumS, static_cast<std::int64_t>(N));

        phase = 0;
        phaseError = 0;
        for (std::size_t n = 0; n < N; ++n) {
            const std::int64_t s = basisValue(phase, true, lut) - meanS;
            const std::int64_t c = basisValue(phase, false, lut) - meanC;
            const std::int64_t modelQ8 = divideRounded(
                static_cast<std::int64_t>(tone.cosineQ8) * c +
                static_cast<std::int64_t>(tone.sineQ8) * s, Q15_ONE);
            residual[n] += static_cast<std::int64_t>(sign) * modelQ8;

            phase += static_cast<std::uint32_t>(stepBase);
            phaseError += stepRemainder;
            if (phaseError >= denominator) {
                ++phase;
                phaseError -= denominator;
            }
        }
    }
};

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

    std::cout << "\nSynthetic ground truth:\n";
    for (const auto& tone : inputTones) {
        std::cout << "bin " << tone.bin << " ("
                  << (tone.bin * SAMPLE_RATE_HZ / N) << " Hz), amplitude "
                  << tone.amplitude << '\n';
    }

    // Second demonstration: deterministic additive noise, Wiener coefficient
    // shrinkage, and reconstruction of the clean tone model. Xorshift keeps
    // this test reproducible and does not require a random-number library.
    Analyzer::InputBlock noisySamples{};
    std::uint32_t rng = 0x6D2B79F5U;
    for (std::size_t n = 0; n < N; ++n) {
        const double angle = 2.0 * PI * 14.25 * static_cast<double>(n) /
                             static_cast<double>(N) + 0.20;
        const double cleanValue = 1800.0 + 12000.0 * std::sin(angle);
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        const std::int32_t noise = static_cast<std::int32_t>(rng % 3001U) - 1500;
        long rounded = std::lround(cleanValue) + noise;
        rounded = std::max(-32768L, std::min(32767L, rounded));
        noisySamples[n] = static_cast<std::int16_t>(rounded);
    }

    Analyzer::WienerOptions wienerOptions{}; // Estimate noise from fitted residual RMS.
    Analyzer::WienerReport wienerReport{};
    Analyzer::ToneList filteredTones{};
    Analyzer::ResidualBlock wienerResidual{};
    Analyzer::DenoisedBlock denoised{};
    std::size_t filteredToneCount = 0;
    std::int32_t wienerDc = 0;
    if (analyzer.decomposeGridWiener(noisySamples, frequencyGridQ16,
                                     100, wienerOptions, filteredTones,
                                     filteredToneCount, wienerResidual,
                                     wienerDc, wienerReport)) {
        Analyzer::reconstructDenoised(noisySamples, wienerResidual, denoised);
        std::cout << "\nWiener coefficient-shrinkage demonstration (one tone + noise)\n";
        std::cout << "Estimated residual noise RMS: "
                  << (static_cast<double>(wienerReport.noiseSigmaQ8) / 256.0)
                  << " samples; attenuated components: "
                  << wienerReport.tonesAttenuated << "; suppressed components: "
                  << wienerReport.tonesSuppressed << '\n';
        std::cout << "#\tfrequency_Hz\tpeak_amplitude\tWiener_gain\n";
        for (std::size_t i = 0; i < filteredToneCount; ++i) {
            const auto& t = filteredTones[i];
            std::cout << i << '\t'
                      << Analyzer::frequencyQ16ToHz(t.frequencyQ16, SAMPLE_RATE_HZ)
                      << '\t' << t.peakAmplitude << '\t'
                      << (static_cast<double>(t.wienerGainQ15) / 32768.0) << '\n';
        }
        std::cout << "Denoised block reconstructed from the selected tone model.\n";
    } else {
        std::cerr << "Wiener demonstration failed due to invalid parameters.\n";
        return 1;
    }
}
