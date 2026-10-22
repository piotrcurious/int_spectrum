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

public:
    using InputBlock = std::array<std::int16_t, N>;
    using ResidualBlock = std::array<std::int64_t, N>; // Q8 sample units

    struct Tone {
        std::uint32_t frequencyQ16 = 0; // bin position, Q16
        std::uint32_t peakAmplitude = 0; // original sample units
        std::int32_t cosineQ8 = 0; // fitted coefficient, Q8 sample units
        std::int32_t sineQ8 = 0;   // fitted coefficient, Q8 sample units
        std::uint8_t discoveryBank = 0; // 0=integer; 1=half-bin bank
    };

    using ToneList = std::array<Tone, MAX_TONES>;

    static void prepare() { (void)sineTable(); }

    // Decompose the AC component of samples over [firstBin, lastBin]. DC is
    // removed first and reported separately through removedDc. The caller
    // supplies the fixed-size residual buffer, so no heap allocation occurs.
    // minimumAmplitude is a stopping threshold in input sample units.
    bool decompose(const InputBlock& samples,
                   std::uint32_t firstBin,
                   std::uint32_t lastBin,
                   std::uint32_t minimumAmplitude,
                   ToneList& tones,
                   std::size_t& toneCount,
                   ResidualBlock& residual,
                   std::int32_t& removedDc) const {
        constexpr std::uint32_t NYQUIST = static_cast<std::uint32_t>(N / 2);
        if (firstBin < 1 || firstBin >= lastBin || lastBin >= NYQUIST)
            return false;

        std::int64_t sampleSum = 0;
        for (const std::int16_t x : samples) sampleSum += x;
        const std::int64_t dcQ8 = divideRounded(sampleSum * Q8_ONE,
                                                static_cast<std::int64_t>(N));
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
            if (!findStrongestCombResponse(residual, firstBin, lastBin, best))
                break;
            if (best.fit.amplitudeQ8 < minimumAmplitudeQ8) break;

            // The coarse bank comb has half-bin spacing. Refine around its
            // winning tooth with 1/256-bin steps, bounded to +/- 1/4 bin.
            // This is local frequency fitting, not extra FFT zero-padding.
            Candidate refined = refineCandidate(
                residual, best.frequencyQ16, firstBin, lastBin, 16384, 256);
            if (refined.fit.amplitudeQ8 < minimumAmplitudeQ8) break;

            Tone t{};
            t.frequencyQ16 = refined.frequencyQ16;
            t.cosineQ8 = clampI32(refined.fit.cosineQ8);
            t.sineQ8 = clampI32(refined.fit.sineQ8);
            t.peakAmplitude = static_cast<std::uint32_t>(
                (refined.fit.amplitudeQ8 + Q8_ONE / 2) / Q8_ONE);
            t.discoveryBank = static_cast<std::uint8_t>(
                ((best.frequencyQ16 & 0xFFFFU) == 32768U) ? 1U : 0U);

            tones[toneCount++] = t;
            applyTone(residual, t, -1); // spectral peeling in the time domain
        }

        // Coordinate-descent refit: add one existing component back, retune it
        // locally against the residual of all the other components, then
        // subtract its updated fit. This reduces errors from tone-to-tone
        // correlation without forming a large floating-point normal matrix.
        for (unsigned pass = 0; pass < 2; ++pass) {
            for (std::size_t i = 0; i < toneCount; ++i) {
                const Tone old = tones[i];
                applyTone(residual, old, +1);
                const Candidate update = refineCandidate(
                    residual, old.frequencyQ16, firstBin, lastBin, 8192, 256);
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

        // Return strongest component first for convenient reporting.
        // Tiny bounded insertion sort avoids the general-purpose sort's
        // implementation-dependent temporary threshold and stays allocation-free.
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

    static std::uint64_t residualRmsQ8(const ResidualBlock& residual) {
        std::uint64_t energy = 0;
        for (const std::int64_t x : residual) {
            const std::uint64_t mag = static_cast<std::uint64_t>(x < 0 ? -x : x);
            energy += mag * mag;
        }
        return integerSqrt(energy / N);
    }

private:
    struct Fit {
        std::int64_t cosineQ8 = 0;
        std::int64_t sineQ8 = 0;
        std::uint64_t amplitudeQ8 = 0;
        bool valid = false;
    };

    struct Candidate {
        std::uint32_t frequencyQ16 = 0;
        Fit fit{};
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
        constexpr std::size_t QUARTER = LUT_SIZE / 4;

        for (std::size_t n = 0; n < N; ++n) {
            const std::size_t idx = phase >> PHASE_SHIFT;
            const std::int64_t s = lut[idx];
            const std::int64_t c = lut[(idx + QUARTER) & LUT_MASK];
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
        const std::uint64_t amplitude = integerSqrt(aa * aa + bb * bb);
        return Fit{a, b, amplitude, true};
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
            if (fit.valid && (!found || fit.amplitudeQ8 > best.fit.amplitudeQ8)) {
                best = Candidate{f, fit};
                found = true;
            }
        }
        for (std::uint32_t k = firstBin; k < lastBin; ++k) {
            const std::uint32_t f = k * 65536U + 32768U;
            const Fit fit = fitAtFrequency(residual, f); // bank B, offset by 1/2 bin
            if (fit.valid && (!found || fit.amplitudeQ8 > best.fit.amplitudeQ8)) {
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
            if (fit.valid && (!best.fit.valid ||
                              fit.amplitudeQ8 > best.fit.amplitudeQ8)) {
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
        constexpr std::size_t QUARTER = LUT_SIZE / 4;

        for (std::size_t n = 0; n < N; ++n) {
            const std::size_t idx = phase >> PHASE_SHIFT;
            sumS += lut[idx];
            sumC += lut[(idx + QUARTER) & LUT_MASK];
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
            const std::size_t idx = phase >> PHASE_SHIFT;
            const std::int64_t s = static_cast<std::int64_t>(lut[idx]) - meanS;
            const std::int64_t c = static_cast<std::int64_t>(lut[(idx + QUARTER) & LUT_MASK]) - meanC;
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
    constexpr std::uint32_t FIRST_BIN = 8;
    constexpr std::uint32_t LAST_BIN = 40;
    constexpr double PI = 3.1415926535897932384626433832795;

    struct TestTone { double bin; double amplitude; double phase; };
    constexpr std::array<TestTone, 3> inputTones{{
        {14.25, 12000.0, 0.20},
        {27.65,  6500.0, 1.10},
        {33.10,  3500.0, 2.00}
    }};

    using Analyzer = MultiToneComb<N, MAX_TONES, 12>;
    Analyzer::prepare();
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
    if (!analyzer.decompose(samples, FIRST_BIN, LAST_BIN,
                            MINIMUM_AMPLITUDE, tones, toneCount,
                            residual, removedDc)) {
        std::cerr << "Invalid frequency range or decomposition parameters.\n";
        return 1;
    }

    const std::uint64_t rmsQ8 = Analyzer::residualRmsQ8(residual);
    std::cout << "Two-bank Bresenham comb / iterative multi-tone decomposition\n";
    std::cout << "N=" << N << ", Fs=" << SAMPLE_RATE_HZ
              << " Hz, bins=" << FIRST_BIN << ".." << LAST_BIN
              << ", removed DC=" << removedDc << " samples\n";
    std::cout << "Detected " << toneCount << " tones; residual RMS ~"
              << (static_cast<double>(rmsQ8) / 256.0) << " samples\n\n";
    std::cout << "#\tbank\tfrequency_bin\tfrequency_Hz\tpeak_amplitude\n";
    for (std::size_t i = 0; i < toneCount; ++i) {
        const auto& t = tones[i];
        const double bin = static_cast<double>(t.frequencyQ16) / 65536.0;
        std::cout << i << '\t' << (t.discoveryBank == 0 ? "integer" : "half-bin")
                  << '\t' << bin << '\t' << (bin * SAMPLE_RATE_HZ / N)
                  << '\t' << t.peakAmplitude << '\n';
    }

    std::cout << "\nSynthetic ground truth:\n";
    for (const auto& tone : inputTones) {
        std::cout << "bin " << tone.bin << " ("
                  << (tone.bin * SAMPLE_RATE_HZ / N) << " Hz), amplitude "
                  << tone.amplitude << '\n';
    }
}
