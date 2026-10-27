#ifndef INT_SPECTRUM_BRESENHAM_SPECTRUM_HPP
#define INT_SPECTRUM_BRESENHAM_SPECTRUM_HPP

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

namespace int_spectrum {

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
    static constexpr std::size_t LUT_MASK = LUT_SIZE - 1;
    static constexpr std::uint32_t PHASE_INDEX_SHIFT = 32U - LUT_BITS;
    static constexpr std::int64_t Q15_ONE = 32767;
    static constexpr std::int64_t PI_Q30 = 3373259426LL;
    static constexpr std::int64_t ONE_Q30 = std::int64_t{1} << 30;

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

        std::uint32_t estimator_valid_mask = 0;
        std::uint32_t estimator_inlier_mask = 0;
        std::uint32_t consensus_spread_q16 = 0;
        std::uint16_t consensus_confidence_q15 = 0;
        std::uint8_t selected_estimator = 0xFFU;
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

    // Refine one dominant sinusoid using long-lag recurrence or half-block differential phase.
    bool refineSingleTone(const InputBlock& samples,
                          std::uint32_t coarseBin,
                          DifferentialResult& out) const {
        if constexpr ((N % 2) != 0) return false;

        constexpr std::uint32_t NYQUIST = static_cast<std::uint32_t>(N / 2);
        if (coarseBin == 0 || coarseBin >= NYQUIST) return false;

        const auto& lut = sineTable();
        const std::uint64_t coarseFrequencyQ16 =
            static_cast<std::uint64_t>(coarseBin) * 65536ULL;

        // Candidate 1: Half-block differential phase
        ComplexPair halves[2]{};
        correlateHalves(samples, coarseFrequencyQ16, lut, halves);
        const ComplexPair& first = halves[0];
        const ComplexPair& second = halves[1];

        std::int64_t phaseOffsetQ16 = 0;
        std::int64_t angleQ30 = 0;
        bool phaseValid = false;

        if ((first.re != 0 || first.im != 0) && (second.re != 0 || second.im != 0)) {
            std::int64_t dot = 0, cross = 0;
            if (computeScaledDotCross(first, second, dot, cross) && (dot != 0 || cross != 0)) {
                angleQ30 = atan2Q30(cross, dot);
                phaseOffsetQ16 = divideRounded(angleQ30 * 65536LL, PI_Q30);
                if (phaseOffsetQ16 >= -32768LL && phaseOffsetQ16 <= 32768LL) {
                    phaseValid = true;
                }
            }
        }

        // Candidate 2: Long-lag recurrence estimate
        std::uint32_t recurrenceFrequencyQ16 = 0;
        bool recurrenceValid = false;
        std::int64_t recurrenceOffsetQ16 = 0;

        if (estimateFrequencyFromRecurrence(samples, coarseBin, lut, recurrenceFrequencyQ16)) {
            recurrenceOffsetQ16 =
                static_cast<std::int64_t>(recurrenceFrequencyQ16) -
                static_cast<std::int64_t>(coarseFrequencyQ16);
            if (recurrenceOffsetQ16 >= -32768LL && recurrenceOffsetQ16 <= 32768LL) {
                recurrenceValid = true;
            }
        }

        std::int64_t offsetQ16 = 0;
        bool haveOffset = false;
        std::uint8_t selectedEstimator = 0xFFU;

        if (recurrenceValid) {
            offsetQ16 = recurrenceOffsetQ16;
            haveOffset = true;
            selectedEstimator = 2U; // lag_recurrence
        } else if (phaseValid) {
            offsetQ16 = phaseOffsetQ16;
            haveOffset = true;
            selectedEstimator = 0U; // half_block_phase
        }

        if (!haveOffset) return false;

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
        const std::uint64_t strength = static_cast<std::uint64_t>(amplitude) * N;

        out.coarse_bin = coarseBin;
        out.offset_q16 = static_cast<std::int32_t>(offsetQ16);
        out.phase_offset_q16 = static_cast<std::int32_t>(phaseOffsetQ16);
        out.frequency_q16 = static_cast<std::uint32_t>(refinedFrequencyQ16);
        out.peak_amplitude = amplitude;
        out.strength = strength;
        out.phase_difference_q30 = angleQ30;
        out.estimator_valid_mask = (phaseValid ? 1U : 0U) | (recurrenceValid ? 4U : 0U);
        out.estimator_inlier_mask = out.estimator_valid_mask;
        out.consensus_spread_q16 = (phaseValid && recurrenceValid) ?
            static_cast<std::uint32_t>(std::abs(phaseOffsetQ16 - recurrenceOffsetQ16)) : 0U;
        out.consensus_confidence_q15 = 32767U;
        out.selected_estimator = selectedEstimator;
        return true;
    }

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

        constexpr std::uint32_t NYQUIST = static_cast<std::uint32_t>(N / 2);
        auto best = out.end();
        std::uint64_t best_strength = 0;

        for (auto it = out.begin(); it != out.end(); ++it) {
            if (it->bin > 0 && it->bin < NYQUIST) {
                if (it->strength > best_strength || best == out.end()) {
                    best_strength = it->strength;
                    best = it;
                }
            }
        }

        if (best == out.end()) return false;
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

    static std::int64_t interpolateSine(std::uint32_t phase,
                                        bool sine,
                                        const std::array<std::int16_t, LUT_SIZE>& lut) {
        const std::uint32_t lookupPhase = sine ? phase : phase + 0x40000000U;
        const std::size_t idx = lookupPhase >> PHASE_INDEX_SHIFT;
        const std::uint32_t fractionMask = (std::uint32_t{1} << PHASE_INDEX_SHIFT) - 1U;
        const std::int64_t fraction = lookupPhase & fractionMask;
        const std::int64_t first = lut[idx];
        const std::int64_t next = lut[(idx + 1U) & LUT_MASK];
        const std::int64_t interpolatedDelta = divideRounded(
            (next - first) * fraction,
            static_cast<std::int64_t>(std::uint64_t{1} << PHASE_INDEX_SHIFT));
        return first + interpolatedDelta;
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

    static std::int64_t arithmeticShiftRight(std::int64_t value, unsigned shift) {
        if (shift == 0) return value;
        if (value >= 0) return value >> shift;
        const std::int64_t divisor = std::int64_t{1} << shift;
        return -((-value + divisor - 1) / divisor);
    }

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
        if (x == 0 && y == 0) return 0;
        if (x == 0) {
            return (y > 0) ? (PI_Q30 / 2) : (-PI_Q30 / 2);
        }

        std::uint64_t absX = static_cast<std::uint64_t>(x < 0 ? -x : x);
        std::uint64_t absY = static_cast<std::uint64_t>(y < 0 ? -y : y);
        std::uint64_t maxVal = std::max(absX, absY);

        if (maxVal > 0) {
            int shift = 0;
            constexpr std::uint64_t TARGET_MAX = std::uint64_t{1} << 30;
            if (maxVal < TARGET_MAX) {
                while (maxVal < (TARGET_MAX >> 1)) {
                    maxVal <<= 1;
                    shift++;
                }
                const std::int64_t scale = std::int64_t{1} << shift;
                x *= scale;
                y *= scale;
            } else if (maxVal > TARGET_MAX) {
                while (maxVal > TARGET_MAX) {
                    maxVal >>= 1;
                    shift++;
                }
                x = arithmeticShiftRight(x, shift);
                y = arithmeticShiftRight(y, shift);
            }
        }

        std::int64_t z = 0;
        if (x < 0) {
            z = (y >= 0) ? PI_Q30 : -PI_Q30;
            x = -x;
            y = -y;
        }

        for (unsigned i = 0; i < ATAN.size(); ++i) {
            const std::int64_t oldX = x;
            const std::int64_t oldY = y;
            if (oldY > 0) {
                x = oldX + arithmeticShiftRight(oldY, i);
                y = oldY - arithmeticShiftRight(oldX, i);
                z += ATAN[i];
            } else if (oldY < 0) {
                x = oldX - arithmeticShiftRight(oldY, i);
                y = oldY + arithmeticShiftRight(oldX, i);
                z -= ATAN[i];
            }
        }
        return z;
    }

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

        for (std::size_t n = 0; n < N; ++n) {
            const std::int32_t sinQ15 = static_cast<std::int32_t>(interpolateSine(phase, true, lut));
            const std::int32_t cosQ15 = static_cast<std::int32_t>(interpolateSine(phase, false, lut));
            const std::int64_t sample = samples[n];
            realAccQ15 += sample * cosQ15;
            imagAccQ15 -= sample * sinQ15;

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
        constexpr std::size_t HALF = N / 2;

        std::int64_t sumSamples = 0;
        for (std::size_t n = 0; n < N; ++n) {
            sumSamples += samples[n];
        }
        const std::int64_t dcMean = divideRounded(sumSamples, static_cast<std::int64_t>(N));

        for (std::size_t n = 0; n < N; ++n) {
            const std::size_t h = (n < HALF) ? 0 : 1;
            const std::int32_t sinQ15 = static_cast<std::int32_t>(interpolateSine(phase, true, lut));
            const std::int32_t cosQ15 = static_cast<std::int32_t>(interpolateSine(phase, false, lut));
            const std::int64_t sample = samples[n] - dcMean;
            realAccQ15[h] += sample * cosQ15;
            imagAccQ15[h] -= sample * sinQ15;

            phase += static_cast<std::uint32_t>(stepBase);
            phaseError += stepRemainder;
            if (phaseError >= denominator) {
                ++phase;
                phaseError -= denominator;
            }
        }

        halves[0] = {realAccQ15[0], imagAccQ15[0]};
        halves[1] = {realAccQ15[1], imagAccQ15[1]};
    }

    static bool computeScaledDotCross(const ComplexPair& first,
                                     const ComplexPair& second,
                                     std::int64_t& outDot,
                                     std::int64_t& outCross) {
        std::uint64_t maxMag = 0;
        auto updateMax = [&maxMag](std::int64_t v) {
            std::uint64_t absV = static_cast<std::uint64_t>(v < 0 ? -v : v);
            if (absV > maxMag) maxMag = absV;
        };
        updateMax(first.re);
        updateMax(first.im);
        updateMax(second.re);
        updateMax(second.im);

        int shift = 0;
        constexpr std::uint64_t SAFE_MAX = 1073741823ULL;
        while (maxMag > SAFE_MAX) {
            maxMag >>= 1;
            shift++;
        }

        std::int64_t f_re = arithmeticShiftRight(first.re, shift);
        std::int64_t f_im = arithmeticShiftRight(first.im, shift);
        std::int64_t s_re = arithmeticShiftRight(second.re, shift);
        std::int64_t s_im = arithmeticShiftRight(second.im, shift);

        outDot = f_re * s_re + f_im * s_im;
        outCross = f_re * s_im - f_im * s_re;
        return true;
    }

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

    static std::int64_t recurrenceSample(
        const InputBlock& samples,
        std::size_t n,
        bool reflectNyquist) {

        if (!reflectNyquist) {
            return samples[n];
        }

        const std::int64_t difference =
            static_cast<std::int64_t>(samples[n]) - samples[n - 1];

        return (n & 1U) ? -difference : difference;
    }

    static bool estimateFrequencyFromRecurrence(
        const InputBlock& samples,
        std::uint32_t coarseBin,
        const std::array<std::int16_t, LUT_SIZE>& lut,
        std::uint32_t& frequencyQ16) {

        constexpr std::uint32_t NYQUIST =
            static_cast<std::uint32_t>(N / 2);

        if (coarseBin == 0 || coarseBin >= NYQUIST) {
            return false;
        }

        const bool reflectNyquist = coarseBin > N / 4;
        const std::uint32_t targetBin = reflectNyquist
            ? NYQUIST - coarseBin
            : coarseBin;

        std::size_t lag =
            (N + 6U * targetBin) / (12U * targetBin);

        if (lag == 0) {
            lag = 1;
        }

        const std::size_t maxJ = reflectNyquist
            ? ((N >= 5) ? (N - 5) / 2 : 0)
            : ((N >= 4) ? (N - 4) / 2 : 0);

        const std::size_t firstJ = reflectNyquist ? 3 : 2;

        if (maxJ < firstJ) {
            return false;
        }

        std::size_t J = 0;

        if (lag >= 2 && lag + 2 <= maxJ) {
            J = lag + 2;
        } else {
            lag = 1;
            const std::size_t lastJ = std::min<std::size_t>(8, maxJ);
            std::uint64_t bestMagnitude = 0;

            for (std::size_t j = firstJ; j <= lastJ; ++j) {
                const std::uint64_t phaseNumerator =
                    (static_cast<std::uint64_t>(targetBin) * j) << 32;

                const std::uint32_t phase =
                    static_cast<std::uint32_t>(phaseNumerator / N);

                const std::int64_t cosine =
                    interpolateSine(phase, false, lut);

                const std::uint64_t magnitude =
                    static_cast<std::uint64_t>(
                        cosine < 0 ? -cosine : cosine);

                if (magnitude > bestMagnitude) {
                    bestMagnitude = magnitude;
                    J = j;
                }
            }

            if (J == 0) {
                return false;
            }
        }

        const std::size_t start = reflectNyquist ? J + 1 : J;
        const std::size_t end = N - J;

        if (end <= start || end - start < 4) {
            return false;
        }

        std::int64_t sumX = 0;
        std::int64_t sumY = 0;
        std::int64_t sumZ = 0;
        std::int64_t sumXZ = 0;
        std::int64_t sumYZ = 0;

        for (std::size_t n = start; n < end; ++n) {
            const std::int64_t x =
                recurrenceSample(samples, n, reflectNyquist);

            const std::int64_t xm =
                recurrenceSample(samples, n - lag, reflectNyquist);
            const std::int64_t xp =
                recurrenceSample(samples, n + lag, reflectNyquist);

            const std::int64_t zm =
                recurrenceSample(samples, n - J, reflectNyquist);
            const std::int64_t zp =
                recurrenceSample(samples, n + J, reflectNyquist);

            std::int64_t y = xm + xp;
            std::int64_t z = zm + zp;

            if (reflectNyquist) {
                y = divideRounded(y, 2);
                z = divideRounded(z, 2);
            }

            sumX += x;
            sumY += y;
            sumZ += z;
            sumXZ += x * z;
            sumYZ += y * z;
        }

        const std::int64_t count = static_cast<std::int64_t>(end - start);

        const std::int64_t covarianceXZ = count * sumXZ - sumX * sumZ;
        const std::int64_t covarianceYZ = count * sumYZ - sumY * sumZ;

        if (covarianceXZ == 0) {
            return false;
        }

        std::int64_t cosineNumerator = covarianceYZ;
        std::int64_t cosineDenominator = reflectNyquist ? covarianceXZ : 2 * covarianceXZ;

        if (cosineDenominator < 0) {
            cosineDenominator = -cosineDenominator;
            cosineNumerator = -cosineNumerator;
        }

        if (cosineDenominator <= 0) {
            return false;
        }

        const std::int64_t cosineQ30 = ratioQ30(cosineNumerator, cosineDenominator);
        constexpr std::int64_t ONE_Q30 = std::int64_t{1} << 30;
        const std::int64_t boundedCosine = std::max(-ONE_Q30, std::min(ONE_Q30, cosineQ30));

        const std::uint64_t sineSquared = static_cast<std::uint64_t>(
            ONE_Q30 * ONE_Q30 - boundedCosine * boundedCosine);

        const std::int64_t sineQ30 = static_cast<std::int64_t>(integerSqrt(sineSquared));

        std::int64_t angleQ30 = atan2Q30(sineQ30, boundedCosine);
        angleQ30 = divideRounded(angleQ30, static_cast<std::int64_t>(lag));

        const std::int64_t targetFrequencyQ16 = divideRounded(
            angleQ30 * static_cast<std::int64_t>(N) * 65536LL,
            2 * PI_Q30);

        const std::int64_t nyquistQ16 = static_cast<std::int64_t>(NYQUIST) * 65536LL;

        const std::int64_t absoluteFrequencyQ16 = reflectNyquist
            ? nyquistQ16 - targetFrequencyQ16
            : targetFrequencyQ16;

        if (absoluteFrequencyQ16 <= 0 || absoluteFrequencyQ16 >= nyquistQ16) {
            return false;
        }

        frequencyQ16 = static_cast<std::uint32_t>(absoluteFrequencyQ16);
        return true;
    }

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

        for (std::size_t n = 0; n < N; ++n) {
            const std::int64_t sinQ15 = interpolateSine(phase, true, lut);
            const std::int64_t cosQ15 = interpolateSine(phase, false, lut);
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

        constexpr std::int64_t Q15_SQUARED = Q15_ONE * Q15_ONE;
        const std::int64_t sumCosQ16 = divideRounded(cosSumQ15 * 65536LL, Q15_ONE);
        const std::int64_t sumSinQ16 = divideRounded(sinSumQ15 * 65536LL, Q15_ONE);

        const std::int64_t dcRCos = divideRounded(sampleSum * sumCosQ16, static_cast<std::int64_t>(N) * 65536LL);
        const std::int64_t dcRSin = divideRounded(sampleSum * sumSinQ16, static_cast<std::int64_t>(N) * 65536LL);

        const std::int64_t rCos = divideRounded(xCosQ15, Q15_ONE) - dcRCos;
        const std::int64_t rSin = divideRounded(xSinQ15, Q15_ONE) - dcRSin;

        const std::int64_t dcCC = divideRounded(sumCosQ16 * sumCosQ16, static_cast<std::int64_t>(N) * 65536LL);
        const std::int64_t dcSS = divideRounded(sumSinQ16 * sumSinQ16, static_cast<std::int64_t>(N) * 65536LL);
        const std::int64_t dcCS = divideRounded(sumCosQ16 * sumSinQ16, static_cast<std::int64_t>(N) * 65536LL);

        const std::int64_t gCC = divideRounded(cosCosQ30 * 65536LL, Q15_SQUARED) - dcCC;
        const std::int64_t gSS = divideRounded(sinSinQ30 * 65536LL, Q15_SQUARED) - dcSS;
        const std::int64_t gCS = divideRounded(cosSinQ30 * 65536LL, Q15_SQUARED) - dcCS;

        const std::int64_t determinant = gCC * gSS - gCS * gCS;
        const std::int64_t scaledDeterminant = determinant / 65536LL;

        if (gCC <= 0 || gSS <= 0 || scaledDeterminant <= 0) return false;

        const std::int64_t gScale = gCC * gSS;
        if (determinant < gScale / 100LL) return false;

        const std::int64_t numeratorA = rCos * gSS - rSin * gCS;
        const std::int64_t numeratorB = rSin * gCC - rCos * gCS;
        const std::int64_t a = divideRounded(numeratorA, scaledDeterminant);
        const std::int64_t b = divideRounded(numeratorB, scaledDeterminant);

        constexpr std::int64_t MAX_I32 = 2147483647LL;
        const std::uint64_t absA = static_cast<std::uint64_t>(a < 0 ? -a : a);
        const std::uint64_t absB = static_cast<std::uint64_t>(b < 0 ? -b : b);
        if (absA > static_cast<std::uint64_t>(MAX_I32) || absB > static_cast<std::uint64_t>(MAX_I32)) return false;

        const std::uint64_t coefficientMagnitudeSquared = absA * absA + absB * absB;
        const std::uint64_t coefficientMagnitude = integerSqrt(coefficientMagnitudeSquared);
        if (coefficientMagnitude > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) return false;

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

} // namespace int_spectrum

#endif // INT_SPECTRUM_BRESENHAM_SPECTRUM_HPP
