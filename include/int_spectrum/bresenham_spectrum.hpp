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
    // halves of the block or difference recurrence.
    bool refineSingleTone(const InputBlock& samples,
                          std::uint32_t coarseBin,
                          DifferentialResult& out) const {
        if constexpr ((N % 2) != 0) return false;

        constexpr std::uint32_t NYQUIST = static_cast<std::uint32_t>(N / 2);
        if (coarseBin == 0 || coarseBin >= NYQUIST) return false;

        const auto& lut = sineTable();
        const std::uint64_t coarseFrequencyQ16 =
            static_cast<std::uint64_t>(coarseBin) * 65536ULL;

        // Candidate 1: Frequency from second-difference recurrence
        std::uint32_t recurrenceFrequencyQ16 = 0;
        bool recurrenceValid = false;
        std::int64_t recurrenceOffsetQ16 = 0;
        if (estimateFrequencyFromDifferences(samples, recurrenceFrequencyQ16)) {
            recurrenceOffsetQ16 =
                static_cast<std::int64_t>(recurrenceFrequencyQ16) -
                static_cast<std::int64_t>(coarseFrequencyQ16);
            if (recurrenceOffsetQ16 >= -32768LL && recurrenceOffsetQ16 <= 32768LL) {
                recurrenceValid = true;
            }
        }

        // Candidate 2: Half-block differential phase
        ComplexPair halves[2]{};
        correlateHalves(samples, coarseFrequencyQ16, lut, halves);
        const ComplexPair& first = halves[0];
        const ComplexPair& second = halves[1];

        bool phaseValid = false;
        std::int64_t phaseOffsetQ16 = 0;
        std::int64_t angleQ30 = 0;

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

        if (!recurrenceValid && !phaseValid) return false;

        // Evaluate candidate offsets and pick best fit
        std::int64_t bestOffsetQ16 = 0;
        std::uint32_t bestAmplitude = 0;
        bool foundBest = false;

        auto testCandidate = [&](std::int64_t candOffset) {
            std::int64_t candFreq = static_cast<std::int64_t>(coarseFrequencyQ16) + candOffset;
            std::int64_t maxFreq = static_cast<std::int64_t>(NYQUIST) * 65536LL;
            if (candFreq <= 0 || candFreq >= maxFreq) return;

            std::uint32_t amp = 0;
            if (fitSinusoidAmplitude(samples, static_cast<std::uint64_t>(candFreq), lut, amp)) {
                if (!foundBest || amp > bestAmplitude) {
                    bestAmplitude = amp;
                    bestOffsetQ16 = candOffset;
                    foundBest = true;
                }
            }
        };

        if (recurrenceValid) testCandidate(recurrenceOffsetQ16);
        if (phaseValid) testCandidate(phaseOffsetQ16);

        if (!foundBest) return false;

        const std::int64_t refinedFrequencyQ16 =
            static_cast<std::int64_t>(coarseFrequencyQ16) + bestOffsetQ16;
        const std::uint64_t strength = static_cast<std::uint64_t>(bestAmplitude) * N;

        out.coarse_bin = coarseBin;
        out.offset_q16 = static_cast<std::int32_t>(bestOffsetQ16);
        out.phase_offset_q16 = static_cast<std::int32_t>(phaseOffsetQ16);
        out.frequency_q16 = static_cast<std::uint32_t>(refinedFrequencyQ16);
        out.peak_amplitude = bestAmplitude;
        out.strength = strength;
        out.phase_difference_q30 = angleQ30;
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

        // Select the strongest detector excluding DC (0) and Nyquist (N/2)
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

    // Scale-independent CORDIC vectoring atan2(y, x) scaled to Q30 radians
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

        // Normalize (x, y) magnitude so max(|x|, |y|) is near 2^30 for full CORDIC precision
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
            if (y >= 0) z = PI_Q30;
            else z = -PI_Q30;
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
        constexpr std::uint64_t SAFE_MAX = 1073741823ULL; // 2^30 - 1
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
        // Retain sums at Q16 fractional scale
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

        // Overflow-safe conditioning check (determinant < gScale / 100)
        const std::int64_t gScale = gCC * gSS;
        if (determinant < gScale / 100LL) return false;

        const std::int64_t numeratorA = rCos * gSS - rSin * gCS;
        const std::int64_t numeratorB = rSin * gCC - rCos * gCS;
        const std::int64_t a = divideRounded(numeratorA, scaledDeterminant);
        const std::int64_t b = divideRounded(numeratorB, scaledDeterminant);

        // Checked multiplication and squaring
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
