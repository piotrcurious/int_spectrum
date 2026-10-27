#ifndef INT_SPECTRUM_MULTI_TONE_COMB_HPP
#define INT_SPECTRUM_MULTI_TONE_COMB_HPP

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace int_spectrum {

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

    static std::uint64_t magnitude64(std::int64_t value) {
        return value < 0
            ? static_cast<std::uint64_t>(-(value + 1)) + 1ULL
            : static_cast<std::uint64_t>(value);
    }

    static std::uint64_t residualRmsQ8(const ResidualBlock& residual) {
        std::uint64_t energy = 0;
        for (const std::int64_t x : residual) {
            const std::uint64_t mag = magnitude64(x);
            if (mag != 0 && mag > (std::numeric_limits<std::uint64_t>::max() - energy) / mag) {
                energy = std::numeric_limits<std::uint64_t>::max();
                break;
            }
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
        constexpr std::uint64_t MAX_SIGMA_Q8 = 32768ULL * 256ULL; // 8,388,608 Q8 units

        if (toneCount > MAX_TONES || options.noisePowerScaleQ8 == 0 ||
            options.noisePowerScaleQ8 > 4096 ||
            options.minimumGainQ15 > 32768 ||
            options.suppressBelowGainQ15 > 32768 ||
            options.noiseSigmaQ8 > MAX_SIGMA_Q8) return false;

        const std::uint64_t estimatedSigmaQ8 =
            options.noiseSigmaQ8 == 0 ? residualRmsQ8(residual) : options.noiseSigmaQ8;

        if (estimatedSigmaQ8 > MAX_SIGMA_Q8) return false;
        report.noiseSigmaQ8 = static_cast<std::uint32_t>(estimatedSigmaQ8);

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
            if (keptCount < MAX_TONES) {
                keptTones[keptCount++] = tone;
            }
        }

        tones = keptTones;
        toneCount = keptCount;

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

        bool jointSolved = (toneCount == 0);
        if (toneCount != 0) jointSolved = jointRefit(samples, tones, toneCount);
        if (jointSolved && toneCount != 0) {
            ToneList pruned{};
            std::size_t kept = 0;
            for (std::size_t i = 0; i < toneCount; ++i) {
                if (tones[i].peakAmplitude >= minimumAmplitude) {
                    if (kept < MAX_TONES) {
                        pruned[kept++] = tones[i];
                    }
                }
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

        for (std::size_t n = 0; n < N; ++n) {
            residual[n] = static_cast<std::int64_t>(samples[n]) * Q8_ONE - dcQ8;
        }
        for (std::size_t i = 0; i < toneCount; ++i)
            applyTone(residual, tones[i], -1);

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

    static std::int64_t divideSigned(std::int64_t x, std::int64_t d) {
        if (d == 0) return 0;
        if (d < 0) {
            if (d == std::numeric_limits<std::int64_t>::min()) return 0;
            x = -x;
            d = -d;
        }
        if (x >= 0) return (x + d / 2) / d;
        return -((-x + d / 2) / d);
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

            phase += static_cast<std::uint32_t>(stepBase);
            phaseError += stepRemainder;
            if (phaseError >= denominator) {
                ++phase;
                phaseError -= denominator;
            }
        }

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

        const std::int64_t gScale = gCC * gSS;
        if (gScale <= 0 || determinant < gScale / 100LL) return {};

        const std::int64_t a = divideRounded(xC * gSS - xS * gCS, determinant);
        const std::int64_t b = divideRounded(xS * gCC - xC * gCS, determinant);
        const std::uint64_t aa = magnitude64(a);
        const std::uint64_t bb = magnitude64(b);
        if (aa > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
            bb > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) return {};
        const std::uint64_t amplitude = integerSqrt(aa * aa + bb * bb);

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
            const std::uint64_t magnitude = magnitude64(error);
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
            phase += static_cast<std::uint32_t>(base);
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
                const std::int64_t factorQ30 = divideSigned(
                    matrix[r][p] * ONE_Q30, pivot);
                for (std::size_t c = p + 1; c < dimensions; ++c) {
                    matrix[r][c] -= divideSigned(
                        factorQ30 * matrix[p][c], ONE_Q30);
                }
                rhs[r] -= divideSigned(factorQ30 * rhs[p], ONE_Q30);
                matrix[r][p] = 0;
            }
        }

        for (std::size_t reverse = dimensions; reverse > 0; --reverse) {
            const std::size_t i = reverse - 1;
            std::int64_t value = rhs[i];
            for (std::size_t j = i + 1; j < dimensions; ++j) {
                value -= divideSigned(matrix[i][j] * solution[j], ONE_Q30);
            }
            const std::int64_t pivot = matrix[i][i];
            const std::int64_t pivotMagnitude = (pivot < 0) ? -pivot : pivot;
            if (pivotMagnitude < MIN_PIVOT_Q30 ||
                value > std::numeric_limits<std::int64_t>::max() / ONE_Q30 ||
                value < std::numeric_limits<std::int64_t>::min() / ONE_Q30)
                return false;
            solution[i] = divideSigned(value * ONE_Q30, pivot);
            if (solution[i] > std::numeric_limits<std::int32_t>::max() ||
                solution[i] < std::numeric_limits<std::int32_t>::min()) return false;
        }

        ToneList updated = tones;
        for (std::size_t i = 0; i < toneCount; ++i) {
            const std::int64_t a = solution[2 * i];
            const std::int64_t b = solution[2 * i + 1];
            const std::uint64_t aa = magnitude64(a);
            const std::uint64_t bb = magnitude64(b);
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
            const std::uint64_t magnitude = magnitude64(value);
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

        constexpr std::uint64_t GOLDEN_LO_Q16 = 25033;
        constexpr std::uint64_t GOLDEN_HI_Q16 = 40503;
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

    static bool findStrongestCombResponse(const ResidualBlock& residual,
                                          std::uint32_t firstBin,
                                          std::uint32_t lastBin,
                                          Candidate& best) {
        bool found = false;
        for (std::uint32_t k = firstBin; k <= lastBin; ++k) {
            const std::uint32_t f = k * 65536U;
            const Fit fit = fitAtFrequency(residual, f);
            if (fit.valid && (!found || betterFit(fit, best.fit))) {
                best = Candidate{f, fit};
                found = true;
            }
        }
        for (std::uint32_t k = firstBin; k < lastBin; ++k) {
            const std::uint32_t f = k * 65536U + 32768U;
            const Fit fit = fitAtFrequency(residual, f);
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

} // namespace int_spectrum

#endif // INT_SPECTRUM_MULTI_TONE_COMB_HPP
