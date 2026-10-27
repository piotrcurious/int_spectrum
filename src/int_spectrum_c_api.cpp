#include "int_spectrum/c_api.h"
#include "int_spectrum/multi_tone_comb.hpp"
#include "int_spectrum/bresenham_spectrum.hpp"

#include <vector>
#include <cmath>
#include <cstring>

using namespace int_spectrum;

template <size_t N>
static int fuse_estimates_dispatch(
    const int16_t* samples,
    uint32_t coarseBin,
    IntSpectrumFusionResult* outFusion
) {
    using Analyzer = BresenhamSpectrum<N, 16, 12>;
    Analyzer analyzer;
    typename Analyzer::InputBlock in_block;
    std::memcpy(in_block.data(), samples, sizeof(int16_t) * N);

    typename Analyzer::DifferentialResult diff_res;
    bool ok = analyzer.refineSingleTone(in_block, coarseBin, diff_res);
    if (!ok) return 0;

    outFusion->coarseBin = diff_res.coarse_bin;
    outFusion->offsetQ16 = diff_res.offset_q16;
    outFusion->phaseOffsetQ16 = diff_res.phase_offset_q16;
    outFusion->frequencyQ16 = diff_res.frequency_q16;
    outFusion->peakAmplitude = diff_res.peak_amplitude;
    outFusion->strength = diff_res.strength;
    outFusion->phaseDifferenceQ30 = diff_res.phase_difference_q30;
    outFusion->estimatorValidMask = diff_res.estimator_valid_mask;
    outFusion->estimatorInlierMask = diff_res.estimator_inlier_mask;
    outFusion->consensusSpreadQ16 = diff_res.consensus_spread_q16;
    outFusion->consensusConfidenceQ15 = diff_res.consensus_confidence_q15;
    outFusion->selectedEstimator = diff_res.selected_estimator;
    return 1;
}

template <size_t N>
static int decompose_grid_dispatch(
    const int16_t* samples,
    const uint32_t* gridQ16,
    size_t gridPoints,
    uint32_t minimumAmplitude,
    size_t maxTones,
    IntSpectrumTone* outTones,
    size_t* outToneCount,
    int64_t* outResidualQ8,
    int32_t* outRemovedDc,
    const IntSpectrumWienerOptions* wienerOpt,
    IntSpectrumWienerReport* wienerReport
) {
    if (maxTones > 16 || maxTones == 0) return 0;

    auto run_decomposition = [&](auto tone_tag) {
        constexpr size_t MAX_T = decltype(tone_tag)::value;
        using Analyzer = MultiToneComb<N, MAX_T, 12>;
        Analyzer analyzer;
        typename Analyzer::InputBlock in_block;
        std::memcpy(in_block.data(), samples, sizeof(int16_t) * N);

        typename Analyzer::ToneList tones;
        typename Analyzer::ResidualBlock residual;
        size_t toneCount = 0;
        int32_t dc = 0;
        bool ok = false;

        if (wienerOpt) {
            typename Analyzer::WienerOptions w_opt;
            w_opt.noiseSigmaQ8 = wienerOpt->noiseSigmaQ8;
            w_opt.noisePowerScaleQ8 = wienerOpt->noisePowerScaleQ8;
            w_opt.minimumGainQ15 = wienerOpt->minimumGainQ15;
            w_opt.suppressBelowGainQ15 = wienerOpt->suppressBelowGainQ15;

            typename Analyzer::WienerReport w_rep;
            ok = analyzer.decomposeFrequenciesWiener(
                in_block, gridQ16, gridPoints, minimumAmplitude,
                w_opt, tones, toneCount, residual, dc, w_rep);

            if (wienerReport) {
                wienerReport->noiseSigmaQ8 = w_rep.noiseSigmaQ8;
                wienerReport->noiseCoefficientPowerQ16 = w_rep.noiseCoefficientPowerQ16;
                wienerReport->tonesAttenuated = w_rep.tonesAttenuated;
                wienerReport->tonesSuppressed = w_rep.tonesSuppressed;
            }
        } else {
            ok = analyzer.decomposeFrequencies(
                in_block, gridQ16, gridPoints, minimumAmplitude,
                tones, toneCount, residual, dc);
        }

        if (!ok) return 0;

        *outToneCount = toneCount;
        *outRemovedDc = dc;
        if (outResidualQ8) {
            std::memcpy(outResidualQ8, residual.data(), sizeof(int64_t) * N);
        }

        for (size_t i = 0; i < toneCount; ++i) {
            outTones[i].frequencyQ16 = tones[i].frequencyQ16;
            outTones[i].peakAmplitude = tones[i].peakAmplitude;
            outTones[i].cosineQ8 = tones[i].cosineQ8;
            outTones[i].sineQ8 = tones[i].sineQ8;
            outTones[i].discoveryBank = tones[i].discoveryBank;
            outTones[i].wienerGainQ15 = tones[i].wienerGainQ15;
        }

        return 1;
    };

    if (maxTones <= 1) return run_decomposition(std::integral_constant<size_t, 1>{});
    if (maxTones <= 2) return run_decomposition(std::integral_constant<size_t, 2>{});
    if (maxTones <= 3) return run_decomposition(std::integral_constant<size_t, 3>{});
    if (maxTones <= 4) return run_decomposition(std::integral_constant<size_t, 4>{});
    if (maxTones <= 5) return run_decomposition(std::integral_constant<size_t, 5>{});
    if (maxTones <= 8) return run_decomposition(std::integral_constant<size_t, 8>{});
    if (maxTones <= 12) return run_decomposition(std::integral_constant<size_t, 12>{});
    return run_decomposition(std::integral_constant<size_t, 16>{});
}

template <size_t N>
static int decompose_comb_dispatch(
    const int16_t* samples,
    uint32_t firstBin,
    uint32_t lastBin,
    uint32_t minimumAmplitude,
    size_t maxTones,
    IntSpectrumTone* outTones,
    size_t* outToneCount,
    int64_t* outResidualQ8,
    int32_t* outRemovedDc
) {
    if (maxTones > 16 || maxTones == 0) return 0;

    auto run_comb = [&](auto tone_tag) {
        constexpr size_t MAX_T = decltype(tone_tag)::value;
        using Analyzer = MultiToneComb<N, MAX_T, 12>;
        Analyzer analyzer;
        typename Analyzer::InputBlock in_block;
        std::memcpy(in_block.data(), samples, sizeof(int16_t) * N);

        typename Analyzer::ToneList tones;
        typename Analyzer::ResidualBlock residual;
        size_t toneCount = 0;
        int32_t dc = 0;

        bool ok = analyzer.decompose(in_block, firstBin, lastBin, minimumAmplitude,
                                     tones, toneCount, residual, dc);
        if (!ok) return 0;

        *outToneCount = toneCount;
        *outRemovedDc = dc;
        if (outResidualQ8) {
            std::memcpy(outResidualQ8, residual.data(), sizeof(int64_t) * N);
        }

        for (size_t i = 0; i < toneCount; ++i) {
            outTones[i].frequencyQ16 = tones[i].frequencyQ16;
            outTones[i].peakAmplitude = tones[i].peakAmplitude;
            outTones[i].cosineQ8 = tones[i].cosineQ8;
            outTones[i].sineQ8 = tones[i].sineQ8;
            outTones[i].discoveryBank = tones[i].discoveryBank;
            outTones[i].wienerGainQ15 = tones[i].wienerGainQ15;
        }

        return 1;
    };

    if (maxTones <= 1) return run_comb(std::integral_constant<size_t, 1>{});
    if (maxTones <= 2) return run_comb(std::integral_constant<size_t, 2>{});
    if (maxTones <= 3) return run_comb(std::integral_constant<size_t, 3>{});
    if (maxTones <= 4) return run_comb(std::integral_constant<size_t, 4>{});
    if (maxTones <= 5) return run_comb(std::integral_constant<size_t, 5>{});
    if (maxTones <= 8) return run_comb(std::integral_constant<size_t, 8>{});
    if (maxTones <= 12) return run_comb(std::integral_constant<size_t, 12>{});
    return run_comb(std::integral_constant<size_t, 16>{});
}

extern "C" {

int int_spectrum_make_linear_grid(size_t N, double minHz, double maxHz, double sampleRateHz,
                                  uint32_t* outGridQ16, size_t gridPoints) {
    if (gridPoints < 2 || !outGridQ16) return 0;
    if (!(sampleRateHz > 0.0) || !(minHz > 0.0) || !(maxHz > minHz) || !(maxHz < sampleRateHz * 0.5))
        return 0;

    double scale = static_cast<double>(N) * 65536.0 / sampleRateHz;
    uint64_t first = static_cast<uint64_t>(std::llround(minHz * scale));
    uint64_t last = static_cast<uint64_t>(std::llround(maxHz * scale));
    if (last <= first || last - first < gridPoints - 1) return 0;

    uint64_t intervals = gridPoints - 1;
    uint64_t span = last - first;
    uint64_t step = span / intervals;
    uint64_t remainder = span % intervals;
    uint64_t f = first;
    uint64_t error = 0;

    for (size_t i = 0; i < gridPoints; ++i) {
        outGridQ16[i] = static_cast<uint32_t>(f);
        if (i + 1 < gridPoints) {
            f += step;
            error += remainder;
            if (error >= intervals) {
                ++f;
                error -= intervals;
            }
        }
    }
    return 1;
}

int int_spectrum_make_log_grid(size_t N, double minHz, double maxHz, double sampleRateHz,
                               uint32_t* outGridQ16, size_t gridPoints) {
    if (gridPoints < 2 || !outGridQ16) return 0;
    if (!(sampleRateHz > 0.0) || !(minHz > 0.0) || !(maxHz > minHz) || !(maxHz < sampleRateHz * 0.5))
        return 0;

    double logMin = std::log(minHz);
    double logSpan = std::log(maxHz) - logMin;
    double scale = static_cast<double>(N) * 65536.0 / sampleRateHz;
    uint32_t previous = 0;

    for (size_t i = 0; i < gridPoints; ++i) {
        double fraction = static_cast<double>(i) / static_cast<double>(gridPoints - 1);
        double hz = std::exp(logMin + fraction * logSpan);
        double binQ16D = hz * scale;
        uint32_t binQ16 = static_cast<uint32_t>(std::llround(binQ16D));
        if (i != 0 && binQ16 <= previous) return 0;
        outGridQ16[i] = binQ16;
        previous = binQ16;
    }
    return 1;
}

double int_spectrum_bin_q16_to_hz(size_t N, uint32_t binQ16, double sampleRateHz) {
    return static_cast<double>(binQ16) * sampleRateHz / (static_cast<double>(N) * 65536.0);
}

uint32_t int_spectrum_hz_to_bin_q16(size_t N, double hz, double sampleRateHz) {
    if (sampleRateHz <= 0.0 || hz <= 0.0) return 0;
    double val = hz * static_cast<double>(N) * 65536.0 / sampleRateHz;
    return static_cast<uint32_t>(std::llround(val));
}

int int_spectrum_decompose_grid(
    size_t N,
    const int16_t* samples,
    const uint32_t* gridQ16,
    size_t gridPoints,
    uint32_t minimumAmplitude,
    size_t maxTones,
    IntSpectrumTone* outTones,
    size_t* outToneCount,
    int64_t* outResidualQ8,
    int32_t* outRemovedDc,
    const IntSpectrumWienerOptions* wienerOpt,
    IntSpectrumWienerReport* wienerReport
) {
    if (!samples || !gridQ16 || !outTones || !outToneCount || !outRemovedDc) return 0;

    switch (N) {
        case 64: return decompose_grid_dispatch<64>(samples, gridQ16, gridPoints, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc, wienerOpt, wienerReport);
        case 128: return decompose_grid_dispatch<128>(samples, gridQ16, gridPoints, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc, wienerOpt, wienerReport);
        case 256: return decompose_grid_dispatch<256>(samples, gridQ16, gridPoints, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc, wienerOpt, wienerReport);
        case 512: return decompose_grid_dispatch<512>(samples, gridQ16, gridPoints, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc, wienerOpt, wienerReport);
        case 1024: return decompose_grid_dispatch<1024>(samples, gridQ16, gridPoints, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc, wienerOpt, wienerReport);
        case 2048: return decompose_grid_dispatch<2048>(samples, gridQ16, gridPoints, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc, wienerOpt, wienerReport);
        case 4096: return decompose_grid_dispatch<4096>(samples, gridQ16, gridPoints, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc, wienerOpt, wienerReport);
        default: return 0;
    }
}

int int_spectrum_decompose_comb(
    size_t N,
    const int16_t* samples,
    uint32_t firstBin,
    uint32_t lastBin,
    uint32_t minimumAmplitude,
    size_t maxTones,
    IntSpectrumTone* outTones,
    size_t* outToneCount,
    int64_t* outResidualQ8,
    int32_t* outRemovedDc
) {
    if (!samples || !outTones || !outToneCount || !outRemovedDc) return 0;

    switch (N) {
        case 64: return decompose_comb_dispatch<64>(samples, firstBin, lastBin, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc);
        case 128: return decompose_comb_dispatch<128>(samples, firstBin, lastBin, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc);
        case 256: return decompose_comb_dispatch<256>(samples, firstBin, lastBin, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc);
        case 512: return decompose_comb_dispatch<512>(samples, firstBin, lastBin, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc);
        case 1024: return decompose_comb_dispatch<1024>(samples, firstBin, lastBin, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc);
        case 2048: return decompose_comb_dispatch<2048>(samples, firstBin, lastBin, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc);
        case 4096: return decompose_comb_dispatch<4096>(samples, firstBin, lastBin, minimumAmplitude, maxTones, outTones, outToneCount, outResidualQ8, outRemovedDc);
        default: return 0;
    }
}

int int_spectrum_fuse_estimates(
    size_t N,
    const int16_t* samples,
    uint32_t coarseBin,
    IntSpectrumFusionResult* outFusion
) {
    if (!samples || !outFusion) return 0;

    switch (N) {
        case 64: return fuse_estimates_dispatch<64>(samples, coarseBin, outFusion);
        case 128: return fuse_estimates_dispatch<128>(samples, coarseBin, outFusion);
        case 256: return fuse_estimates_dispatch<256>(samples, coarseBin, outFusion);
        case 512: return fuse_estimates_dispatch<512>(samples, coarseBin, outFusion);
        case 1024: return fuse_estimates_dispatch<1024>(samples, coarseBin, outFusion);
        case 2048: return fuse_estimates_dispatch<2048>(samples, coarseBin, outFusion);
        case 4096: return fuse_estimates_dispatch<4096>(samples, coarseBin, outFusion);
        default: return 0;
    }
}

void int_spectrum_reconstruct_denoised(
    size_t N,
    const int16_t* samples,
    const int64_t* residualQ8,
    int16_t* outDenoised
) {
    if (!samples || !residualQ8 || !outDenoised) return;
    for (size_t n = 0; n < N; ++n) {
        int64_t modelQ8 = static_cast<int64_t>(samples[n]) * 256LL - residualQ8[n];
        int64_t val = (modelQ8 >= 0) ? (modelQ8 + 128LL) / 256LL : -((-modelQ8 + 128LL) / 256LL);
        if (val > 32767) val = 32767;
        if (val < -32768) val = -32768;
        outDenoised[n] = static_cast<int16_t>(val);
    }
}

uint64_t int_spectrum_residual_rms_q8(size_t N, const int64_t* residualQ8) {
    if (!residualQ8 || N == 0) return 0;
    uint64_t energy = 0;
    for (size_t i = 0; i < N; ++i) {
        int64_t x = residualQ8[i];
        uint64_t mag = MultiToneComb<64, 1>::magnitude64(x);
        if (mag != 0 && mag > (std::numeric_limits<uint64_t>::max() - energy) / mag) {
            energy = std::numeric_limits<uint64_t>::max();
            break;
        }
        energy += mag * mag;
    }
    uint64_t val = energy / N;
    uint64_t root = 0;
    uint64_t bit = uint64_t{1} << 62;
    while (bit > val) bit >>= 2;
    while (bit != 0) {
        if (val >= root + bit) {
            val -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }
    return root;
}

} // extern "C"
