#ifndef INT_SPECTRUM_C_API_H
#define INT_SPECTRUM_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t frequencyQ16;  // Q16 DFT bin position
    uint32_t peakAmplitude; // input sample peak units
    int32_t cosineQ8;       // Q8 fitted cosine coeff
    int32_t sineQ8;         // Q8 fitted sine coeff
    uint8_t discoveryBank;  // 0=integer, 1=half-bin, 2=grid
    uint16_t wienerGainQ15; // 32768 = 1.0
} IntSpectrumTone;

typedef struct {
    uint32_t noiseSigmaQ8;       // 0 for auto
    uint16_t noisePowerScaleQ8;  // 256 = 1.0
    uint16_t minimumGainQ15;
    uint16_t suppressBelowGainQ15;
} IntSpectrumWienerOptions;

typedef struct {
    uint32_t noiseSigmaQ8;
    uint64_t noiseCoefficientPowerQ16;
    uint16_t tonesAttenuated;
    uint16_t tonesSuppressed;
} IntSpectrumWienerReport;

// Grid generator functions
int int_spectrum_make_linear_grid(size_t N, double minHz, double maxHz, double sampleRateHz,
                                  uint32_t* outGridQ16, size_t gridPoints);

int int_spectrum_make_log_grid(size_t N, double minHz, double maxHz, double sampleRateHz,
                               uint32_t* outGridQ16, size_t gridPoints);

double int_spectrum_bin_q16_to_hz(size_t N, uint32_t binQ16, double sampleRateHz);
uint32_t int_spectrum_hz_to_bin_q16(size_t N, double hz, double sampleRateHz);

// Main dynamic multi-tone decomposition function supporting standard block sizes
// N in {128, 256, 512, 1024, 2048, 4096}, maxTones <= 16
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
    const IntSpectrumWienerOptions* wienerOpt, // NULL if unused
    IntSpectrumWienerReport* wienerReport     // NULL if unused
);

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
);

void int_spectrum_reconstruct_denoised(
    size_t N,
    const int16_t* samples,
    const int64_t* residualQ8,
    int16_t* outDenoised
);

uint64_t int_spectrum_residual_rms_q8(size_t N, const int64_t* residualQ8);

#ifdef __cplusplus
}
#endif

#endif // INT_SPECTRUM_C_API_H
