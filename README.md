# int_spectrum
dsp frequency discriminator using crude integer methods
# Bresenham Spectrum — fixed-point multi-tone decomposition

A C++17 experimental spectral analyser that uses integer phase accumulators, Bresenham remainder distribution, configurable frequency grids, and iterative multi-tone fitting. The current translation unit contains both the implementation and a runnable synthetic demonstration.

The main use case is a bounded-memory embedded analyser where a conventional FFT is unnecessary or undesirable, and the frequencies of interest may be nonuniformly spaced—for example, a logarithmic grid with denser frequency points at the low end.

> **Status:** research/prototype code. The bundled synthetic test demonstrates one well-conditioned three-tone case; it is not a substitute for a full numerical validation suite or target-device benchmarking.

## Features

- **Integer modular phase:** 32-bit unsigned phase wraps modulo `2^32`, representing one full turn. `pi` is half a turn. Frequency increments are represented as a quotient plus remainder and scheduled with a Bresenham accumulator.
- **Fixed-point correlations:** Q15 lookup-table basis functions and Q8 residual/model sample units. The sample-loop correlation and coefficient-fitting path uses integer arithmetic.
- **Two-bank comb scan:** an integer-bin bank and a half-bin-offset bank form a half-bin detector comb for the built-in linear-bin mode.
- **Custom frequency grids:** sorted caller-supplied frequencies, expressed as Q16 DFT-bin positions.
- **Grid generators:** linear-Hz and logarithmic-Hz grids; log/exponential arithmetic is used during setup only.
- **Multi-tone peeling:** select a strong candidate, fit a sinusoid, subtract it from the residual, and repeat up to a fixed maximum number of tones.
- **Frequency refinement:** refine each selected candidate within its local grid region or around the half-bin comb candidate.
- **Joint coefficient fit:** construct a centred Gram system for all selected sine/cosine basis functions, then solve it using bounded, pivoted fixed-point Gaussian elimination. Poorly conditioned systems can be rejected.
- **Fixed-size storage:** no dynamic allocation in the decomposition path. Array dimensions and work limits are compile-time bounded.
- **Synthetic demonstration:** a three-tone test with a DC offset is built into `main()`.

## Build and run

Requirements: a C++17 compiler. Example with GCC or Clang:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Wconversion -Wsign-conversion -pedantic \
    bresenham_spectrum.cpp -o bresenham_spectrum
./bresenham_spectrum
```

The bundled demo uses 512 samples at 8 kHz and a 129-point logarithmic grid from 100 Hz to 700 Hz. Its three synthetic tones are at 222.65625 Hz, 432.03125 Hz, and 517.1875 Hz, with peak amplitudes 12,000, 6,500, and 3,500 sample units. With the current source and GCC, the demo reports these three components and a residual RMS of approximately 0.45 sample units. Small output differences can occur across compilers or targets because of implementation details in initialization and rounding.

The source currently includes `main()`. To integrate the classes into an application, move the class definitions into a header/implementation unit or compile a copy with the demonstration `main()` removed. A future cleanup should separate the library from the test executable.

## Quick start: logarithmic frequency grid

The principal multi-tone API is `MultiToneComb<N, MAX_TONES, LUT_BITS>`. This is a representative integration pattern; the project file itself contains a fully compilable demonstration.

```cpp
#include <array>
#include <cstdint>
// Include the MultiToneComb definition after moving it to a header.

constexpr std::size_t N = 512;
constexpr std::size_t MAX_TONES = 5;
constexpr std::size_t GRID_POINTS = 129;
constexpr double FS_HZ = 8000.0;

using Analyzer = MultiToneComb<N, MAX_TONES, 12>;
Analyzer::InputBlock samples{}; // Fill with signed 16-bit ADC samples.
Analyzer::ToneList tones{};
Analyzer::ResidualBlock residual{};
std::array<std::uint32_t, GRID_POINTS> gridQ16{};
std::size_t toneCount = 0;
std::int32_t removedDc = 0;

Analyzer::prepare(); // Call before a time-critical acquisition loop.

if (!Analyzer::makeLogGridHz(100.0, 700.0, FS_HZ, gridQ16)) {
    // Invalid range or too few grid points after fixed-point quantization.
}

Analyzer analyzer;
const std::uint32_t minimumPeakAmplitude = 500;
const bool ok = analyzer.decomposeGrid(
    samples, gridQ16, minimumPeakAmplitude,
    tones, toneCount, residual, removedDc);

if (ok) {
    for (std::size_t i = 0; i < toneCount; ++i) {
        const double hz = Analyzer::frequencyQ16ToHz(
            tones[i].frequencyQ16, FS_HZ);
        // tones[i].peakAmplitude is in input-sample units.
        // tones[i].frequencyQ16 is a DFT-bin position in Q16.
        (void)hz;
    }
    const double residualRms =
        static_cast<double>(Analyzer::residualRmsQ8(residual)) / 256.0;
    (void)residualRms;
}
```

### Frequency units

The internal grid unit is **Q16 DFT bins**, not Hz:

\[
 f_{Q16} = \operatorname{round}\!\left(\frac{f_{Hz} N}{F_s}\,2^{16}\right)
\]

- `65536` means one DFT bin.
- `32768` means half a bin.
- `frequencyQ16ToHz()` converts a detected frequency back to Hz.
- `frequencyHzToBinQ16()` converts one positive frequency below Nyquist to the internal format.
- `makeLinearGridHz()` and `makeLogGridHz()` generate ordered arrays for a given sample rate and block length.

A logarithmic grid uses equal steps in `log(Hz)`, creating closer absolute spacing toward low frequencies. A caller can instead construct any strictly increasing Q16-bin array—e.g. octave bands, mel/Bark/ERB-inspired positions, or a grid concentrated around known mechanical/electrical harmonics.

**Important:** the grid controls where the search starts and how it distributes candidate points; it does not change the information contained in the samples. The observation duration `N / sampleRate` and the signal-to-noise ratio still govern how reliably nearby components can be separated. A denser low-frequency grid is not, by itself, a way to resolve frequencies that the observation cannot distinguish.

## APIs

### `MultiToneComb<N, MAX_TONES, LUT_BITS>`

| API | Purpose |
|---|---|
| `prepare()` | Initialise the static sine table before entering a real-time path. |
| `decompose(samples, firstBin, lastBin, minimumAmplitude, ...)` | Search the integer/half-bin comb over an inclusive integer-bin range. |
| `decomposeFrequencies(samples, frequenciesQ16, count, ...)` | Use a caller-owned array of custom Q16-bin frequencies. |
| `decomposeGrid(samples, std::array<...>, ...)` | Convenience wrapper for a compile-time-sized custom grid. |
| `makeLinearGridHz(minHz, maxHz, sampleRateHz, out)` | Generate a linearly spaced frequency grid; integer quotient/remainder spacing is distributed exactly in Q16-bin space. |
| `makeLogGridHz(minHz, maxHz, sampleRateHz, out)` | Generate a logarithmically spaced frequency grid during setup. |
| `frequencyHzToBinQ16(hz, sampleRateHz, out)` | Convert Hz to Q16-bin representation. |
| `frequencyQ16ToHz(binQ16, sampleRateHz)` | Convert a Q16-bin result to Hz. |
| `residualRmsQ8(residual)` | Return residual RMS in Q8 units; divide by 256 to express it in input-sample units. |

The decomposition APIs return `false` for invalid ranges, non-monotonic or too-large grids, or unsupported parameters. On success, `toneCount` is the number of components found; output arrays remain caller-owned and fixed-size.

### `BresenhamSpectrum<N, DETECTORS, LUT_BITS>`

The older single-tone detector remains in the source. It exposes an integer-bin detector bank (`analyze`) and a differential-phase refinement path (`refineSingleTone` / `analyzeWithDifferentialRefinement`). Differential-phase refinement assumes one dominant tone near the supplied coarse bin. It is not the general multi-tone decomposition API and should not be treated as a universal leakage-removal method.

## Configuration and bounds

For `MultiToneComb`:

- `N`: even sample count from **64 through 4096**.
- `MAX_TONES`: **1 through 16**.
- `LUT_BITS`: **8 through 12**, selecting 256–4096 table entries.
- Caller-provided grid: **2 through 4096** strictly increasing frequencies.
- Grid frequencies: strictly above DC and strictly below Nyquist.
- Input: signed 16-bit samples (`std::int16_t`).
- `minimumAmplitude`: peak amplitude threshold in input-sample units.
- Tone `frequencyQ16`: Q16 DFT-bin units.
- Tone `cosineQ8` and `sineQ8`: fitted coefficients scaled by 256.
- Residual buffer: Q8 sample units; convert to sample units by dividing by 256.

The linear-comb API requires `1 <= firstBin < lastBin < N/2`. It does not scan DC or the exact Nyquist component. The custom-grid API likewise excludes both endpoints. The DC mean is estimated and removed separately; the rounded value is reported as `removedDc`.

## Algorithm sketch

1. Estimate and remove the block's mean (DC component), keeping the residual in Q8 units.
2. Evaluate candidate frequencies using centred sine/cosine correlations and a two-coefficient least-squares fit. This accounts for finite-block non-orthogonality more accurately than raw correlation magnitude alone.
3. Pick the best candidate according to fitted residual error.
4. Refine frequency locally. In custom-grid mode, refinement follows the spacing of the selected grid cell, so it also works with nonuniform/logarithmic grids.
5. Subtract the fitted sinusoid from the residual and search for another component, up to `MAX_TONES` or until the candidate falls below `minimumAmplitude`.
6. Perform coordinate updates against the other selected components, then jointly refit all sine/cosine coefficients with the full centred Gram matrix.
7. Rebuild the final residual from the final component list and sort returned tones by descending peak amplitude.

The joint solve addresses correlation between selected detector basis functions. It does not prove that the chosen set of frequencies is the unique explanation of the signal.

## Numeric design

- Unsigned 32-bit phase arithmetic intentionally wraps modulo `2^32`; this is modular phase arithmetic (a finite ring), not a finite field.
- Bresenham quotient/remainder accumulators distribute the fractional phase increment across samples rather than rounding one increment and accumulating its bias.
- Correlation and normal-equation calculations use signed 64-bit intermediates and fixed-point scaling.
- `LUT_BITS` controls the table size. Linear interpolation between adjacent table values smooths the objective used by local frequency refinement.
- The lookup table is generated using floating-point sine once. Log-grid generation and Hz/bin conversion also use floating point, but they are setup/conversion operations rather than the inner sample loops. Call `prepare()` early if lazy table initialisation must not happen in a time-critical section.

The source does not rely on unsigned overflow of sample accumulators; phase wrap is intentional, while other arithmetic is expected to remain in its documented bounds. Still, build and test on the actual target, since compiler, ABI, stack, and runtime constraints vary.

## Limits and interpretation

1. **Not an FFT.** This is a bank of direct correlations and iterative fitting. Cost grows with the number of candidate frequencies, sample count, tone count, and refinement passes. A large grid combined with a large block can be expensive; benchmark on the target.
2. **No guaranteed arbitrary mixture recovery.** Closely spaced tones, large amplitude disparities, low SNR, harmonics, modulation, transients and coloured noise can cause missed or spurious components.
3. **Conditioning matters.** Nearby frequencies create highly correlated sine/cosine columns. The fixed-point Gram solver rejects some ill-conditioned fits, but its pivot threshold is heuristic, not a formal uncertainty bound.
4. **Finite observation remains finite.** The method reduces leakage from already-modelled tones by subtracting their fitted waveforms. It cannot universally eliminate leakage, and it cannot create information absent from a finite sample block.
5. **Rectangular observation model.** There is no selectable Hann/Blackman or other analysis window in the current code. Windowing may improve sidelobes but must be integrated consistently into both fitting and reconstruction.
6. **Frequency-domain grid vs. precision.** The custom grid is a candidate search structure; local frequency refinement can return values between its points. It is not necessary to use an extremely dense grid everywhere, but a poor coarse grid can still miss a peak or put the search in an unfavourable basin.
7. **DC and Nyquist handling.** The decomposition path estimates/removes DC and rejects grid endpoints. A dedicated DC/Nyquist estimator is not currently exposed.
8. **No production status claims.** The current demo covers a small controlled test, not noise statistics, stress limits, real ADC data, long-run streaming, or every supported template configuration.
9. **Memory / stack.** For `N=4096`, the caller's `InputBlock` is about 8 KiB and `ResidualBlock` about 32 KiB. With `MAX_TONES=16`, the fixed 32×32 64-bit joint-solver matrix alone is about 8 KiB of local storage, in addition to other work arrays. This may be too much for a small embedded task stack; inspect stack usage or move workspaces to static/caller-owned storage if required.
10. **Thread safety.** Static local table construction is guarded by C++11+ initialization rules, but the analysis outputs and buffers belong to the caller. Do not share mutable output buffers between concurrent calls.

## Roadmap

Priorities for making this more useful and robust:

1. **Separate library and tests:** move the classes into a header/source pair; remove demo `main()` from the reusable target; add a repeatable unit-test harness and CI build.
2. **Reference-model validation:** compare output against a floating-point least-squares reference and a high-resolution FFT on deterministic/randomised mixtures, with published frequency, amplitude and residual error tolerances.
3. **Stress tests:** cover noise floors, clipping, DC drift, chirps/modulation, phase randomisation, near-colliding tones, strong/weak adjacent components, and tones near DC/Nyquist.
4. **Conditioning and confidence:** report Gram-matrix conditioning/pivot diagnostics, per-tone confidence, fit improvement, and a noise-calibrated stopping criterion instead of relying only on an absolute amplitude threshold.
5. **Window support:** implement weighted least squares with consistent synthesis/subtraction for selectable rectangular, Hann, Blackman and other windows.
6. **Faster candidate search:** coarse-to-fine grids, a bounded top-K candidate queue, recurrence-based reference generation, candidate reuse, SIMD/DSP intrinsics, or an FFT/Goertzel coarse preselector.
7. **Streaming support:** overlap/hop processing, persistent ring buffers, optional phase-continuous detectors, and explicit latency/throughput measurements.
8. **Workspace control:** allow the caller to provide Gram matrices, solver scratch space and residual workspaces to avoid large embedded stack allocations.
9. **Adaptive scales:** generate logarithmic, octave, mel/Bark/ERB, harmonic and user-defined grids with explicit spacing/error targets; allocate more search density where needed rather than uniformly over all frequencies.
10. **Improved solvers:** scaled or fraction-free elimination, fixed-point LDLᵀ/QR alternatives, regularisation, and robust handling of ill-conditioned or overcomplete tone sets.
11. **Amplitude/phase conventions:** expose phase, RMS-vs-peak amplitude, coefficient uncertainty and calibrated scaling for the selected lookup table and quantization.

## License

No license is specified in the current source. Add an explicit license before redistributing or incorporating the code into a product.
