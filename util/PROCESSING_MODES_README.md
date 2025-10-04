# Processing Global and Local Frame Structure Factors

This document describes the workflow for processing structure factor data from both taylor (local frame) and global (global frame) modes, including channel combination and QFI calculation.

## Overview

The TPQ_DSSF code can generate structure factor data in two modes:
- **Taylor mode** (local frame): Standard time evolution in the local spin frame
- **Global mode** (global frame): Time evolution in the global frame with SF (spin-flip) and NSF (non-spin-flip) channels

## Directory Structure

After running TPQ_DSSF with both modes, you'll have:

```
structure_factor_results/
├── beta_1.0/
│   ├── taylor/
│   │   ├── time_corr_rand0_SpSm_q_Qx0.0_Qy0.0_Qz0.0_beta=1.0.dat
│   │   ├── time_corr_rand0_SmSp_q_Qx0.0_Qy0.0_Qz0.0_beta=1.0.dat
│   │   └── ...
│   └── global/
│       ├── time_corr_rand0_SpSm_q_Qx0.0_Qy0.0_Qz0.0_SF_beta=1.0.dat
│       ├── time_corr_rand0_SmSp_q_Qx0.0_Qy0.0_Qz0.0_SF_beta=1.0.dat
│       ├── time_corr_rand0_SpSm_q_Qx0.0_Qy0.0_Qz0.0_NSF_beta=1.0.dat
│       ├── time_corr_rand0_SmSp_q_Qx0.0_Qy0.0_Qz0.0_NSF_beta=1.0.dat
│       └── ...
├── beta_2.0/
│   └── ...
└── beta_inf/
    └── ...
```

## Processing Steps

### Step 1: Combine Channels

The `combine_channels.py` script performs the following operations:

#### For Taylor (Local Frame) Mode:
- Combines SpSm and SmSp: `(SpSm + SmSp)/2`
- Output: `taylor_combined/time_corr_rand*_SpSm+SmSp_*.dat`

#### For Global (Global Frame) Mode:
1. Combines SF channel: `(SpSm_SF + SmSp_SF)/2`
2. Combines NSF channel: `(SpSm_NSF + SmSp_NSF)/2`
3. Sums both channels: `(SpSm_SF + SmSp_SF)/2 + (SpSm_NSF + SmSp_NSF)/2`

Outputs:
- `global_combined/time_corr_rand*_SpSm+SmSp_*_SF_*.dat`
- `global_combined/time_corr_rand*_SpSm+SmSp_*_NSF_*.dat`
- `global_combined/time_corr_rand*_SpSm+SmSp_*_SF+NSF_*.dat`

**Usage:**
```bash
python3 combine_channels.py <structure_factor_results_dir> [--beta BETA]

# Process all beta values
python3 combine_channels.py ./structure_factor_results

# Process specific beta only
python3 combine_channels.py ./structure_factor_results --beta 1.0
```

### Step 2: Calculate QFI

The `calc_QFI.py` script processes the time-correlation data to compute:
- Spectral functions S(ω) via Fourier transform
- Quantum Fisher Information (QFI)
- Peak positions and properties

It can process different modes separately or all at once.

**Usage:**
```bash
python3 calc_QFI.py <structure_factor_results_dir> <across_QFI> <mode>

# Process all modes
python3 calc_QFI.py ./structure_factor_results False all

# Process specific mode
python3 calc_QFI.py ./structure_factor_results False taylor
python3 calc_QFI.py ./structure_factor_results False taylor_combined
python3 calc_QFI.py ./structure_factor_results False global
python3 calc_QFI.py ./structure_factor_results False global_combined
```

### All-in-One Processing

For convenience, use the wrapper script that runs both steps:

```bash
bash process_all_modes.sh <structure_factor_results_dir>
```

This script will:
1. Combine all channels (SpSm/SmSp and SF/NSF)
2. Calculate QFI for all modes (taylor, taylor_combined, global, global_combined)

## Output Structure

After processing, you'll have:

```
structure_factor_results/
├── beta_1.0/
│   ├── taylor/                    # Original taylor mode data
│   ├── taylor_combined/           # Combined (SpSm+SmSp)/2
│   ├── global/                    # Original global mode data
│   └── global_combined/           # Combined SF, NSF, and SF+NSF
├── processed_data_taylor/         # QFI for taylor mode
│   └── SpSm_q_Qx0.0_Qy0.0_Qz0.0/
│       ├── spectral_beta_1.0.dat
│       ├── peaks_beta_1.0.dat
│       └── spectral_function_*.png
├── processed_data_taylor_combined/    # QFI for combined taylor
├── processed_data_global/             # QFI for global mode
├── processed_data_global_combined/    # QFI for combined global
└── plots/                            # Summary plots
```

## Data Format

### Time Correlation Files
```
# t correlation_real correlation_imag
0.00000000e+00  1.23456789e+00  0.00000000e+00
1.00000000e-02  1.23456789e+00  -1.23456789e-03
...
```

### Spectral Function Files
```
# freq spectral_function
-3.00000000e+00  1.23456789e-05
-2.99000000e+00  1.23456789e-05
...
```

### Peak Files
```
# freq height prominence
1.23456789e+00  5.67890123e-02  3.45678901e-02
...
```

## Physical Interpretation

### Local Frame (Taylor Mode)
- **SpSm**: S⁺S⁻ correlation in local frame
- **SmSp**: S⁻S⁺ correlation in local frame
- **(SpSm+SmSp)/2**: Average transverse correlation, symmetric combination

### Global Frame (Global Mode)
- **SF (Spin-Flip)**: Spin-flip scattering channel
- **NSF (Non-Spin-Flip)**: Non-spin-flip scattering channel
- **(SpSm_SF+SmSp_SF)/2**: Average SF channel
- **(SpSm_NSF+SmSp_NSF)/2**: Average NSF channel
- **SF+NSF**: Total scattering (sum of both channels)

The global frame representation is particularly useful for:
- Comparing with neutron scattering experiments
- Understanding polarization-dependent scattering
- Analyzing magnetic excitations in different scattering geometries

## Quantum Fisher Information (QFI)

The QFI is calculated as:

For finite β:
```
QFI = 4 ∫ S(ω) tanh(βω/2) [1 - exp(-βω)] dω
```

For β→∞:
```
QFI = 4 ∫ S(ω) dω
```

The QFI quantifies quantum correlations and is useful for:
- Identifying quantum phase transitions (peaks in dQFI/dβ)
- Characterizing entanglement
- Comparing different spin channels

## Examples

### Example 1: Process specific beta value
```bash
python3 combine_channels.py ./structure_factor_results --beta 2.5
python3 calc_QFI.py ./structure_factor_results False taylor_combined
```

### Example 2: Process all modes for all beta values
```bash
bash process_all_modes.sh ./structure_factor_results
```

### Example 3: Compare local and global frames
```bash
# Process both
bash process_all_modes.sh ./structure_factor_results

# Compare QFI values
cat processed_data_taylor_combined/SpSm+SmSp_*/spectral_beta_*.dat
cat processed_data_global_combined/SpSm+SmSp_*_SF+NSF/spectral_beta_*.dat
```

## Notes

1. **Memory**: Processing large systems may require significant memory, especially for β→∞
2. **Time**: FFT computation scales with the number of time points
3. **Broadening**: A Lorentzian broadening (γ=0.3 by default) is applied in time domain before FFT
4. **Compensation**: The spectral function is compensated to preserve the integral after truncating to ω>0

## Troubleshooting

### Missing channels
If you see warnings about missing SpSm/SmSp or SF/NSF channels, check that:
- TPQ_DSSF was run with the correct spin_combinations parameter
- Both SpSm (op=0,1) and SmSp (op=1,0) were included
- For global mode, ensure method='global' was used

### Time array mismatch
If time arrays don't match between files:
- Check that all files used the same dt and t_end parameters
- Verify files are from the same sample and beta value

### Low QFI values
Low QFI may indicate:
- Weak quantum correlations
- Need for higher beta (lower temperature)
- Possible numerical precision issues

## References

For more details on the TPQ method and structure factor calculations, see:
- `TPQ_DSSF.cpp` - Main implementation
- `observables.h` - Operator definitions for global/local frames
- `EIGENMODE_METHOD.md` - Documentation on eigenmode expansion
