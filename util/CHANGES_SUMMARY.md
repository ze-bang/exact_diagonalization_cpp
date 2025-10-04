# Summary of Changes for Global Mode Processing

## Files Created/Modified

### 1. New Script: `combine_channels.py`
**Location:** `exact_diagonalization_cpp/util/combine_channels.py`

**Purpose:** Combines spin correlation channels for both taylor and global modes

**Features:**
- **Taylor mode (local frame):**
  - Combines `SpSm + SmSp → (SpSm + SmSp)/2`
  
- **Global mode (global frame):**
  - Combines `SpSm_SF + SmSp_SF → (SpSm_SF + SmSp_SF)/2`
  - Combines `SpSm_NSF + SmSp_NSF → (SpSm_NSF + SmSp_NSF)/2`
  - Sums channels: `SF + NSF → (SpSm_SF + SmSp_SF)/2 + (SpSm_NSF + SmSp_NSF)/2`

**Usage:**
```bash
python3 combine_channels.py <structure_factor_results_dir> [--beta BETA]
```

**Output directories:**
- `beta_*/taylor_combined/` - Combined taylor mode data
- `beta_*/global_combined/` - Combined global mode data (SF, NSF, and SF+NSF)

---

### 2. Modified Script: `calc_QFI.py`
**Location:** `exact_diagonalization_cpp/util/calc_QFI.py`

**Changes:**
- Added `mode` parameter to support: `'taylor'`, `'global'`, `'taylor_combined'`, `'global_combined'`, or `'all'`
- Updated `_collect_data_files()` to search in mode-specific directories
- Updated `parse_QFI_data_new()` to accept mode parameter
- Updated `_process_species_data()` to pass mode through processing pipeline
- Updated `_save_species_results()` to save in mode-specific directories (`processed_data_<mode>/`)
- Updated `_plot_spectral_function()` to include mode in plot titles
- Modified main section to process multiple modes

**New Usage:**
```bash
# Process all modes
python3 calc_QFI.py <dir> False all

# Process specific mode
python3 calc_QFI.py <dir> False taylor
python3 calc_QFI.py <dir> False taylor_combined
python3 calc_QFI.py <dir> False global
python3 calc_QFI.py <dir> False global_combined
```

**Output directories:**
- `processed_data_taylor/`
- `processed_data_taylor_combined/`
- `processed_data_global/`
- `processed_data_global_combined/`

---

### 3. New Script: `process_all_modes.sh`
**Location:** `exact_diagonalization_cpp/util/process_all_modes.sh`

**Purpose:** Wrapper script that runs the complete processing pipeline

**What it does:**
1. Runs `combine_channels.py` to combine all channels
2. Runs `calc_QFI.py` with `mode='all'` to process all modes

**Usage:**
```bash
bash process_all_modes.sh <structure_factor_results_dir>
```

---

### 4. Documentation: `PROCESSING_MODES_README.md`
**Location:** `exact_diagonalization_cpp/util/PROCESSING_MODES_README.md`

**Contents:**
- Overview of taylor vs global modes
- Directory structure explanations
- Step-by-step processing instructions
- Output format specifications
- Physical interpretation of channels
- QFI calculation details
- Usage examples
- Troubleshooting guide

---

## Workflow Summary

### Input (from TPQ_DSSF.cpp with method='global')
```
structure_factor_results/
├── beta_1.0/
│   ├── taylor/
│   │   ├── time_corr_rand*_SpSm_*.dat
│   │   └── time_corr_rand*_SmSp_*.dat
│   └── global/
│       ├── time_corr_rand*_SpSm_*_SF_*.dat
│       ├── time_corr_rand*_SmSp_*_SF_*.dat
│       ├── time_corr_rand*_SpSm_*_NSF_*.dat
│       └── time_corr_rand*_SmSp_*_NSF_*.dat
```

### After combine_channels.py
```
structure_factor_results/
├── beta_1.0/
│   ├── taylor_combined/
│   │   └── time_corr_rand*_SpSm+SmSp_*.dat          # (SpSm+SmSp)/2
│   └── global_combined/
│       ├── time_corr_rand*_SpSm+SmSp_*_SF_*.dat     # (SpSm_SF+SmSp_SF)/2
│       ├── time_corr_rand*_SpSm+SmSp_*_NSF_*.dat    # (SpSm_NSF+SmSp_NSF)/2
│       └── time_corr_rand*_SpSm+SmSp_*_SF+NSF_*.dat # SF + NSF
```

### After calc_QFI.py
```
structure_factor_results/
├── processed_data_taylor/           # QFI from original taylor
├── processed_data_taylor_combined/  # QFI from (SpSm+SmSp)/2 taylor
├── processed_data_global/           # QFI from original global
└── processed_data_global_combined/  # QFI from combined global channels
    ├── SpSm+SmSp_*_SF/             # QFI for SF channel only
    ├── SpSm+SmSp_*_NSF/            # QFI for NSF channel only
    └── SpSm+SmSp_*_SF+NSF/         # QFI for total (SF+NSF)
```

---

## Key Features

### 1. Channel Combination
✅ **Taylor mode:** Symmetrizes SpSm and SmSp to get `(SpSm + SmSp)/2`
✅ **Global mode:** 
   - Symmetrizes each channel separately: SF and NSF
   - Provides total scattering: `SF + NSF`

### 2. Multi-Mode Processing
✅ Process all modes in one command: `mode='all'`
✅ Or process selectively: individual mode names
✅ Output organized by mode in separate directories

### 3. Complete Pipeline
✅ One command runs everything: `process_all_modes.sh`
✅ Comprehensive error checking
✅ Progress reporting at each step

### 4. Physical Interpretation
✅ **Local frame (taylor):** Standard spin correlations in local basis
✅ **Global frame (global):** Scattering channels for comparison with experiments
   - **SF channel:** Spin-flip scattering
   - **NSF channel:** Non-spin-flip scattering
   - **Total (SF+NSF):** Full scattering intensity

---

## Testing the Implementation

To test with your data:

```bash
# Navigate to util directory
cd /home/zhouzb79/projects/def-ybkim/zhouzb79/exact_diagonalization_cpp/util

# Run the complete pipeline
bash process_all_modes.sh ../path/to/structure_factor_results

# Or run steps individually:
# Step 1: Combine channels
python3 combine_channels.py ../path/to/structure_factor_results

# Step 2: Calculate QFI for all modes
python3 calc_QFI.py ../path/to/structure_factor_results False all
```

---

## Expected Output

For each mode, you will get:
1. **Combined time correlations** in `*_combined/` directories
2. **Spectral functions** S(ω) from FFT
3. **QFI values** for each beta
4. **Peak positions** and properties
5. **Plots** of spectral functions with marked peaks

### Example QFI comparison:
- **Taylor combined:** QFI from `(SpSm+SmSp)/2` in local frame
- **Global SF:** QFI from `(SpSm_SF+SmSp_SF)/2` (spin-flip only)
- **Global NSF:** QFI from `(SpSm_NSF+SmSp_NSF)/2` (non-spin-flip only)
- **Global Total:** QFI from `SF + NSF` (total scattering)

This allows you to:
- Compare local vs global frame results
- Analyze individual scattering channels
- Study the relative contributions of SF and NSF to total QFI
