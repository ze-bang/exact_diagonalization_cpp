#!/bin/bash
# Comprehensive script to process both taylor and global modes, 
# combine channels, and calculate QFI for all modes

# Usage: ./process_all_modes.sh <structure_factor_results_dir>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <structure_factor_results_dir>"
    echo "Example: $0 ./structure_factor_results"
    exit 1
fi

STRUCTURE_FACTOR_DIR="$1"

if [ ! -d "$STRUCTURE_FACTOR_DIR" ]; then
    echo "Error: Directory not found: $STRUCTURE_FACTOR_DIR"
    exit 1
fi

echo "=========================================="
echo "Processing structure factor data"
echo "Directory: $STRUCTURE_FACTOR_DIR"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Step 1: Combine channels for both taylor and global modes
echo "Step 1: Combining SpSm/SmSp and SF/NSF channels..."
python3 "$SCRIPT_DIR/combine_channels.py" "$STRUCTURE_FACTOR_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Channel combination failed"
    exit 1
fi

echo ""
echo "Step 1 completed successfully!"
echo ""

# Step 2: Calculate QFI for all modes
echo "Step 2: Calculating QFI for all modes..."
python3 "$SCRIPT_DIR/calc_QFI.py" "$STRUCTURE_FACTOR_DIR" False all

if [ $? -ne 0 ]; then
    echo "Error: QFI calculation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "All processing completed successfully!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - Combined channels: $STRUCTURE_FACTOR_DIR/beta_*/taylor_combined/"
echo "  - Combined channels: $STRUCTURE_FACTOR_DIR/beta_*/global_combined/"
echo "  - QFI results: $STRUCTURE_FACTOR_DIR/processed_data_*/"
echo ""
