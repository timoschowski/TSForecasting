#!/bin/bash
# Complete installation script for GluonTS with PyTorch and running the real comparison

set -e  # Exit on error

echo "========================================================================"
echo "GluonTS Developer Setup and Real Comparison Script"
echo "========================================================================"

# Wait for any ongoing PyTorch installation
echo -e "\n[1/5] Checking for ongoing installations..."
while pgrep -f "pip install torch" > /dev/null; do
    echo "  Waiting for PyTorch installation to complete..."
    sleep 10
done

# Install PyTorch if not already installed
echo -e "\n[2/5] Installing PyTorch..."
if python3 -c "import torch" 2>/dev/null; then
    echo "  ✓ PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
    echo "  Installing PyTorch CPU version..."
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir
    echo "  ✓ PyTorch installed: $(python3 -c 'import torch; print(torch.__version__)')"
fi

# Install GluonTS in development mode with PyTorch support
echo -e "\n[3/5] Installing GluonTS with PyTorch support..."
cd /home/user/gluonts
pip install -e ".[torch]" --no-cache-dir
echo "  ✓ GluonTS installed with torch support"

# Verify installation
echo -e "\n[4/5] Verifying installation..."
python3 -c "
import torch
import lightning
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import TweedieOutput, NormalOutput
print('  ✓ torch:', torch.__version__)
print('  ✓ lightning:', lightning.__version__)
print('  ✓ GluonTS DeepAR: OK')
print('  ✓ TweedieOutput: OK')
print('  ✓ NormalOutput: OK')
"

# Run the real comparison script
echo -e "\n[5/5] Running real DeepAR comparison..."
echo "========================================================================"
cd /home/user/TSForecasting
python3 create_real_comparison_plot.py

echo -e "\n========================================================================"
echo "Setup and comparison completed successfully!"
echo "========================================================================"
echo -e "\nGenerated plots:"
ls -lh /home/user/TSForecasting/*_real.png /home/user/TSForecasting/*_performance_*.png 2>/dev/null || echo "  (Plots will appear after training completes)"
