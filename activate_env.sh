#!/bin/bash
# F1Tenth Simulator Environment Activation Script

echo "Activating F1Tenth Simulator Environment..."
source f1tenth_sim_env/bin/activate
echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Available packages:"
echo "- numpy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "- opencv: $(python -c 'import cv2; print(cv2.__version__)')"
echo "- PIL: $(python -c 'from PIL import Image; print(Image.__version__)')"
echo ""
echo "Ready to run F1Tenth simulator scripts!"
echo "Example: python test_vision_hybrid.py --mode depth --frames 5"