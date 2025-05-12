"""Test script to verify environment setup."""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def test_environment():
    """Verify that all required components are working."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Check if MPS (Metal Performance Shaders) is available for Mac
    if torch.backends.mps.is_available():
        print("MPS (Metal) is available! You can use GPU acceleration on Mac")
        device = torch.device("mps")
    else:
        print("MPS not available. Using CPU")
        device = torch.device("cpu")
    
    # Test PyTorch operations
    x = torch.randn(3, 224, 224).to(device)
    print(f"Tensor shape: {x.shape}, device: {x.device}")
    
    # Test OpenCV operations
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (0, 255, 0), 2)
    
    print("All imports and basic operations successful!")
    return True

if __name__ == "__main__":
    test_environment()