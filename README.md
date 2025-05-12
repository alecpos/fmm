# Fast Mockup Maker (FMM)

A deep learning project that converts UI mockups into code using a combination of computer vision and natural language processing techniques.

## Overview

FMM is a tool that can automatically generate code from UI mockups. It uses a combination of:
- Computer vision for UI element detection
- Layout analysis for understanding component positioning
- Code generation using transformer-based models

## Features

- UI mockup to code conversion
- Support for multiple UI frameworks
- Layout-aware code generation
- Grid detection and alignment
- Jira integration for project management

## Project Structure

```
fmm/
├── pix2code_project/          # Core project files
│   ├── data/                  # Dataset handling and processing
│   │   ├── webcode2m/        # WebCode2M dataset
│   │   ├── pix2code/         # Pix2Code dataset
│   │   └── rico/            # Rico dataset
│   ├── models/               # Model architecture definitions
│   │   ├── encoder/         # Visual encoders
│   │   ├── decoder/         # Code decoders
│   │   └── attention/       # Attention mechanisms
│   ├── utils/               # Utility functions
│   ├── training/            # Training utilities
│   └── configs/             # Configuration files
├── grid_detector/           # Grid detection implementation
├── implementation/          # Core implementation files
├── prepare_datasets.py      # Dataset preparation script
├── run_training.py          # Training script
├── test_model.py           # Model testing script
├── run_api.py              # API implementation
└── jira.py                 # Jira integration utilities
```

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/alecpos/fmm.git
cd fmm
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r pix2code_project/requirements.txt
```

### Google Colab Setup

1. Clone the repository:
```python
!git clone https://github.com/alecpos/fmm.git
%cd fmm
```

2. Install dependencies:
```python
!pip install -r requirements.txt
!pip install -r pix2code_project/requirements.txt
```

3. Set up the Python path:
```python
import sys
import os
sys.path.append(os.getcwd())
```

## Usage

### Training

To train the model:

```bash
python run_training.py --data_dir pix2code_project/data --output_dir pix2code_project/results --train
```

Key training parameters:
- `--data_dir`: Path to dataset directory
- `--output_dir`: Path to save model checkpoints and results
- `--train`: Flag to start training

### Testing

To test the model:

```bash
python test_model.py
```

### API Usage

To run the API server:

```bash
python run_api.py
```

## Troubleshooting

### Common Issues

1. TensorFlow Warnings
   - If you see oneDNN custom operations warnings, they can be safely ignored
   - To disable: `os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'`

2. Module Import Errors
   - Ensure project root is in Python path
   - Verify all required files are present
   - Check you're in the correct directory

3. CUDA Warnings
   - These are typically informational
   - Don't affect functionality
   - Can be ignored in most cases

### Dataset Issues

If you encounter dataset-related errors:
1. Verify dataset directory structure
2. Check file permissions
3. Ensure all required files are present

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license here]

## Contact

For questions and support, please open an issue in the GitHub repository. 