# FMM Project

This repository contains the implementation of the FMM (Fast Mockup Maker) project, which includes tools for dataset preparation, model training, and testing.

## Project Structure

- `prepare_datasets.py`: Script for preparing and processing datasets
- `run_training.py`: Main training script for the model
- `test_model.py`: Script for testing the trained model
- `run_api.py`: API implementation for model inference
- `jira.py`: Jira integration utilities
- `grid_detector/`: Grid detection implementation
- `implementation/`: Core implementation files
- `pix2code_project/`: Pix2Code related files

## Setup

1. Clone the repository:
```bash
git clone https://github.com/alecpos/fmm.git
cd fmm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage in Google Colab

1. Clone the repository in Colab:
```python
!git clone https://github.com/yourusername/fmm.git
%cd fmm
```

2. Install dependencies:
```python
!pip install -r requirements.txt
```

3. Run the desired script:
```python
# For training
!python run_training.py

# For testing
!python test_model.py
```

## Training

To train the model:
```bash
python run_training.py
```

## Testing

To test the model:
```bash
python test_model.py
```

## API Usage

To run the API:
```bash
python run_api.py
```

## License

[Add your license here] 