# Pix2Code: CNN-LSTM Architecture with Dual Attention for Mockup-to-Code Conversion

This project implements a deep learning model for converting UI mockups to code using a CNN-LSTM architecture with dual attention mechanisms.

## Project Structure

```
pix2code_project/
├── data/
│   ├── preprocessing/
│   ├── dataset_utils.py
│   └── data_loader.py
├── models/
│   ├── cnn_feature_extractor.py
│   ├── lstm_code_generator.py
│   ├── attention_module.py
│   └── pix2code_model.py
├── training/
│   ├── loss_functions.py
│   ├── metrics.py
│   └── trainer.py
├── utils/
│   ├── visualization.py
│   └── evaluation.py
├── configs/
│   └── config.yaml
├── notebooks/
│   └── exploration.ipynb
├── results/
├── main.py
└── README.md
```

## Setup

1. Create and activate a conda environment:
```bash
conda create -n pix2code python=3.10
conda activate pix2code
```

2. Install dependencies:
```bash
conda install -c pytorch pytorch=2.3.1 torchvision
pip install tensorflow==2.14.0
pip install tensorflow-addons==0.23.0
pip install keras-cv==0.8.2
pip install matplotlib pandas scikit-learn scikit-image
pip install opencv-python pillow nltk
```

## Usage

1. Prepare your dataset:
   - Place mockup images in `data/training/images/` and `data/validation/images/`
   - Place corresponding code files in `data/training/codes/` and `data/validation/codes/`

2. Train the model:
```bash
python main.py --data_dir data --output_dir results --train --save_model
```

3. Use the trained model:
```bash
python main.py --data_dir data --output_dir results
```

## Model Architecture

The model consists of several key components:

1. **CNN Feature Extractor**:
   - ResNet-50 backbone with transfer learning
   - Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context
   - Coordinate Attention for spatial relationship preservation

2. **LSTM Code Generator**:
   - Bidirectional LSTM for context capture
   - Second LSTM with state output for attention
   - Dropout for regularization

3. **Dual Attention Mechanism**:
   - Visual attention gate for image features
   - Syntactic attention gate for code features
   - Co-attention framework for feature fusion

## Training

The model is trained using:
- Teacher forcing for sequence generation
- Masked loss to handle padding
- BLEU-4 and exact match metrics for evaluation
- Checkpointing for model saving and loading

## Configuration

Model parameters can be adjusted in `configs/config.yaml`:
- Image size and code length
- Vocabulary size and embedding dimensions
- LSTM units and attention units
- Training batch size and epochs
- Learning rate schedule
- Regularization parameters

## License

This project is licensed under the MIT License - see the LICENSE file for details. 