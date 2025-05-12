import os
import sys
import torch
import argparse
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoProcessor
import json
import yaml
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # Insert at beginning to ensure our modules are found first

# Now import our modules
from pix2code_project.utils.preprocessing import ImagePreprocessor
from pix2code_project.utils.tokenizer import AdaptiveTokenizer
from pix2code_project.models.ui2code import UI2Code
from pix2code_project.data.dataset import UI2CodeDataset
from pix2code_project.training.trainer import Trainer
from pix2code_project.utils.logging import TrainingLogger
from pix2code_project.utils.validation import Validator
from pix2code_project.models.encoder.visual_encoder import VisualEncoder
from pix2code_project.models.attention.layout_attention import LayoutAttention
from pix2code_project.models.decoder.code_decoder import CodeDecoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def custom_collate_fn(batch, preprocessor, tokenizer):
    """Custom collate function to preprocess images and text before batching."""
    # Process images
    images = [preprocessor.preprocess(item['image']) for item in batch]
    images = torch.stack(images)
    
    # Process HTML code from text field
    code_texts = [item['text'] for item in batch]
    code_tokens = tokenizer.encode(code_texts)
    
    # Process layout from bbox field if available
    layouts = []
    for item in batch:
        try:
            bbox = json.loads(item['bbox'])
            # Extract layout features (x, y, width, height)
            layout_features = torch.tensor([
                bbox.get('x', 0),
                bbox.get('y', 0),
                bbox.get('width', 0),
                bbox.get('height', 0)
            ], dtype=torch.float32)
            layouts.append(layout_features)
        except (json.JSONDecodeError, KeyError):
            layouts.append(None)
    
    if all(layout is not None for layout in layouts):
        # Stack layout features
        layouts = torch.stack(layouts)  # Shape: (batch_size, 4)
    else:
        layouts = None
    
    # Return a dictionary with all tensors
    return {
        'image': images,
        'input_ids': code_tokens['input_ids'],
        'attention_mask': code_tokens['attention_mask'],
        'layout': layouts
    }

def create_dataloader(dataset, batch_size, num_workers=0, shuffle=True, collate_fn=None):
    """Create a DataLoader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # Load configuration
    config_path = os.path.join(project_root, 'pix2code_project', 'configs', 'config.yaml')
    logger.info(f"Loading config from: {os.path.abspath(config_path)}")
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = UI2CodeDataset(
        data_dir=config['data']['data_dir'],
        max_samples=config['data'].get('max_samples', None)
    )
    logger.info(f"Loaded dataset with {len(dataset)} samples.\n")
    
    # Log dataset structure
    logger.info("Dataset structure:")
    sample = dataset[0]
    logger.info(f"First item keys: {sample.keys()}")
    logger.info(f"First item example: {sample}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        logger.warning("Not enough samples to split, using all data for both training and validation.")
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 0),
        collate_fn=dataset.collate_fn
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 0),
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = UI2Code(
        visual_encoder_config=config['visual_encoder'],
        layout_processor_config=config['layout_processor'],
        code_decoder_config=config['code_decoder']
    )
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training'].get('weight_decay', 0.01))
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(config['training']['epochs']),
        eta_min=float(config['training'].get('min_lr', 1e-6))
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=config['training']['output_dir']
    )
    
    # Log hyperparameters
    trainer.log_hyperparameters(config)
    
    # Initialize validator
    validator = Validator(model, dataset.tokenizer, device)
    
    # Training loop
    logger.info("Starting training...")
    best_bleu = 0.0
    
    for epoch in range(config['training']['epochs']):
        # Train epoch
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = validator.validate_batch(next(iter(val_loader)))
        val_loss = val_metrics['loss']
        bleu_score = val_metrics['bleu']
        
        # Save checkpoint
        is_best = bleu_score > best_bleu
        if is_best:
            best_bleu = bleu_score
        trainer.save_checkpoint(epoch, bleu_score, is_best)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']} - "
                   f"Train Loss: {train_loss:.4f} - "
                   f"Val Loss: {val_loss:.4f} - "
                   f"BLEU: {bleu_score:.4f}")
    
    logger.info("Training completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UI2Code model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Path to output directory')
    parser.add_argument('--train', action='store_true',
                      help='Train the model')
    parser.add_argument('--save_model', action='store_true',
                      help='Save the model after training')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args) 