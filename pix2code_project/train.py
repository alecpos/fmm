import torch
from torch.utils.data import DataLoader
from data.dataset import WebCode2MDataset
from utils.preprocessing import ImagePreprocessor
from utils.tokenizer import AdaptiveTokenizer
from models.ui2code import UI2Code
from training.trainer import UIToCodeTrainer
import yaml

def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    preprocessor = ImagePreprocessor()
    tokenizer = AdaptiveTokenizer()
    
    # Create datasets
    train_dataset = WebCode2MDataset(
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        split='train'
    )
    
    val_dataset = WebCode2MDataset(
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Initialize model
    model = UI2Code(
        visual_encoder=config['visual_encoder'],
        layout_processor=config['layout_processor'],
        code_decoder=config['code_decoder']
    )
    
    # Initialize trainer
    trainer = UIToCodeTrainer(
        model=model,
        tokenizer=tokenizer,
        device=config['device']
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs']
    )

if __name__ == '__main__':
    main() 