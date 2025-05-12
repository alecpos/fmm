import tensorflow as tf
import argparse
import os
import yaml

from data.data_loader import Pix2CodeDataset, CodeTokenizer
from models.pix2code_model import create_pix2code_model
from training.trainer import Pix2CodeTrainer

def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Create tokenizer
    tokenizer = CodeTokenizer(vocab_size=config['model']['vocab_size'])
    
    # Create dataset
    train_dataset = Pix2CodeDataset(
        data_dir=os.path.join(args.data_dir, 'training'),
        img_size=tuple(config['data']['img_size']),
        max_code_length=config['data']['max_code_length'],
        batch_size=config['training']['batch_size'],
        tokenizer=tokenizer
    )
    
    val_dataset = Pix2CodeDataset(
        data_dir=os.path.join(args.data_dir, 'validation'),
        img_size=tuple(config['data']['img_size']),
        max_code_length=config['data']['max_code_length'],
        batch_size=config['training']['batch_size'],
        tokenizer=tokenizer
    )
    
    # Prepare tokenizer if not already prepared
    if args.train:
        print("Preparing tokenizer...")
        tokenizer = train_dataset.prepare_tokenizer()
        tokenizer.save(os.path.join(args.output_dir, 'tokenizer.json'))
    else:
        tokenizer.load(os.path.join(args.output_dir, 'tokenizer.json'))
    
    # Create TF datasets
    train_tf_dataset = train_dataset.create_tf_dataset(shuffle=True)
    val_tf_dataset = val_dataset.create_tf_dataset(shuffle=False)
    
    # Create model
    model = create_pix2code_model(
        vocab_size=tokenizer.vocab_size,
        img_shape=tuple(config['data']['img_size']) + (3,),
        embedding_dim=config['model']['embedding_dim'],
        lstm_units=config['model']['lstm_units']
    )
    
    # Create trainer
    trainer = Pix2CodeTrainer(
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoints')
    )
    
    # Print model summary
    model.summary()
    
    # Train model if requested
    if args.train:
        print("Starting training...")
        trainer.train(
            train_dataset=train_tf_dataset,
            val_dataset=val_tf_dataset,
            epochs=config['training']['epochs'],
            save_freq=config['training']['save_freq']
        )
    
    # Save model
    if args.save_model:
        print("Saving model...")
        model.save(os.path.join(args.output_dir, 'model'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pix2Code Training Script')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Path to output directory')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the model after training')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args) 