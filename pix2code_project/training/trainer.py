import tensorflow as tf
import time
import os
from .loss_functions import masked_loss
from .metrics import BLEU4Metric, ExactMatchMetric
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

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

class Pix2CodeTrainer:
    def __init__(self, model, tokenizer, checkpoint_dir='./checkpoints'):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Setup optimizer with learning rate schedule
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=3e-4,
            decay_steps=10000,
            decay_rate=0.96
        )
        
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.05
        )
        
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.bleu4 = BLEU4Metric()
        self.exact_match = ExactMatchMetric()
        
        # Checkpoint manager
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_dir, max_to_keep=5
        )
        
        # Load checkpoint if available
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Restored from checkpoint: {self.ckpt_manager.latest_checkpoint}")
    
    @tf.function
    def train_step(self, img_batch, target_batch):
        """Execute a single training step."""
        # Create input and target sequences for teacher forcing
        # Input sequence is target with last token removed
        # Target sequence is target with first token removed
        inp_seq = target_batch[:, :-1]  # Input sequence
        tar_seq = target_batch[:, 1:]   # Target sequence
        
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model([img_batch, inp_seq], training=True)
            
            # Calculate loss with masking for padding
            loss = masked_loss(tar_seq, predictions)
        
        # Compute gradients and apply
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients to avoid explosion
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        self.bleu4.update_state(tar_seq, predictions)
        self.exact_match.update_state(tar_seq, predictions)
        
        return loss
    
    @tf.function
    def val_step(self, img_batch, target_batch):
        """Execute a single validation step."""
        # Create input and target sequences for teacher forcing
        inp_seq = target_batch[:, :-1]  # Input sequence
        tar_seq = target_batch[:, 1:]   # Target sequence
        
        # Forward pass
        predictions = self.model([img_batch, inp_seq], training=False)
        
        # Calculate loss with masking for padding
        loss = masked_loss(tar_seq, predictions)
        
        # Update metrics
        self.val_loss.update_state(loss)
        
        return loss
    
    def train(self, train_dataset, val_dataset, epochs=150, save_freq=5):
        """Train the model for specified number of epochs."""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Reset metrics
            self.train_loss.reset_states()
            self.val_loss.reset_states()
            self.bleu4.reset_states()
            self.exact_match.reset_states()
            
            # Training loop
            for step, (img_batch, target_batch) in enumerate(train_dataset):
                loss = self.train_step(img_batch, target_batch)
                
                if step % 100 == 0:
                    print(f"Epoch {epoch+1} Step {step} Loss {loss.numpy():.4f}")
            
            # Validation loop
            for img_batch, target_batch in val_dataset:
                val_loss = self.val_step(img_batch, target_batch)
            
            # Print epoch results
            train_loss = self.train_loss.result()
            val_loss = self.val_loss.result()
            bleu4 = self.bleu4.result()
            exact_match = self.exact_match.result()
            
            print(f"Epoch {epoch+1}/{epochs}, Time: {time.time()-start_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"BLEU-4: {bleu4:.4f}")
            print(f"Exact Match: {exact_match:.4f}")
            
            # Save checkpoint if better model found
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = self.ckpt_manager.save()
                print(f"Saved checkpoint for epoch {epoch+1}: {ckpt_path}")
            
            # Save periodically
            if (epoch + 1) % save_freq == 0:
                ckpt_path = self.ckpt_manager.save()
                print(f"Periodic save for epoch {epoch+1}: {ckpt_path}")

class UIToCodeTrainer:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=3e-5,
            weight_decay=1e-6
        )
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            layouts = batch['layout'].to(self.device) if batch['layout'] is not None else None
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layouts=layouts,
                labels=input_ids  # Use input_ids as labels for next token prediction
            )
            
            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0
        } 

class Trainer:
    def __init__(self, model, optimizer, scheduler, device, output_dir):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.writer = SummaryWriter(output_dir)
        self.best_bleu = 0.0
        self.start_time = time.time()
        
        # Log model architecture
        logger.info(f"Model architecture:\n{model}")
        logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_steps = len(train_loader)
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=True)
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            image = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            layout = batch['layout'].to(self.device) if batch['layout'] is not None else None
            
            # Forward pass
            outputs = self.model(
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layout=layout,
                labels=input_ids  # Use input_ids as labels for next token prediction
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(step+1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to tensorboard
            global_step = epoch * total_steps + step
            self.writer.add_scalar('train/loss', loss.item(), global_step)
            self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
            
            # Log gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, global_step)
            
            # Log every 10 steps
            if step % 10 == 0:
                logger.info(f'Epoch {epoch} - Step {step}/{total_steps} - '
                          f'Loss: {loss.item():.4f} - '
                          f'Avg Loss: {total_loss/(step+1):.4f} - '
                          f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
        
        # Step the scheduler at the end of each epoch
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Log epoch summary
        avg_loss = total_loss / total_steps
        logger.info(f'Epoch {epoch} completed - '
                   f'Average Loss: {avg_loss:.4f} - '
                   f'Time: {time.time() - self.start_time:.2f}s')
        
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_steps = len(val_loader)
        
        # Create progress bar
        pbar = tqdm(val_loader, desc=f'Validation {epoch}', leave=True)
        
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                # Move batch to device
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                layouts = batch['layout'].to(self.device) if batch['layout'] is not None else None
                
                # Forward pass
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layouts=layouts,
                    labels=input_ids
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(step+1):.4f}'
                })
        
        # Calculate average loss
        avg_loss = total_loss / total_steps
        
        # Log validation results
        logger.info(f'Validation Epoch {epoch} - '
                   f'Average Loss: {avg_loss:.4f} - '
                   f'Time: {time.time() - self.start_time:.2f}s')
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, bleu_score, is_best=False):
        """Save model checkpoint."""
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'bleu_score': bleu_score
            }
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
            try:
                torch.save(checkpoint, checkpoint_path)
                logger.info(f'Saved checkpoint to {checkpoint_path}')
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {str(e)}")
                # Try saving with a different name
                backup_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}_backup.pt')
                try:
                    torch.save(checkpoint, backup_path)
                    logger.info(f'Saved backup checkpoint to {backup_path}')
                except Exception as e2:
                    logger.error(f"Failed to save backup checkpoint: {str(e2)}")
            
            # Save best model
            if is_best:
                best_path = os.path.join(self.output_dir, 'best_model.pt')
                try:
                    torch.save(checkpoint, best_path)
                    logger.info(f'Saved best model with BLEU score: {bleu_score:.4f}')
                except Exception as e:
                    logger.error(f"Failed to save best model: {str(e)}")
                    # Try saving with a different name
                    backup_best_path = os.path.join(self.output_dir, 'best_model_backup.pt')
                    try:
                        torch.save(checkpoint, backup_best_path)
                        logger.info(f'Saved backup best model to {backup_best_path}')
                    except Exception as e2:
                        logger.error(f"Failed to save backup best model: {str(e2)}")
        
        except Exception as e:
            logger.error(f"Error in save_checkpoint: {str(e)}")
            # Continue training even if saving fails
            pass
    
    def log_hyperparameters(self, config):
        """Log hyperparameters to tensorboard."""
        # Flatten nested config
        flat_config = {}
        def flatten_dict(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten_dict(v, f'{prefix}{k}/')
                else:
                    flat_config[f'{prefix}{k}'] = v
        
        flatten_dict(config)
        
        # Log to tensorboard
        for k, v in flat_config.items():
            if isinstance(v, (int, float, str, bool)):
                self.writer.add_text(f'config/{k}', str(v))
        
        logger.info("Logged hyperparameters to tensorboard") 