import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Validator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.smooth = SmoothingFunction()
    
    def validate_batch(self, batch):
        """Validate a batch of data and return metrics."""
        self.model.eval()
        with torch.no_grad():
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
            
            # Calculate metrics
            metrics = {
                'loss': outputs.loss.item(),
                'bleu': self._calculate_bleu(outputs.logits, input_ids),
                'exact_match': self._calculate_exact_match(outputs.logits, input_ids)
            }
            
            return metrics
    
    def _calculate_bleu(self, predictions, targets):
        """Calculate BLEU score."""
        bleu_scores = []
        for pred, target in zip(predictions, targets):
            # Convert to lists of tokens
            pred_tokens = self.tokenizer.decode(pred.argmax(dim=-1)).split()
            target_tokens = self.tokenizer.decode(target).split()
            
            # Calculate BLEU
            score = sentence_bleu([target_tokens], pred_tokens, smoothing_function=self.smooth.method1)
            bleu_scores.append(score)
        
        return np.mean(bleu_scores)
    
    def _calculate_exact_match(self, predictions, targets):
        """Calculate exact match accuracy."""
        pred_ids = predictions.argmax(dim=-1)
        matches = (pred_ids == targets).all(dim=1).float()
        return matches.mean().item() 