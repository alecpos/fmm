import json
import os
from transformers import AutoTokenizer

class AdaptiveTokenizer:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def encode(self, texts):
        """Encode text to token IDs."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def save(self, path):
        """Save tokenizer configuration."""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path):
        """Load tokenizer configuration."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 