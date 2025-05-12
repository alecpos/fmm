import torch
import torch.nn as nn
import logging
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

# Configure logging
logging.getLogger("transformers").setLevel(logging.ERROR)  # Suppress warnings

class CodeDecoder(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        # Load tokenizer first to get vocab size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token
        
        # Load and modify config to enable cross-attention
        config = AutoConfig.from_pretrained(model_name)
        config.add_cross_attention = True
        config.is_decoder = True  # Explicitly set as decoder
        config.use_cache = False  # Disable caching for training
        
        # Task-specific configuration
        config.task_specific_params = {
            "text-generation": {
                "do_sample": True,
                "max_length": 1024,  # Increased for longer code sequences
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            }
        }
        
        # Initialize model with modified config
        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True  # Handle size mismatches gracefully
        )
        
        # Initialize cross-attention layers with proper scaling
        for layer in self.decoder.transformer.h:
            if hasattr(layer, 'crossattention'):
                # Initialize weights with scaled Xavier uniform
                nn.init.xavier_uniform_(layer.crossattention.c_attn.weight, gain=0.02)
                nn.init.xavier_uniform_(layer.crossattention.c_proj.weight, gain=0.02)
                nn.init.xavier_uniform_(layer.crossattention.q_attn.weight, gain=0.02)
                
                # Initialize biases to zero
                if layer.crossattention.c_attn.bias is not None:
                    nn.init.zeros_(layer.crossattention.c_attn.bias)
                if layer.crossattention.c_proj.bias is not None:
                    nn.init.zeros_(layer.crossattention.c_proj.bias)
                if layer.crossattention.q_attn.bias is not None:
                    nn.init.zeros_(layer.crossattention.q_attn.bias)
                
                # Initialize layer norm weights to one and biases to zero
                if hasattr(layer.crossattention, 'ln_cross_attn'):
                    nn.init.ones_(layer.crossattention.ln_cross_attn.weight)
                    nn.init.zeros_(layer.crossattention.ln_cross_attn.bias)
        
        # Set model to training mode
        self.decoder.train()
        
        print(f"Initialized CodeDecoder with {model_name}")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        print(f"Model config: {config}")
        print(f"Model is in training mode: {self.decoder.training}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
    def forward(self, input_ids, attention_mask, encoder_hidden_states, labels=None):
        """Forward pass through the decoder.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            encoder_hidden_states (torch.Tensor): Hidden states from the encoder
            labels (torch.Tensor, optional): Target token IDs for loss calculation
            
        Returns:
            transformers.modeling_outputs.CausalLMOutputWithCrossAttentions: Model outputs
        """
        # Ensure model is in training mode
        self.decoder.train()
        
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels,  # Pass labels for loss calculation
            use_cache=False  # Disable caching for training
        )
        
        return outputs 