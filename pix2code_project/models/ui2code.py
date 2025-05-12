import torch
import torch.nn as nn
from .encoder.visual_encoder import VisualEncoder
from .attention.layout_attention import LayoutAttention
from .decoder.code_decoder import CodeDecoder

class UI2Code(nn.Module):
    def __init__(self, visual_encoder_config, layout_processor_config, code_decoder_config):
        """Initialize the UI2Code model.
        
        Args:
            visual_encoder_config (dict): Configuration for the visual encoder
            layout_processor_config (dict): Configuration for the layout processor
            code_decoder_config (dict): Configuration for the code decoder
        """
        super().__init__()
        
        # Initialize visual encoder
        self.visual_encoder = VisualEncoder(
            model_name=visual_encoder_config['model_name'],
            pretrained=visual_encoder_config.get('pretrained', True),
            freeze_backbone=visual_encoder_config.get('freeze_backbone', False)
        )
        
        # Initialize layout processor
        self.layout_processor = LayoutAttention(
            in_channels=4,  # x, y, width, height
            out_channels=layout_processor_config['hidden_size'],
            num_heads=layout_processor_config['num_attention_heads'],
            dropout=layout_processor_config.get('dropout', 0.1)
        )
        
        # Initialize code decoder
        self.code_decoder = CodeDecoder(
            model_name=code_decoder_config['model_name']
        )
        
        # Add projection layer to match GPT2's hidden size
        self.visual_projection = nn.Linear(
            self.visual_encoder.out_channels,  # 2048 from ResNet50
            self.code_decoder.decoder.config.hidden_size  # 768 from GPT2
        )
        
        # Add final projection layer for combined features
        self.final_projection = nn.Linear(
            self.code_decoder.decoder.config.hidden_size + layout_processor_config['hidden_size'],  # 768 + 256
            self.code_decoder.decoder.config.hidden_size  # 768
        )
    
    def forward(self, image, input_ids=None, attention_mask=None, layout=None, labels=None):
        """Forward pass through the model.
        
        Args:
            image (torch.Tensor): Input image tensor
            input_ids (torch.Tensor, optional): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            layout (torch.Tensor, optional): Layout information
            labels (torch.Tensor, optional): Labels for loss calculation
        
        Returns:
            dict: Model outputs including logits and loss if labels are provided
        """
        # Encode image
        visual_features = self.visual_encoder(image)  # Shape: (batch_size, channels, height, width)
        
        # Reshape visual features to match layout features
        batch_size = visual_features.size(0)
        visual_features = visual_features.view(batch_size, -1, visual_features.size(1))  # Shape: (batch_size, seq_len, channels)
        
        # Project visual features to match GPT2's hidden size
        visual_features = self.visual_projection(visual_features)  # Shape: (batch_size, seq_len, 768)
        
        # Process layout if available
        if layout is not None:
            layout_features = self.layout_processor(layout)  # Shape: (batch_size, 1, out_channels)
            # Repeat layout features to match visual features' sequence length
            layout_features = layout_features.repeat(1, visual_features.size(1), 1)
            # Combine visual and layout features
            combined_features = torch.cat([visual_features, layout_features], dim=-1)  # Shape: (batch_size, seq_len, 1024)
            # Project combined features back to GPT2's hidden size
            combined_features = self.final_projection(combined_features)  # Shape: (batch_size, seq_len, 768)
        else:
            combined_features = visual_features
        
        # Decode to code
        outputs = self.code_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=combined_features,
            labels=labels
        )
        
        return outputs 