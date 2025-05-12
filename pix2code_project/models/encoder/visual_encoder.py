import torch
import torch.nn as nn
import timm

class VisualEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, freeze_backbone=False):
        super().__init__()
        # Create model with classification head to get the correct output dimension
        temp_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1000  # Use default number of classes
        )
        
        # Get the output dimension based on model type
        if hasattr(temp_model, 'embed_dim'):
            # For ViT models
            self.out_channels = temp_model.embed_dim
        else:
            # For ResNet models
            if hasattr(temp_model, 'fc'):
                self.out_channels = temp_model.fc.in_features
            elif hasattr(temp_model, 'head'):
                self.out_channels = temp_model.head.in_features
            else:
                # Try to get the output dimension from the last layer
                last_layer = list(temp_model.children())[-1]
                if hasattr(last_layer, 'in_features'):
                    self.out_channels = last_layer.in_features
                else:
                    raise ValueError(f"Could not determine output dimension for model {model_name}")
        
        # Now create the actual model without classification head
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        print(f"Initialized {model_name} with output dimension: {self.out_channels}")
    
    def forward(self, x):
        """Forward pass through the visual encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Visual features of shape (batch_size, seq_len, embed_dim)
        """
        # Get features from the last layer
        features = self.model.forward_features(x)
        
        # Add sequence dimension if not present
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        
        return features 