import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class LayoutAttention(nn.Module):
    def __init__(self, in_channels=4, out_channels=768, num_heads=8, dropout=0.1):
        """Initialize the layout attention module.
        
        Args:
            in_channels (int): Number of input channels (default: 4 for x,y,width,height)
            out_channels (int): Number of output channels
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear projection for layout features
        self.layout_proj = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, layouts):
        """Process layout features.
        
        Args:
            layouts (torch.Tensor): Layout features of shape (batch_size, 1, 4)
            
        Returns:
            torch.Tensor: Processed layout features of shape (batch_size, 1, out_channels)
        """
        # Reshape layout features for linear projection
        batch_size = layouts.size(0)
        x = layouts.view(batch_size, -1)  # Shape: (batch_size, 4)
        
        # Project layout features
        x = self.layout_proj(x)  # Shape: (batch_size, out_channels)
        
        # Add sequence dimension back
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, out_channels)
        
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Add residual connection and normalize
        x = self.norm(x + self.dropout(attn_output))
        
        return x 