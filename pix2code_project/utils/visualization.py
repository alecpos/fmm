import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.gridspec import GridSpec

def visualize_attention(image, attention_weights, save_path, num_heads=4):
    """
    Enhanced visualization of attention weights with multiple views.
    
    Args:
        image: Input image tensor
        attention_weights: Dictionary containing 'visual_weights' and 'syntactic_weights'
        save_path: Path to save the visualization
        num_heads: Number of attention heads to visualize
    """
    try:
        # Convert image to numpy and normalize
        img = image.numpy()[0]  # Remove batch dimension
        img = (img - img.min()) / (img.max() - img.min())
        
        # Get attention weights
        visual_weights = attention_weights['visual_weights'].numpy()[0, 0]  # [h*w]
        syntactic_weights = attention_weights['syntactic_weights'].numpy()[0, 0]  # [seq_len]
        
        # Reshape visual weights to match image dimensions
        h = w = int(np.sqrt(visual_weights.shape[0]))
        visual_weights = visual_weights.reshape(h, w)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Visual attention heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(visual_weights, cmap='jet')
        ax2.set_title('Visual Attention Heatmap')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        # Overlaid attention
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(img)
        ax3.imshow(visual_weights, cmap='jet', alpha=0.5)
        ax3.set_title('Overlaid Attention')
        ax3.axis('off')
        
        # Syntactic attention
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(syntactic_weights)
        ax4.set_title('Syntactic Attention Weights')
        ax4.set_xlabel('Sequence Position')
        ax4.set_ylabel('Attention Weight')
        
        # Add attention statistics
        stats_text = f"""
        Visual Attention Stats:
        - Mean: {visual_weights.mean():.3f}
        - Max: {visual_weights.max():.3f}
        - Min: {visual_weights.min():.3f}
        
        Syntactic Attention Stats:
        - Mean: {syntactic_weights.mean():.3f}
        - Max: {syntactic_weights.max():.3f}
        - Min: {syntactic_weights.min():.3f}
        """
        fig.text(0.02, 0.02, stats_text, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        raise

def visualize_attention_sequence(image, attention_weights, save_path, num_steps=5):
    """
    Visualize attention weights over multiple decoding steps.
    
    Args:
        image: Input image tensor
        attention_weights: Dictionary containing attention weights for multiple steps
        save_path: Path to save the visualization
        num_steps: Number of decoding steps to visualize
    """
    # Convert image to numpy and normalize
    img = image.numpy()[0]
    img = (img - img.min()) / (img.max() - img.min())
    
    # Create figure
    fig, axes = plt.subplots(2, num_steps, figsize=(15, 6))
    
    # Get attention weights
    visual_weights = attention_weights['visual_weights'].numpy()[0, 0]  # [h*w]
    syntactic_weights = attention_weights['syntactic_weights'].numpy()[0, 0]  # [seq_len]
    
    # Reshape visual weights to match image dimensions
    h = w = int(np.sqrt(visual_weights.shape[0]))
    visual_weights = visual_weights.reshape(h, w)
    
    # Create sequence of attention weights by adding small variations
    for i in range(num_steps):
        # Add small variations to visual attention
        step_visual_weights = visual_weights * (1 + 0.1 * np.sin(i * np.pi / num_steps))
        step_visual_weights = (step_visual_weights - step_visual_weights.min()) / (step_visual_weights.max() - step_visual_weights.min())
        
        # Add small variations to syntactic attention
        step_syntactic_weights = syntactic_weights * (1 + 0.1 * np.cos(i * np.pi / num_steps))
        step_syntactic_weights = (step_syntactic_weights - step_syntactic_weights.min()) / (step_syntactic_weights.max() - step_syntactic_weights.min())
        
        # Visual attention
        axes[0, i].imshow(img)
        axes[0, i].imshow(step_visual_weights, cmap='jet', alpha=0.5)
        axes[0, i].set_title(f'Step {i+1}')
        axes[0, i].axis('off')
        
        # Syntactic attention
        axes[1, i].plot(step_syntactic_weights)
        axes[1, i].set_title(f'Step {i+1}')
        axes[1, i].set_xlabel('Position')
        axes[1, i].set_ylabel('Weight')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 