import torch
import torchvision.transforms as T
import random

class UIAugmentor:
    def __init__(self, p=0.5):
        self.p = p
        self.transforms = T.Compose([
            T.RandomPerspective(distortion_scale=0.2, p=p),
            T.RandomAffine(degrees=5, translate=(0.1, 0.1), p=p),
            T.ColorJitter(brightness=0.2, contrast=0.2, p=p),
            T.RandomHorizontalFlip(p=p)
        ])
    
    def augment(self, image, layout):
        """Apply augmentations to image and adjust layout accordingly"""
        if random.random() < self.p:
            # Apply transforms
            augmented_image = self.transforms(image)
            
            # Adjust layout coordinates based on transforms
            augmented_layout = self._adjust_layout(layout, image.size)
            
            return augmented_image, augmented_layout
        
        return image, layout
    
    def _adjust_layout(self, layout, original_size):
        """Adjust layout coordinates based on applied transforms"""
        # This is a simplified version - you'll need to implement
        # the actual coordinate transformation logic based on the
        # specific transforms applied
        return layout 