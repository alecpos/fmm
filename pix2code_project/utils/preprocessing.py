import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImagePreprocessor:
    def __init__(self, img_size=(224, 224)):
        """Initialize the image preprocessor.
        
        Args:
            img_size (tuple): Target image size (height, width)
        """
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image):
        """Preprocess an image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        return self.transform(image)
    
    def _apply_histogram_equalization(self, image):
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply histogram equalization to Y channel
        img_array[:,:,0] = self._equalize_histogram(img_array[:,:,0])
        
        return Image.fromarray(img_array)
    
    def _equalize_histogram(self, channel):
        hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        cdf_m = np.ma.masked_equal(cdf_normalized, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        return cdf[channel] 