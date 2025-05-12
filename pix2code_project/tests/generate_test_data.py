import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_mockup_image(output_path, size=(256, 256)):
    """Create a simple mockup image for testing."""
    # Create a white background
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw a header
    draw.rectangle([(50, 50), (206, 100)], fill='#f0f0f0')
    draw.text((80, 65), 'Welcome', fill='black')
    
    # Draw a button
    draw.rectangle([(80, 150), (176, 190)], fill='#007bff')
    draw.text((100, 160), 'Click Me', fill='white')
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

if __name__ == '__main__':
    # Create test images
    create_mockup_image('pix2code_project/data/test/images/sample1.png')
    print("Test data generated successfully!") 