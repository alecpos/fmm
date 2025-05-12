import torch
from pix2code_project.models.ui2code import UI2Code
from pix2code_project.utils.preprocessing import ImagePreprocessor
from pix2code_project.utils.tokenizer import AdaptiveTokenizer
import yaml
import os

def test_model(image_path, checkpoint_path):
    # Load configuration
    config_path = os.path.join('pix2code_project', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    preprocessor = ImagePreprocessor()
    tokenizer = AdaptiveTokenizer()
    
    # Load model
    model = UI2Code(
        visual_encoder=config['visual_encoder'],
        layout_processor=config['layout_processor'],
        code_decoder=config['code_decoder']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Process image
    image = preprocessor.preprocess(image_path)
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Generate code
    with torch.no_grad():
        output = model.generate(image)
    
    # Decode output
    code = tokenizer.decode(output[0])
    
    return code

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    code = test_model(args.image, args.checkpoint)
    print("Generated Code:")
    print(code) 