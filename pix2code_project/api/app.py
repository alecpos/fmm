from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io
from ..utils.preprocessing import ImagePreprocessor
from ..models.ui2code import UI2Code

app = FastAPI()

# Initialize model and preprocessor
preprocessor = ImagePreprocessor()
model = UI2Code.load_from_checkpoint('checkpoints/best_model.ckpt')
model.eval()

@app.post("/generate")
async def generate_code(image: UploadFile = File(...)):
    # Read and preprocess image
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_tensor = preprocessor.preprocess(image)
    
    # Generate code
    with torch.no_grad():
        output = model.generate(image_tensor.unsqueeze(0))
    
    # Decode output
    code = model.tokenizer.decode(output[0])
    
    return {"code": code}

@app.post("/batch_generate")
async def batch_generate(images: list[UploadFile] = File(...)):
    # Process multiple images
    results = []
    for image in images:
        result = await generate_code(image)
        results.append(result)
    
    return {"results": results} 