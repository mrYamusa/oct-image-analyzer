from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2
import base64
import matplotlib.pyplot as plt
from typing import Dict, List
import os
from huggingface_hub import hf_hub_download

# Define the model architecture
class RetinalModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Load the torchvision EfficientNet-B3 with pretrained weights
        # Use a different approach to avoid hash verification issues
        # base = models.efficientnet_b3(pretrained=False)
        base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Freeze feature extractor
        for p in base.features.parameters():
            p.requires_grad = False
        
        # Replace classifier with custom head
        in_feats = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(in_feats, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Expose pieces for Grad-CAM
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Class names mapping
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Initialize FastAPI app
app = FastAPI(title="Retinal OCT Image Analyzer API",
              description="API for analyzing OCT retinal images and predicting disease categories",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables
IMG_SIZE = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Using FastAPI lifespan instead of on_event which is deprecated
"""
@app.on_event("startup")
async def load_model():
    global model
    # Initialize the model
    model = RetinalModel(num_classes=len(CLASS_NAMES)).to(device)
    
    # Load checkpoint (update path for your deployment)
    # checkpoint_path = "./best_model.pth"
    # Load checkpoint (update path for your deployment)
    checkpoint_path = os.environ.get("MODEL_PATH", "./best_model.pth")

    try:
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint file {checkpoint_path} not found. Using randomly initialized weights.")
        model.eval()  # Set to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        # Initialize with random weights if checkpoint not found
        model.eval()
    """
@app.on_event("startup")
async def load_model():
    global model
    model = RetinalModel(num_classes=len(CLASS_NAMES)).to(device)

    checkpoint_path = hf_hub_download(
        repo_id="mryamusa/oct-image-analyzer",
        filename="best_model.pth",
        repo_type="model"
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

def generate_grad_cam(model, img_tensor, target_class=None):
    """Generate Grad-CAM for the given image tensor"""
    # Make sure tensor requires grad
    img_tensor = img_tensor.clone().detach().requires_grad_(True)
    
    # Get the feature module
    feature_module = model.features
    layers = list(feature_module.children())
    target_layer = layers[-1]  # last conv block
    
    # Hook definition
    activations, gradients = [], []
    def forward_hook(_, __, out):
        out.requires_grad_(True)
        activations.append(out)
        return out
    
    def backward_hook(_, __, grad_out):
        gradients.append(grad_out[0])
        return None
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(img_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Zero gradients and backward pass
    model.zero_grad()
    score = output[0, target_class]
    score.backward(retain_graph=True)
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Get activations and gradients
    act = activations[0].detach()
    grad = gradients[0].detach()
    
    # Calculate weights and CAM
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam, target_class

def image_to_base64(img_array):
    """Convert image array to base64 string"""
    success, encoded_img = cv2.imencode('.png', img_array)
    if success:
        return base64.b64encode(encoded_img).decode('utf-8')
    return None

@app.post("/predict/", response_model=Dict)
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict and generate Grad-CAM for uploaded OCT image"""
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Original image for display
        orig_img = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        
        # Process for model
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = CLASS_NAMES[pred_idx]
            confidence = probs[0, pred_idx].item()
        
        # Generate Grad-CAM
        cam, _ = generate_grad_cam(model, img_tensor, target_class=pred_idx)
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
        
        # Get prediction probabilities for all classes
        class_probs = {CLASS_NAMES[i]: float(probs[0, i].item()) for i in range(len(CLASS_NAMES))}
        
        # Convert images to base64 for response
        orig_b64 = image_to_base64(cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        heatmap_b64 = image_to_base64(heatmap)
        overlay_b64 = image_to_base64(overlay)
        
        # Prepare response
        response = {
            "prediction": pred_label,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "images": {
                "original": orig_b64,
                "heatmap": heatmap_b64,
                "overlay": overlay_b64
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return FileResponse("static/index.html")
    # return {"message": "OCT Image Analysis API. POST an image to /predict/ to get predictions."}

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable for Heroku compatibility
    port = int(os.environ.get("PORT", 8000))
    # Run app with uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)