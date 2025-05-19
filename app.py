from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2
import base64
import gc
import os
import logging
import asyncio
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Retinal OCT Image Analyzer API",
              description="API for analyzing OCT retinal images and predicting disease categories",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
IMG_SIZE = 224
device = torch.device('cpu')  # Explicitly use CPU for Heroku
model = None
is_model_loaded = False
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Define transform pipeline
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Define model class - simplified
class RetinalModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Use ResNet18 which is smaller and loads faster
        base = models.resnet18(pretrained=False)  # No pretrained weights to save startup time
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Load model in background to avoid timeout
@app.on_event("startup")
async def startup_event():
    # Start background task to load model
    asyncio.create_task(load_model_async())

async def load_model_async():
    global model, is_model_loaded
    try:
        logger.info("Starting model loading in background")
        
        # Initialize the model
        model = RetinalModel(num_classes=len(CLASS_NAMES)).to(device)
        logger.info("Model initialized")
        
        # Load checkpoint if available
        checkpoint_path = "best_model.pth"
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                logger.info("Model weights loaded")
            except Exception as e:
                logger.error(f"Error loading weights: {e}")
                # Continue with uninitialized weights
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}, using random weights")
        
        # Set model to evaluation mode
        model.eval()
        is_model_loaded = True
        logger.info("Model ready for inference")
        
    except Exception as e:
        logger.error(f"Error in async model loading: {e}")
        is_model_loaded = False

def generate_grad_cam(model, img_tensor, target_class=None):
    """Generate Grad-CAM with memory optimization"""
    img_tensor = img_tensor.clone().detach().requires_grad_(True)
    
    # Store only what we need
    activations = None
    gradients = None
    
    def forward_hook(_, __, out):
        nonlocal activations
        activations = out.detach()
        return None
    
    def backward_hook(_, __, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()
        return None
    
    # Get target layer - last layer in features
    target_layer = list(model.features.children())[-1]
    
    # Register hooks
    hooks = [
        target_layer.register_forward_hook(forward_hook),
        target_layer.register_full_backward_hook(backward_hook)
    ]
    
    # Forward pass
    output = model(img_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Zero gradients and backward pass
    model.zero_grad()
    score = output[0, target_class]
    score.backward()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate weights and CAM
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1))
    
    # Get current dimensions
    cam_height, cam_width = cam.shape[-2:]
    
    # Resize CAM to original image size - FIXED: proper handling of dimensions
    cam_resized = torch.nn.functional.interpolate(
        cam.unsqueeze(0),  # Add batch dimension
        size=(IMG_SIZE, IMG_SIZE),
        mode='bilinear',
        align_corners=False
    ).squeeze()  # Remove batch dimension
    
    cam_np = cam_resized.cpu().numpy()
    
    # Normalize the CAM
    if cam_np.max() != cam_np.min():
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
    else:
        cam_np = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    
    # Clean up
    del weights, activations, gradients, output, score, cam, cam_resized
    gc.collect()
    
    return cam_np, target_class

def image_to_base64(img_array):
    """Convert image array to base64 string"""
    success, encoded_img = cv2.imencode('.png', img_array, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if success:
        result = base64.b64encode(encoded_img).decode('utf-8')
        del encoded_img
        return result
    return None

def cleanup():
    """Force garbage collection"""
    gc.collect()

# Middleware to handle model not loaded case
@app.middleware("http")
async def check_model_loading(request: Request, call_next):
    # Skip check for non-prediction endpoints
    if request.url.path in ['/', '/health', '/docs', '/openapi.json', '/redoc']:
        return await call_next(request)
    
    # Check if model is loaded
    if not is_model_loaded:
        if request.url.path == '/predict/':
            # Return a 503 Service Unavailable if trying to predict
            return JSONResponse(
                status_code=503,
                content={"detail": "Model is still loading. Please try again in a few moments."}
            )
    
    # Continue with the request
    return await call_next(request)

@app.post("/predict/", response_model=Dict)
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Endpoint to predict and generate Grad-CAM for uploaded OCT image"""
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    try:
        # Read the image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Original image for display
        orig_img = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        
        # Process for model
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Apply temperature scaling
            temperature = 0.5
            scaled_outputs = outputs / temperature
            
            probs = torch.softmax(scaled_outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = CLASS_NAMES[pred_idx]
            confidence = probs[0, pred_idx].item()
            
            # Get prediction probabilities for all classes
            class_probs = {CLASS_NAMES[i]: float(probs[0, i].item()) for i in range(len(CLASS_NAMES))}
            
            # Clean up
            del outputs, scaled_outputs, probs
        
        # Generate Grad-CAM
        cam, _ = generate_grad_cam(model, img_tensor, target_class=pred_idx)
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
        
        # Convert images to base64
        orig_b64 = image_to_base64(cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        heatmap_b64 = image_to_base64(heatmap)
        overlay_b64 = image_to_base64(overlay)
        
        # Clean up
        del img_tensor, cam, heatmap, overlay, orig_img
        
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
        
        # Schedule cleanup
        background_tasks.add_task(cleanup)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if is_model_loaded else "starting",
        "model_loaded": is_model_loaded,
        "device": str(device)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OCT Image Analysis API. POST an image to /predict/ to get predictions.",
        "model_status": "loaded" if is_model_loaded else "loading"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)