from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import base64
from io import BytesIO
from PIL import Image
import os

from app.services.huggingface import HuggingFaceService
from app.utils.image_utils import process_image, image_to_base64

app = FastAPI(
    title="Affecto Backend API",
    description="Emotion transformation API using MagicFace",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize HuggingFace service
hf_service = HuggingFaceService()

@app.get("/")
async def root():
    return {
        "message": "Affecto Backend API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "transform": "/transform"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    hf_healthy = await hf_service.health_check()
    return {
        "status": "healthy",
        "service": "affecto-backend",
        "hf_service": "connected" if hf_healthy else "disconnected",
        "hf_space": hf_service.api_url
    }

@app.post("/transform")
async def transform_emotion(
    image: UploadFile = File(...),
    au_params: str = Form(...)
):
    """
    Transform facial emotion using MagicFace model
    
    Args:
        image: Input image file
        au_params: JSON string of Action Unit parameters
        
    Returns:
        JSON with base64 encoded transformed image
    """
    try:
        print(f"📥 Received transform request")
        
        # Parse AU parameters
        try:
            au_dict = json.loads(au_params)
            print(f"🎭 AU params: {au_dict}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid AU parameters JSON")
        
        # Validate image
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_bytes = await image.read()
        input_image = Image.open(BytesIO(image_bytes))
        print(f"📸 Image size: {input_image.size}")
        
        # Preprocess image
        processed_image = process_image(input_image)
        
        # Call HuggingFace inference
        print(f"🚀 Sending to HF Space...")
        transformed_image = await hf_service.transform_emotion(
            image=processed_image,
            au_params=au_dict
        )
        
        # Convert to base64
        image_base64 = image_to_base64(transformed_image)
        
        print("✅ Transformation complete")
        
        return JSONResponse(content={
            "success": True,
            "transformed_image": image_base64,
            "au_params": au_dict,
            "message": "Transformation successful"
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error in transform endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)

