"""
Affecto Render API - Bridge between Flutter and HuggingFace Space
Author: gauravvjhaa
Date: 2025-11-12
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import io
import base64
from PIL import Image
import json
from typing import Optional, Dict
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Affecto API",
    description="API bridge for Affecto emotion transformation service",
    version="1.0.0"
)

# CORS Configuration for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HuggingFace Space Configuration
HF_SPACE_URL = "https://gauravvjhaa-affecto-inference.hf.space"
HF_API_URL = f"{HF_SPACE_URL}/api/predict"

# Emotion presets
EMOTION_PRESETS = {
    "happy": {"AU6": 4.0, "AU12": 4.0},
    "sad": {"AU1": 4.0, "AU4": 4.0, "AU15": 4.0},
    "surprised": {"AU1": 4.5, "AU2": 4.0, "AU5": 4.5, "AU26": 4.0},
    "angry": {"AU4": 4.0, "AU5": 3.5, "AU7": 3.5, "AU23": 2.5},
    "fearful": {"AU1": 3.5, "AU2": 3.5, "AU4": 2.5, "AU5": 4.0, "AU20": 3.0},
    "neutral": {},
}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Affecto API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "huggingface_space": HF_SPACE_URL
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check if HF Space is accessible
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(HF_SPACE_URL)
            hf_status = "operational" if response.status_code == 200 else "degraded"
    except Exception as e:
        hf_status = "unavailable"
        logger.error(f"HF Space health check failed: {str(e)}")
    
    return {
        "api_status": "operational",
        "huggingface_space_status": hf_status,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/emotions")
async def list_emotions():
    """List available emotion presets"""
    return {
        "emotions": [
            {
                "id": "happy",
                "name": "Happy",
                "description": "Big smile",
                "au_params": EMOTION_PRESETS["happy"]
            },
            {
                "id": "sad",
                "name": "Sad",
                "description": "Sadness expression",
                "au_params": EMOTION_PRESETS["sad"]
            },
            {
                "id": "surprised",
                "name": "Surprised",
                "description": "Shock/Surprise",
                "au_params": EMOTION_PRESETS["surprised"]
            },
            {
                "id": "angry",
                "name": "Angry",
                "description": "Anger expression",
                "au_params": EMOTION_PRESETS["angry"]
            },
            {
                "id": "fearful",
                "name": "Fearful",
                "description": "Fear expression",
                "au_params": EMOTION_PRESETS["fearful"]
            }
        ]
    }


@app.post("/transform")
async def transform_emotion(
    image: UploadFile = File(...),
    emotion: str = Form(default="happy"),
    steps: int = Form(default=30),
    au_params: Optional[str] = Form(default=None),
    return_base64: bool = Form(default=False)
):
    """
    Transform facial emotion in image
    
    Parameters:
    - image: Image file (PNG, JPG)
    - emotion: Emotion preset ID (happy, sad, surprised, angry, fearful) or "custom"
    - steps: Number of inference steps (20-100, default 30)
    - au_params: Custom AU parameters as JSON string (if emotion="custom")
    - return_base64: Return image as base64 string instead of binary
    
    Returns:
    - Transformed image (binary or base64)
    """
    
    start_time = datetime.utcnow()
    logger.info(f"Transform request: emotion={emotion}, steps={steps}")
    
    try:
        # Validate inputs
        if steps < 20 or steps > 100:
            raise HTTPException(status_code=400, detail="Steps must be between 20 and 100")
        
        # Get AU parameters
        if emotion == "custom":
            if not au_params:
                raise HTTPException(status_code=400, detail="AU parameters required for custom emotion")
            try:
                au_dict = json.loads(au_params)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid AU parameters JSON")
        else:
            if emotion not in EMOTION_PRESETS:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid emotion. Choose from: {list(EMOTION_PRESETS.keys())}"
                )
            au_dict = EMOTION_PRESETS[emotion]
        
        # Read uploaded image
        image_bytes = await image.read()
        
        # Validate image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
            logger.info(f"Image validated: {img.format}, {img.size}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Prepare request to HuggingFace Space
        files = {
            'image': ('image.png', image_bytes, 'image/png')
        }
        
        data = {
            'au_params': json.dumps(au_dict),
            'steps': steps
        }
        
        logger.info(f"Calling HF Space: {HF_API_URL}")
        logger.info(f"AU params: {au_dict}")
        
        # Call HuggingFace Space API
        async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min timeout
            response = await client.post(
                HF_API_URL,
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                logger.error(f"HF Space error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"HuggingFace Space error: {response.text}"
                )
            
            result_image_bytes = response.content
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Transformation complete in {processing_time:.2f}s")
        
        # Return response
        if return_base64:
            # Return as JSON with base64 image
            base64_image = base64.b64encode(result_image_bytes).decode('utf-8')
            return JSONResponse({
                "success": True,
                "image": base64_image,
                "emotion": emotion,
                "au_params": au_dict,
                "steps": steps,
                "processing_time": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            # Return raw image
            return StreamingResponse(
                io.BytesIO(result_image_bytes),
                media_type="image/png",
                headers={
                    "X-Processing-Time": str(processing_time),
                    "X-Emotion": emotion,
                    "X-Steps": str(steps)
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transform error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/batch-transform")
async def batch_transform(
    image: UploadFile = File(...),
    emotions: str = Form(default="happy,sad,surprised"),
    steps: int = Form(default=30)
):
    """
    Transform image to multiple emotions at once
    
    Parameters:
    - image: Image file
    - emotions: Comma-separated emotion IDs (e.g., "happy,sad,surprised")
    - steps: Number of inference steps
    
    Returns:
    - JSON with base64 images for each emotion
    """
    
    start_time = datetime.utcnow()
    emotion_list = [e.strip() for e in emotions.split(",")]
    
    logger.info(f"Batch transform request: {emotion_list}, steps={steps}")
    
    try:
        # Validate emotions
        for emo in emotion_list:
            if emo not in EMOTION_PRESETS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid emotion '{emo}'. Choose from: {list(EMOTION_PRESETS.keys())}"
                )
        
        image_bytes = await image.read()
        
        results = {}
        
        # Process each emotion
        for emo in emotion_list:
            try:
                files = {
                    'image': ('image.png', image_bytes, 'image/png')
                }
                
                data = {
                    'au_params': json.dumps(EMOTION_PRESETS[emo]),
                    'steps': steps
                }
                
                async with httpx.AsyncClient(timeout=600.0) as client:
                    response = await client.post(HF_API_URL, files=files, data=data)
                    
                    if response.status_code == 200:
                        result_image_bytes = response.content
                        base64_image = base64.b64encode(result_image_bytes).decode('utf-8')
                        results[emo] = {
                            "success": True,
                            "image": base64_image,
                            "au_params": EMOTION_PRESETS[emo]
                        }
                    else:
                        results[emo] = {
                            "success": False,
                            "error": response.text
                        }
            
            except Exception as e:
                results[emo] = {
                    "success": False,
                    "error": str(e)
                }
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "success": True,
            "results": results,
            "emotions_processed": len(emotion_list),
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch transform error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
