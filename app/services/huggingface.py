import aiohttp
import base64
from io import BytesIO
from PIL import Image
from typing import Dict
import os
import json

class HuggingFaceService:
    def __init__(self):
        # Your HuggingFace Space URL
        self.api_url = os.getenv(
            "HF_SPACE_URL", 
            "https://gauravvjhaaok-affecto-inference.hf.space"
        )
        print(f"🔗 HuggingFace Space URL: {self.api_url}")
        
    async def transform_emotion(
        self, 
        image: Image.Image, 
        au_params: Dict[str, float]
    ) -> Image.Image:
        """
        Transform emotion using HuggingFace Space API (Gradio)
        
        Args:
            image: PIL Image
            au_params: Dictionary of Action Unit parameters
            
        Returns:
            Transformed PIL Image
        """
        try:
            print(f"📤 Sending request to HF Space...")
            print(f"🎭 AU Params: {au_params}")
            
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=95)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Prepare Gradio API payload
            # Format: {"data": [image, au_params_json, steps, seed]}
            payload = {
                "data": [
                    f"data:image/jpeg;base64,{img_base64}",  # Image as data URL
                    json.dumps(au_params),  # AU params as JSON string
                    50,  # inference steps
                    424  # seed
                ]
            }
            
            # Call HuggingFace Space Gradio API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/predict",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180)  # 3 min timeout
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"❌ HF API error {response.status}: {error_text}")
                        raise Exception(f"HF API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    print(f"📥 Received response from HF Space")
                    
                    # Gradio returns: {"data": [output_image]}
                    if "data" not in result or len(result["data"]) == 0:
                        raise Exception("Invalid response from HF Space")
                    
                    output_data = result["data"][0]
                    
                    # Handle different Gradio output formats
                    if isinstance(output_data, dict):
                        # File path format: {"name": "path", "data": null, "is_file": true}
                        if "name" in output_data:
                            file_path = output_data["name"]
                            # Download the file
                            file_url = f"{self.api_url}/file={file_path}"
                            print(f"📥 Downloading from: {file_url}")
                            async with session.get(file_url) as img_response:
                                if img_response.status == 200:
                                    image_bytes = await img_response.read()
                                    transformed_image = Image.open(BytesIO(image_bytes))
                                else:
                                    raise Exception(f"Failed to download image: {img_response.status}")
                        else:
                            raise Exception(f"Unexpected dict format: {output_data}")
                    
                    elif isinstance(output_data, str):
                        # Base64 or data URL format
                        if output_data.startswith("data:image"):
                            # Data URL format
                            base64_data = output_data.split(",")[1]
                            image_bytes = base64.b64decode(base64_data)
                        else:
                            # Plain base64
                            image_bytes = base64.b64decode(output_data)
                        
                        transformed_image = Image.open(BytesIO(image_bytes))
                    
                    else:
                        raise Exception(f"Unexpected output format: {type(output_data)}")
                    
                    print("✅ Transformation successful!")
                    return transformed_image
                    
        except aiohttp.ClientError as e:
            print(f"❌ Network error: {str(e)}")
            raise Exception(f"Failed to connect to inference service: {str(e)}")
        except Exception as e:
            print(f"❌ Transformation error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Transformation failed: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check if HuggingFace Space is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception as e:
            print(f"❌ Health check failed: {str(e)}")
            return False
