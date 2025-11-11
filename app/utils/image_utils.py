from PIL import Image
import base64
from io import BytesIO

def process_image(image: Image.Image, target_size: int = 512) -> Image.Image:
    """
    Preprocess image for MagicFace model
    
    Args:
        image: Input PIL Image
        target_size: Target dimension (default 512x512)
        
    Returns:
        Processed PIL Image (512x512)
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 512x512 (required by MagicFace)
    if image.size != (target_size, target_size):
        # Maintain aspect ratio and crop
        width, height = image.size
        if width > height:
            new_width = int(target_size * width / height)
            new_height = target_size
        else:
            new_height = int(target_size * height / width)
            new_width = target_size
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop to square
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        image = image.crop((left, top, right, bottom))
    
    return image

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert PIL Image to base64 string
    
    Args:
        image: PIL Image
        format: Output format (JPEG, PNG)
        
    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format=format, quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64

def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        PIL Image
    """
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))
