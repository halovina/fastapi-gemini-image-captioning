import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini Pro Vision model
# Ensure you are using a model that supports image input, e.g., 'gemini-pro-vision'
try:
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
except Exception as e:
    raise RuntimeError(f"Failed to initialize Gemini model. Make sure 'gemini-2.5-flash-preview-04-17' is available and your API key is correct. Error: {e}")

app = FastAPI(
    title="Gemini Image Captioning API",
    description="Upload an image and get a descriptive caption from gemini-2.5-flash-preview-04-17",
    version="1.0.0"
)

# --- Prompt Text ---
IMAGE_CAPTIONING_PROMPT = """
Describe this image in detail. Focus on the main subjects, actions, setting, colors, and any discernible emotions or atmosphere.
Provide a concise yet comprehensive caption, suitable for accessibility purposes or a brief summary.
Make sure to include any notable features or context that would help someone understand the image without seeing it.
Translate to bahasa indonesia if necessary.
"""
# --- End Prompt Text ---

@app.get("/")
async def read_root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Gemini Image Captioning API! Visit /docs for more info."}

@app.post("/caption-image/")
async def caption_image(file: UploadFile = File(...)):
    """
    Uploads an image and requests Gemini Pro Vision to generate a caption for it.

    Args:
        file (UploadFile): The image file to be uploaded.

    Returns:
        JSONResponse: A JSON object containing the generated caption or an error message.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Read the image bytes
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Prepare the content for the Gemini model
        # The model expects a list of parts, which can include text and image data.
        # Image parts require the PIL.Image object.
        contents = [
            IMAGE_CAPTIONING_PROMPT,
            image
        ]

        # Generate content using Gemini
        response = model.generate_content(contents)

        # Access the generated caption
        caption = response.text.strip()
        if not caption:
            caption = "No caption could be generated for this image."

        return JSONResponse(content={"filename": file.filename, "caption": caption})

    except genai.APIError as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")
    except Exception as e:
        # Catch other potential errors during file processing or AI response
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")