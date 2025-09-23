import cv2
import pytesseract
import numpy as np
from PIL import Image
import google.generativeai as genai
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.environ.get("GOOGLE_API_KEY")

# Configure Google Gemini API
try:
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash') # Use appropriate model
    else:
        gemini_model = None
        print("Warning: GEMINI_API_KEY not found in environment variables. Gemini enhancement disabled.")
except Exception as e:
    gemini_model = None
    print(f"Warning: Failed to configure Google Gemini API: {e}. Enhancement disabled.")


def preprocess_image(image_path):
    """Applies standard preprocessing: grayscale, thresholding."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image file: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        return thresh
    except Exception as e:
        print(f"Error during preprocessing image {image_path}: {e}")
        raise


def perform_ocr(image_data):
    """Performs OCR using PyTesseract on a preprocessed image (NumPy array or PIL Image)."""
    try:
        if isinstance(image_data, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        elif isinstance(image_data, Image.Image):
             pil_image = image_data
        else:
             pil_image = image_data

        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        print(f"Error: Tesseract executable not found. Please ensure it's in your PATH or configure pytesseract.")
        raise
    except Exception as e:
        print(f"Error during OCR: {e}")
        raise


def enhance_with_gemini(image_path, ocr_text):
    """Uses Gemini to analyze the image and OCR text for corrections and structure."""
    if not gemini_model:
        print("Gemini enhancement skipped: Model not configured.")
        return "Gemini enhancement disabled."

    try:
        print(f"Starting Gemini enhancement for {image_path}")
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        prompt = f"""
Analyze the provided image and the following OCR text extracted from it.
Your goal is to improve the accuracy and structure of the text based on the visual layout in the image.

**Instructions:**
1.  **Identify and Correct Errors:** Correct any misrecognized characters, words, or formatting issues in the OCR text by comparing it with the image content.
2.  **Preserve Layout:** Maintain the original structure (paragraphs, lists, tables if any) as seen in the image. Use Markdown for formatting lists and simple tables if appropriate.
3.  **Contextual Understanding:** Use the visual context to resolve ambiguities in the text.
4.  **Output ONLY the corrected and formatted text.** Do not include any explanations, apologies, or introductory phrases like "Here is the corrected text:".

**OCR Text:**
{ocr_text}

**Corrected and Formatted Text:**
"""
        response = gemini_model.generate_content([prompt, img], stream=False)
        response.resolve()

        print(f"Gemini response received for {image_path}")
        if response.parts:
             enhanced_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
             return enhanced_text.strip()
        elif hasattr(response, 'text'):
             return response.text.strip()
        else:
             print(f"Warning: Gemini response for {image_path} did not contain expected text format.")
             if response.prompt_feedback.block_reason:
                 print(f"Gemini request blocked: {response.prompt_feedback.block_reason}")
                 return f"Gemini processing failed: Blocked ({response.prompt_feedback.block_reason})"
             return "Gemini enhancement failed: No text found in response."

    except Exception as e:
        print(f"Error during Gemini enhancement for {image_path}: {e}")
        return f"Gemini enhancement failed: {str(e)}"

def merge_results(tesseract_text, gemini_text):
    """Basic strategy to merge results. Prefers Gemini if available and seems valid."""
    if gemini_text and not gemini_text.startswith("Gemini enhancement failed") and not gemini_text.startswith("Gemini enhancement disabled"):
        return gemini_text
    return tesseract_text