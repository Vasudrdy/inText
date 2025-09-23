import pytesseract
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
from io import BytesIO
import os
from django.utils import timezone
import urllib.parse

from .service import preprocess_image, perform_ocr, enhance_with_gemini, merge_results

@csrf_exempt
def extract_text_from_image(request):
    if request.method == 'POST':
        image_url = request.POST.get('image_url')
        uploaded_image = request.FILES.get('image_file')

        if not image_url and not uploaded_image:
            return JsonResponse({'error': 'No image provided'}, status=400)

        try:
            if image_url:
                print(f"Attempting to fetch image from URL: {image_url}") # Keeping this log for URL fetch attempt
                response = requests.get(image_url)
                response.raise_for_status()
                image_file = BytesIO(response.content)
                parsed_url = urllib.parse.urlparse(image_url)
                filename = os.path.basename(parsed_url.path)
                if not filename:
                    filename = "web_image_" + str(timezone.now().timestamp())
            elif uploaded_image:
                print(f"Processing uploaded image: {uploaded_image.name}") # Keeping this log for uploaded image processing
                image_file = uploaded_image
                filename = uploaded_image.name
            else:
                return JsonResponse({'error': 'No image data'}, status=400)

            # Create a temporary filename to store the image
            temp_filename = f"temp_{timezone.now().timestamp()}_{filename}"
            # Sanitize filename to be used as a path
            image_path = temp_filename.replace('.', '_')

            # Save the image to a temporary file
            with open(image_path, 'wb') as f:
                if isinstance(image_file, BytesIO):
                    f.write(image_file.getvalue())
                else:
                    for chunk in image_file.chunks():
                        f.write(chunk)

            print(f"Processing image: {filename} at path {image_path}") # Keeping this log for image processing start

            # --- OCR Processing Pipeline ---
            preprocessed_data = preprocess_image(image_path)
            tesseract_text = perform_ocr(preprocessed_data)
            gemini_text = enhance_with_gemini(image_path, tesseract_text)
            final_text = merge_results(tesseract_text, gemini_text)
            # --- End of OCR Processing Pipeline ---

            results = [{
                'id': None, # No model ID
                'status': 'COMPLETED',
                'imageUrl': image_url if image_url else (request.build_absolute_uri(uploaded_image.url) if uploaded_image and hasattr(uploaded_image, 'url') else None),
                'filename': filename,
                'finalText': final_text,
                'error': None
            }]
            return JsonResponse({'results': results})

        except requests.exceptions.RequestException as e:
            error_message = f'Error fetching image from URL: {e}'
            print(error_message)
            return JsonResponse({'error': error_message}, status=500)
        except FileNotFoundError:
            error_message = f'Error: Temporary image file not found.'
            print(error_message)
            return JsonResponse({'error': error_message}, status=500)
        except pytesseract.TesseractNotFoundError:
            error_message = "Error: Tesseract is not installed or not in your PATH."
            print(error_message)
            return JsonResponse({'error': error_message}, status=500)
        except Exception as e:
            error_message = f'Error processing image: {e}'
            print(error_message)
            return JsonResponse({'error': error_message}, status=500)
        finally:
            # Clean up the temporary image file
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)