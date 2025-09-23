from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from query.views import create_history_aware_chatbot_from_retriever, load_documents

@csrf_exempt
def upload_file_view(request):
    if request.method == 'POST':
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            file_name = uploaded_file.name
            loaded_documents = load_documents(uploaded_file=uploaded_file)
            if loaded_documents and not isinstance(loaded_documents, str):
                # Initialization moved to query_view
                request.session['just_uploaded'] = True # Flag to indicate recent upload
                return JsonResponse({"message": f"File '{file_name}' uploaded successfully!", "filename": file_name})
            else:
                return JsonResponse({"error": f"Error loading document: {loaded_documents}" if isinstance(loaded_documents, str) else "Error loading document."})
        else:
            return JsonResponse({"error": "No file was uploaded."}, status=400)
    else:
        return JsonResponse({"error": "Invalid request method."}, status=400)
