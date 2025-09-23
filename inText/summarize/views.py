import os
import tempfile
import re
import validators
from dotenv import load_dotenv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from io import BytesIO
from youtube_handler.transcript_fetcher import fetch_youtube_transcript, extract_video_id 

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def load_pdf(pdf_source):
    try:
        loader = PyPDFLoader(pdf_source)
        docs = loader.load_and_split()
        return docs
    except Exception as e:
        return f"Error loading PDF: {e}"

def load_documents(source=None, uploaded_file=None, preferred_language='en'): # Added preferred_language parameter
    docs = list()
    if uploaded_file:
        print("Attempting to load from uploaded file")
        if uploaded_file.name.endswith(".pdf"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    for chunk in uploaded_file.chunks():
                        tmp_file.write(chunk)
                    temp_file_path = tmp_file.name
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load_and_split()
                os.unlink(temp_file_path)
                print(f"Loaded {len(docs)} documents from uploaded PDF")
            except Exception as e:
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                error_message = f"Error processing uploaded PDF: {e}"
                print(error_message)
                return error_message
        else:
            try:
                content = uploaded_file.read().decode("utf-8")
                docs = [Document(page_content=content)]
                print("Loaded content from uploaded text-based file")
            except UnicodeDecodeError:
                error_message = "Error decoding the uploaded file. Please ensure it's a valid text-based file."
                print(error_message)
                return error_message
    elif source:
        print(f"Attempting to load source: {source}")
        video_id = extract_video_id(source)
        print(f"Extracted video ID: '{video_id}'")
        if video_id:
            print("Detected YouTube URL")
            transcript_text = fetch_youtube_transcript(video_id, preferred_language=preferred_language) # Use the new function
            print(f"Transcript: {transcript_text[:100]}...") # Print first 100 chars
            if "Error" in transcript_text or "Transcript unavailable" in transcript_text:
                print(transcript_text)
                return transcript_text
            docs = [Document(page_content=transcript_text)]
            print(f"Loaded transcript from YouTube video ID: {video_id}")
        elif source.endswith(".pdf"):
            print("Detected PDF URL")
            try:
                loader = PyPDFLoader(source)
                loaded_docs = loader.load_and_split()
                docs.extend(loaded_docs)
                print(f"Loaded {len(loaded_docs)} documents from PDF URL")
            except Exception as e:
                error_message = f"Error loading PDF from URL: {e}"
                print(error_message)
                return error_message
        elif source.startswith("http://") or source.startswith("https://"): # Simpler URL check
            print("Detected generic URL")
            try:
                loader = UnstructuredURLLoader(urls=[source])
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
                print(f"Loaded {len(loaded_docs)} documents from URL")
            except Exception as e:
                error_message = f"Error loading URL: {e}"
                print(error_message)
                return error_message
        else:
            print("Source is not a valid URL or file extension not recognized for direct loading.")
            return "Source is not a valid URL or file extension not recognized."
    else:
        print("No source or uploaded file provided.")
        return "No source or uploaded file provided."
    return docs

def summarize(docs):
    if not docs:
        return "No valid content found to summarize."
    chunk_prompt = PromptTemplate(template="Summarize the below text:\n{text}\nSummary:", input_variables=["text"])
    final_prompt = PromptTemplate(template="Summarize the following text clearly and concisely. Please present the summary in multiple points and paragraphs to improve readability:\n{text}", input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=chunk_prompt, combine_prompt=final_prompt)
    return chain.invoke(docs)

@csrf_exempt
def summarize_view(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        source = request.POST.get('source', '').strip()
        preferred_language = request.POST.get('preferred_language', 'en').strip() # Get preferred language from request
        print(f"Source being passed to summarize_view: '{source}', Preferred Language: '{preferred_language}'")

        # Accessing session to ensure session cookie is set
        request.session['summarize_called'] = True

        if uploaded_file:
            docs = load_documents(uploaded_file=uploaded_file)
        elif source:
            docs = load_documents(source=source, preferred_language=preferred_language) # Pass preferred language
        else:
            return JsonResponse({"error": "No source or file provided."}, status=400)

        if isinstance(docs, str):
            return JsonResponse({"error": docs}, status=400)

        summary_output = summarize(docs)
        if isinstance(summary_output, dict) and 'output_text' in summary_output:
            summary_text = summary_output['output_text']
            return JsonResponse({"summary": summary_text})
        else:
            return JsonResponse({"error": "Failed to extract summary."}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
