# query/views.py (Cleaned up)
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from summarize.views import load_documents, text_splitter
from dotenv import load_dotenv
import json
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from django.shortcuts import get_object_or_404
from django.contrib.sessions.backends.db import SessionStore
from django.utils.text import slugify
import os.path

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# --- Configuration Constants ---
CHROMA_PERSIST_DIR = "./chroma_dbs" # Directory to store persistent Chroma collections
MAX_CHAT_HISTORY_TURNS = 10 # Number of turns to keep in history (user + AI = 1 turn)

# Ensure the persistence directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# --- Initialize Models (Global or per process/thread depending on deployment) ---
# Initialize these once, outside of the request function
try:
    llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
except Exception:
    llm = None
    embeddings = None


# --- Prompt Definitions ---
# Contextualize Query Prompt
contextualize_q_system_prompt = (
    "Given the conversation history and a follow-up question, "
    "rephrase the follow-up question into a concise, standalone question "
    "that includes all necessary context from the history. "
    "The rephrased question should be understandable on its own. "
    "If no rephrasing is needed, return the original question as is. "
    "Your output should be ONLY the standalone question."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt +
         "\n\nExample 1:\nChat history:\nHuman: What is the capital of France?\nAI: Paris\nHuman: What is the population of that city?\nStandalone question: What is the population of Paris?\n\nExample 2:\nChat history:\nHuman: Tell me about quantum computing.\nAI: Quantum computing is...\nHow does it differ from classical computing?\nStandalone question: How does quantum computing differ from classical computing?\n\nExample 3:\nChat history:\nHuman: Who was the first person on the moon?\nAI: Neil Armstrong.\nHuman: Tell me about him.\nStandalone question: Tell me about Neil Armstrong?\n\nExample 4:\nChat history:\nHuman: What is the boiling point of water?\nAI: 100 degrees Celsius.\nHuman: What about freezing point?\nStandalone question: What is the freezing point of water?\n\nExample 5:\nChat history:\nHuman: What is the highest mountain?\nAI: Mount Everest.\nHuman: Okay, thanks!\nStandalone question: Okay, thanks!\n\nExample 6:\nChat history:\n[... previous turns ...]\nHuman: What is the speed of light?\nStandalone question: What is the speed of light?"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# System Prompt for Generation
system_prompt = (
    "You are a helpful assistant whose goal is to answer questions accurately "
    "based ONLY on the specific text provided in the context below. "
    "Synthesize the relevant information from the context to form your answer. "
    "Answer as if the information is part of your own knowledge; "
    "DO NOT mention that you are using the provided context or refer to the context, document, or text in your answer. "
    "Keep the answer concise, typically within one to three sentences. "
    "If the provided context does NOT contain sufficient information to answer the question fully, "
    "state clearly that you are unable to answer based on the information provided. "
    "Do NOT use any external knowledge."
    "\n\n"
    "{context}"
    "\n\n"
    # Examples to match new instructions (no source references, new I don't know phrasing)
    "Example 1:\nContext: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887–1889 as the entrance to the 1889 World's Fair. It is named after the engineer Gustave Eiffel.\nQuestion: Where is the Eiffel Tower located?\nAnswer: The Eiffel Tower is located on the Champ de Mars in Paris, France.\n\nExample 2:\nContext: The capital of Italy is Rome. The Colosseum is in Rome.\nQuestion: What is the population of Rome?\nAnswer: I am unable to answer based on the information provided.\n\nExample 3:\nContext: The fastest land animal is the cheetah, capable of speeds up to 70 mph. The largest animal is the blue whale.\nQuestion: What is the fastest marine animal?\nAnswer: I am unable to answer based on the information provided.\n\nExample 4:\nContext: [Relevant context about a topic]\nQuestion: [A question answerable by the context]\nAnswer: [Concise answer based only on the context, no source reference]\n\nExample 5:\nContext: [Irrelevant context about a different topic]\nQuestion: [A question]\nAnswer: I am unable to answer based on the information provided."
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"), # This {input} is the (potentially contextualized) question
    ]
)

# --- Helper Function to get or create Vectorstore ---
def get_or_create_vectorstore(source_identifier, loaded_documents, embeddings, text_splitter):
    """
    Loads an existing Chroma vectorstore for the given identifier or
    creates a new one if it doesn't exist, then persists it.
    """
    if not source_identifier:
         raise ValueError("source_identifier cannot be empty.")

    # Create a safe directory name from the identifier
    safe_dir_name = slugify(source_identifier)
    persist_directory = os.path.join(CHROMA_PERSIST_DIR, safe_dir_name)

    # Check if the vectorstore already exists
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        try:
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vectorstore
        except Exception as e:
            raise e # Re-raise the error to be handled by the caller

    else:
        if not loaded_documents:
            return None # Cannot create if no documents

        splits = text_splitter.split_documents(loaded_documents)
        if not splits:
             return None

        try:
            # Ensure the directory exists before persisting
            os.makedirs(persist_directory, exist_ok=True)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
            return vectorstore
        except Exception as e:
            raise e # Re-raise the error


# --- Chatbot Creation Function (Takes retriever) ---
def create_history_aware_chatbot_from_retriever(retriever, llm):
    """Creates a history-aware chatbot from a provided retriever."""
    if retriever is None:
        return None

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Use the previously defined qa_prompt for the final answering step
    answer_generation_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combine the history-aware retriever and the answer generation chain
    rag_chain = create_retrieval_chain(history_aware_retriever, answer_generation_chain)

    # This part wraps the rag_chain to manage chat history
    def get_session_history(session_id):
        # This function fetches history from Django session
        if session_id:
            try:
                session = SessionStore(session_key=session_id)
                # Default to empty list if 'chat_history' is not in session
                history = session.get('chat_history', [])
                llm_history = ChatMessageHistory()
                for message in history:
                    if message['type'] == 'human':
                        llm_history.add_user_message(message['content'])
                    elif message['type'] == 'ai':
                        llm_history.add_ai_message(message['content'])
                return llm_history
            except Exception:
                # Return empty history on error to allow query to proceed without history
                return ChatMessageHistory()
        else:
            return ChatMessageHistory()

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history, # Pass the local function
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


# --- Chat History Management (Used by query_view) ---
def manage_chat_history(request, user_query, ai_response, max_history=MAX_CHAT_HISTORY_TURNS):
    chat_history_data = request.session.get('chat_history', [])
    chat_history_data.append({"type": "human", "content": user_query})
    chat_history_data.append({"type": "ai", "content": ai_response})

    # Limit the history to the last max_history turns
    # Each turn is one human + one AI message
    if len(chat_history_data) > max_history * 2:
        chat_history_data = chat_history_data[len(chat_history_data) - (max_history * 2):]

    request.session['chat_history'] = chat_history_data
    request.session.modified = True # Ensure session is saved
    return request.session.get('chat_history', [])


# --- Django View ---
@csrf_exempt
def query_view(request):
    # Ensure LLM and embeddings are initialized
    if llm is None or embeddings is None:
         return JsonResponse({"error": "Server error: Language model not available."}, status=500)

    # Force session creation if it doesn't exist (important for session_id)
    if request.session.session_key is None:
        request.session.save() # Explicitly save to get a session_key

    if request.method == 'POST':
        query = request.POST.get('query', '').strip()

        if not query:
            return JsonResponse({"error": "Query is required."}, status=400)

        uploaded_file = request.FILES.get('file')
        source_url = request.POST.get('source', '').strip() # Renamed to source_url for clarity

        vectorstore = None
        source_identifier = None # Unique identifier for the vectorstore

        # --- Determine source_identifier and load/create vectorstore ---
        if uploaded_file:
            # Use filename as identifier (assuming unique filenames per user/context)
            source_identifier = uploaded_file.name
            try:
                # Load documents from file *first* if creating a new store
                loaded_documents = load_documents(uploaded_file=uploaded_file)
                if isinstance(loaded_documents, str): # Check for load error string
                     return JsonResponse({"error": f"Error loading file: {loaded_documents}"}, status=400)
                if not loaded_documents:
                    return JsonResponse({"error": "No text found in uploaded file."}, status=400)

                # Get or create the vectorstore for this file
                vectorstore = get_or_create_vectorstore(source_identifier, loaded_documents, embeddings, text_splitter)
                # Store the identifier in session to remember which vectorstore to use
                request.session['current_vectorstore_id'] = source_identifier
                request.session.modified = True

            except Exception as e:
                return JsonResponse({"error": f"Error processing file: {e}"}, status=500)

        elif source_url:
            # Use source URL as identifier
            source_identifier = source_url
            try:
                 # Load documents from URL *first* if creating a new store
                 loaded_documents = load_documents(source=source_url)
                 if isinstance(loaded_documents, str): # Check for load error string
                      return JsonResponse({"error": f"Error loading source: {loaded_documents}"}, status=400)
                 if not loaded_documents:
                     return JsonResponse({"error": "No text found at source URL."}, status=400)

                 # Get or create the vectorstore for this URL
                 vectorstore = get_or_create_vectorstore(source_identifier, loaded_documents, embeddings, text_splitter)
                 # Store the identifier in session
                 request.session['current_vectorstore_id'] = source_identifier
                 request.session.modified = True

            except Exception as e:
                 return JsonResponse({"error": f"Error processing source: {e}"}, status=500)

        else:
            # No new file or source provided, try to load from session
            source_identifier = request.session.get('current_vectorstore_id')
            if source_identifier:
                try:
                    # Attempt to load the existing vectorstore
                    # Pass None for loaded_documents as we are loading, not creating
                    vectorstore = get_or_create_vectorstore(source_identifier, None, embeddings, text_splitter)
                except Exception:
                    # Clear session ID if loading fails, force user to re-upload/re-specify source
                    del request.session['current_vectorstore_id']
                    request.session.modified = True
                    return JsonResponse({"error": "Could not load previous document context. Please provide the file or source again."}, status=500)
            else:
                # No file, no source, and no vectorstore in session
                return JsonResponse({"error": "Please provide a document (file or source URL) to start the conversation."}, status=400)

        # --- If we reached here, we should have a vectorstore ---
        if vectorstore:
            # Increased k to retrieve more chunks - experiment with this value
            retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
            conversational_rag_chain = create_history_aware_chatbot_from_retriever(retriever, llm)

            try:
                # Pass the session_key to the chain's config
                response = conversational_rag_chain.invoke(
                    {"input": query},
                    config={"configurable": {"session_id": request.session.session_key}}
                )
                answer = response['answer']

                # Manage chat history in the session
                manage_chat_history(request, query, answer, max_history=MAX_CHAT_HISTORY_TURNS)

                return JsonResponse({"answer": answer})

            except Exception as e:
                # Provide more specific error if possible, but generic 500 might be safer
                return JsonResponse({"error": f"An error occurred while processing your query: {e}"}, status=500)
        else:
             # This case should ideally not be reached if get_or_create_vectorstore returns None only on error
             return JsonResponse({"error": "Failed to create or load document index."}, status=500)

    else:
        return JsonResponse({"error": "Invalid request method."}, status=405) # Use 405 for Method Not Allowed
