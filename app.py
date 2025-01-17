import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Initialize FastAPI app
app = FastAPI(title="CUTM Chatbot API")

# Initialize global vector store
vector_store = None

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    context: Optional[List[str]] = None

def read_text_file(file_path):
    """Try different encodings to read the text file."""
    encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    
    raise RuntimeError(f"Could not read file with any of the following encodings: {encodings}")

def initialize_vector_store():
    """Initialize or load the vector store."""
    try:
        # Check if saved index exists
        if os.path.exists("faiss_index"):
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = FAISS.load_local(
                "faiss_index", 
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("Loaded existing vector store from disk")
            return vector_store
            
        # If no saved index, create new one
        text_content = read_text_file("data.txt")
        doc = Document(page_content=text_content, metadata={"source": "data.txt"})
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents([doc])
        
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Save the index to disk
        vector_store.save_local("faiss_index")
        print("Created and saved new vector store to disk")
        
        return vector_store
    
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize vector store")

# Initialize the chat chain
def get_chat_chain(vector_store):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name='mixtral-8x7b-32768'
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant with expertise in Centurion University of Technology and Management (CUTM). Your knowledge comes from the university's official website data. Please:
    
    1. Provide accurate, factual information from CUTM's website data
    2. Say "I don't have that information in my database" if you cannot find a reliable answer
    3. Keep responses clear and professional
    4. Stay focused on answering the specific question asked
    5. Use natural, conversational language while maintaining professionalism"
    
    <context>
    {context}
    </context>
    
    Question: {input}""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

@app.get("/")
async def root():
    """Root endpoint that confirms the API is running."""
    return {"message": "CUTM RAG Chatbot Backend is up and running"}

@app.on_event("startup")
async def startup_event():
    """Initialize vector store on startup."""
    global vector_store
    vector_store = initialize_vector_store()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that accepts a question and returns an answer with optional context.
    """
    global vector_store
    
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    try:
        chat_chain = get_chat_chain(vector_store)
        response = chat_chain.invoke({"input": request.question})
        
        # Extract context content for response
        context_content = [doc.page_content for doc in response["context"]]
        
        return ChatResponse(
            answer=response["answer"],
            context=context_content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/rebuild-index")
async def rebuild_index():
    """
    Endpoint to rebuild the vector store index.
    """
    global vector_store
    try:
        # Delete existing index if it exists
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")
        
        # Reinitialize vector store
        vector_store = initialize_vector_store()
        return {"message": "Vector store index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
