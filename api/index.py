"""
Serverless API Handler for Court Manual RAG System
Handles chat requests using FAISS vector search and OpenAI GPT-4o-mini
Compatible with Vercel Python runtime
"""

import json
import os
from openai import OpenAI
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Global variables to cache loaded resources
_index = None
_chunks = None

def load_resources():
    """Load FAISS index and chunks (cached after first load)"""
    global _index, _chunks
    
    if _index is None or _chunks is None:
        # Get paths relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rag_dir = os.path.join(os.path.dirname(script_dir), "rag")
        
        index_path = os.path.join(rag_dir, "manual.index")
        chunks_path = os.path.join(rag_dir, "chunks.json")
        
        # Load FAISS index
        _index = faiss.read_index(index_path)
        
        # Load chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            _chunks = json.load(f)
    
    return _index, _chunks

def create_query_embedding(query_text):
    """Create embedding for user query"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query_text]
    )
    return np.array([response.data[0].embedding], dtype=np.float32)

def retrieve_relevant_chunks(query_text, top_k=4):
    """
    Retrieve the top_k most relevant chunks for the query
    Returns list of relevant text chunks
    """
    index, chunks = load_resources()
    
    # Create query embedding
    query_embedding = create_query_embedding(query_text)
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    # Get corresponding chunks
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    
    return relevant_chunks

def generate_answer(user_question, context_chunks):
    """
    Use GPT-4o-mini to generate an answer based on retrieved context
    Enforces strict adherence to manual content
    """
    # Build context from chunks
    context = "\n\n---\n\n".join(context_chunks)
    
    # System prompt enforcing manual-only answers
    system_prompt = """You are a helpful assistant that answers questions about court clerk procedures.

CRITICAL RULES:
1. Answer ONLY using the information provided in the manual excerpts below
2. If the manual excerpts do not contain the answer, explicitly state: "The court clerk manual does not provide information about this topic."
3. Do not use external knowledge or make assumptions
4. Maintain a professional, court-clerk tone
5. Be precise and cite specific procedures when available
6. If information is partial, acknowledge the limitation

Manual excerpts:
"""
    
    full_prompt = system_prompt + context
    
    # Call GPT-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.3,  # Lower temperature for more deterministic responses
        max_tokens=500
    )
    
    return response.choices[0].message.content

def handler(request):
    """Vercel serverless function handler"""
    try:
        # Parse request body
        if hasattr(request, 'body'):
            data = json.loads(request.body.decode('utf-8'))
        else:
            # For Vercel, request body might be in different format
            data = request.get_json() if request.is_json else {}
        
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({"error": "Missing 'message' field"})
            }
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(user_message, top_k=4)
        
        # Generate answer
        answer = generate_answer(user_message, relevant_chunks)
        
        # Send response
        response_data = {"reply": answer}
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_data)
        }
    
    except Exception as e:
        error_message = f"Server error: {str(e)}"
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": error_message})
        }
