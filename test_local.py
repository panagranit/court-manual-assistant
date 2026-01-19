"""
Local test of the RAG chat system
Tests the same logic that will run on Vercel
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the chat module
from api.chat import load_resources, retrieve_relevant_chunks, generate_answer

def test_query(question):
    """Test a query locally"""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    try:
        # Load resources
        print("Loading FAISS index and chunks...")
        load_resources()
        print("\u2705 Resources loaded\n")
        
        # Retrieve relevant chunks
        print("Retrieving relevant chunks...")
        chunks = retrieve_relevant_chunks(question, top_k=4)
        print(f"\u2705 Found {len(chunks)} relevant chunks\n")
        
        print("Relevant excerpts:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- Chunk {i} ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        
        # Generate answer
        print("\n\nGenerating answer with GPT-4o-mini...")
        answer = generate_answer(question, chunks)
        print("\u2705 Answer generated\n")
        
        print(f"{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\u274c Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
