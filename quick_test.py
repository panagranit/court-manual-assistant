"""Quick single question test"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.chat import load_resources, retrieve_relevant_chunks, generate_answer

print("Testing RAG system with one question...\n")

question = "What are the requirements for filing a new criminal case?"
print(f"Question: {question}\n")

print("1. Loading resources...")
load_resources()
print("   















print("="*60)print(answer)print("="*60)print("ANSWER:")print("="*60)print(f"   
 Done\n")answer = generate_answer(question, chunks)print("3. Generating answer...")print(f"   
 Found {len(chunks)} chunks\n")chunks = retrieve_relevant_chunks(question, top_k=4)print("2. Finding relevant chunks...") Loaded\n")