"""Test with criminal procedure questions"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api.chat import load_resources, retrieve_relevant_chunks, generate_answer

load_resources()

questions = [
    "What is the criminal division overview about?",
    "What procedures are covered in this manual?",
    "How should criminal cases be processed?"
]

for q in questions:
    print(f"\nQ: {q}")
    chunks = retrieve_relevant_chunks(q, top_k=4)
    answer = generate_answer(q, chunks)
    print(f"A: {answer}\n" + "="*60)
