"""Quick test to check if PDF can be read"""
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

pdf_path = "criminaloverview.pdf"

try:
    print(f"Opening PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    print(f"\u2705 PDF opened successfully")
    print(f"   Total pages: {len(reader.pages)}")
    
    # Try reading first page
    print("\n--- First page preview ---")
    first_page_text = reader.pages[0].extract_text()
    print(first_page_text[:500])
    print(f"\n\u2705 Successfully extracted {len(first_page_text)} characters from first page")
    
except Exception as e:
    print(f"\u274c Error: {e}")
