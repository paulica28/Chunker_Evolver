# EVOLVE-BLOCK-START
"""
Basic PDF chunker for RAG applications
This is the starting point for evolution - a simple but functional chunker
"""
import re
from typing import List, Dict, Any
import fitz  # PyMuPDF

def chunk_pdf(pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Basic PDF chunker that splits text into fixed-size chunks.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunks with metadata
    """
    try:
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Simple text splitting
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                
                chunks.append({
                    'text': chunk_text,
                    'page': page_num + 1,
                    'start_char': start,
                    'end_char': end,
                    'metadata': {
                        'source': pdf_path,
                        'chunk_size': len(chunk_text)
                    }
                })
                
                start = end - overlap
                
        doc.close()
        return chunks
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

# EVOLVE-BLOCK-END

# Test function (not evolved)
def test_chunker():
    """Test the chunker on a sample PDF"""
    chunks = chunk_pdf("test_pdfs/sample.pdf")
    return len(chunks), chunks[:3] if chunks else []

if __name__ == "__main__":
    count, sample = test_chunker()
    print(f"Generated {count} chunks")
    print("Sample chunks:", sample) 