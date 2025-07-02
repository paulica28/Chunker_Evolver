"""
Evaluator for PDF chunker evolution
"""
import os
import time
import importlib.util
from typing import Dict, Any, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def evaluate(program_path: str) -> Dict[str, float]:
    """
    Evaluate the PDF chunker on multiple criteria
    
    Args:
        program_path: Path to the program file
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        if not hasattr(program, 'chunk_pdf'):
            return {
                'chunking_score': 0.0,
                'semantic_coherence': 0.0,
                'boundary_quality': 0.0,
                'metadata_quality': 0.0,
                'performance_score': 0.0,
                'combined_score': 0.0,
                'error': 'Missing chunk_pdf function'
            }
        
        # Test PDFs - use only PDFs provided by the user
        test_pdfs = get_user_pdfs()
        
        total_chunks = 0
        total_time = 0
        all_chunks = []
        semantic_scores = []
        boundary_scores = []
        metadata_scores = []
        
        for pdf_path in test_pdfs:
            if not os.path.exists(pdf_path):
                continue
                
            start_time = time.time()
            chunks = program.chunk_pdf(pdf_path)
            end_time = time.time()
            
            total_chunks += len(chunks)
            total_time += (end_time - start_time)
            all_chunks.extend(chunks)
            
            if chunks:
                # Evaluate semantic coherence
                semantic_score = evaluate_semantic_coherence(chunks)
                semantic_scores.append(semantic_score)
                
                # Evaluate boundary quality
                boundary_score = evaluate_boundary_quality(chunks)
                boundary_scores.append(boundary_score)
                
                # Evaluate metadata quality
                metadata_score = evaluate_metadata_quality(chunks)
                metadata_scores.append(metadata_score)
        
        # Calculate metrics
        chunking_score = min(1.0, total_chunks / 50)  # Normalize chunk count
        semantic_coherence = np.mean(semantic_scores) if semantic_scores else 0.0
        boundary_quality = np.mean(boundary_scores) if boundary_scores else 0.0
        metadata_quality = np.mean(metadata_scores) if metadata_scores else 0.0
        performance_score = 1.0 / (1.0 + total_time)  # Faster is better
        
        # Combined score (weighted average)
        combined_score = (
            0.25 * chunking_score +
            0.30 * semantic_coherence +
            0.25 * boundary_quality +
            0.10 * metadata_quality +
            0.10 * performance_score
        )
        
        return {
            'chunking_score': float(chunking_score),
            'semantic_coherence': float(semantic_coherence),
            'boundary_quality': float(boundary_quality),
            'metadata_quality': float(metadata_quality),
            'performance_score': float(performance_score),
            'combined_score': float(combined_score),
            'total_chunks': total_chunks,
            'total_time': total_time
        }
        
    except Exception as e:
        return {
            'chunking_score': 0.0,
            'semantic_coherence': 0.0,
            'boundary_quality': 0.0,
            'metadata_quality': 0.0,
            'performance_score': 0.0,
            'combined_score': 0.0,
            'error': str(e)
        }

def get_user_pdfs():
    """Get PDFs provided by the user in test_pdfs directory"""
    test_pdfs = []
    
    # Check if test_pdfs directory exists
    if not os.path.exists("test_pdfs"):
        print("Warning: test_pdfs directory not found. Please add your PDF files to test_pdfs/")
        return test_pdfs
    
    # Find all PDF files in test_pdfs directory
    for file in os.listdir("test_pdfs"):
        if file.lower().endswith('.pdf'):
            pdf_path = os.path.join("test_pdfs", file)
            test_pdfs.append(pdf_path)
            print(f"Found PDF: {pdf_path}")
    
    if not test_pdfs:
        print("Warning: No PDF files found in test_pdfs directory. Please add your PDF files.")
    
    return test_pdfs

def evaluate_semantic_coherence(chunks: List[Dict[str, Any]]) -> float:
    """Evaluate how semantically coherent each chunk is"""
    if not chunks:
        return 0.0
    
    # Extract text from chunks
    texts = [chunk['text'] for chunk in chunks]
    
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate average cosine similarity between adjacent chunks
        similarities = []
        for i in range(len(tfidf_matrix) - 1):
            sim = cosine_similarity(
                tfidf_matrix[i:i+1], 
                tfidf_matrix[i+1:i+2]
            )[0][0]
            similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    except:
        return 0.0

def evaluate_boundary_quality(chunks: List[Dict[str, Any]]) -> float:
    """Evaluate how well chunks break at natural boundaries"""
    if not chunks:
        return 0.0
    
    boundary_scores = []
    
    for chunk in chunks:
        text = chunk['text']
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 1:
            boundary_scores.append(0.5)  # Neutral score for single sentence
            continue
        
        # Check if chunk ends at sentence boundary
        last_sentence = sentences[-1]
        if text.rstrip().endswith(last_sentence.rstrip()):
            boundary_scores.append(1.0)
        else:
            boundary_scores.append(0.0)
    
    return float(np.mean(boundary_scores)) if boundary_scores else 0.0

def evaluate_metadata_quality(chunks: List[Dict[str, Any]]) -> float:
    """Evaluate the quality and completeness of metadata"""
    if not chunks:
        return 0.0
    
    metadata_scores = []
    
    for chunk in chunks:
        score = 0.0
        metadata = chunk.get('metadata', {})
        
        # Check for required metadata fields
        if 'source' in metadata:
            score += 0.3
        if 'chunk_size' in metadata:
            score += 0.2
        if 'page' in chunk:
            score += 0.3
        if 'start_char' in chunk and 'end_char' in chunk:
            score += 0.2
            
        metadata_scores.append(score)
    
    return float(np.mean(metadata_scores)) if metadata_scores else 0.0 