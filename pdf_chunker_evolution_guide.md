# PDF Chunker Evolution Guide for RAG Applications

This guide explains how to use OpenEvolve to evolve an optimal PDF chunker for RAG (Retrieval-Augmented Generation) applications.

## Understanding OpenEvolve

OpenEvolve is an evolutionary coding agent that uses Large Language Models to optimize code through an iterative process. Here's how it works:

### Core Components

1. **Controller** (`openevolve/controller.py`): Orchestrates the entire evolution process
2. **Database** (`openevolve/database.py`): Stores programs and implements MAP-Elites algorithm for diversity
3. **Evaluator** (`openevolve/evaluator.py`): Tests generated programs and assigns scores
4. **LLM Ensemble** (`openevolve/llm/ensemble.py`): Generates code modifications via multiple language models
5. **Prompt Sampler** (`openevolve/prompt/sampler.py`): Creates context-rich prompts for the LLM

### Evolution Process

1. **Initialization**: Start with a basic PDF chunker implementation
2. **Evaluation**: Test the chunker on sample PDFs and measure performance metrics
3. **Selection**: Choose promising programs from the database using MAP-Elites
4. **Variation**: Use LLM to generate improved versions based on past performance
5. **Iteration**: Repeat until convergence or maximum iterations reached

## PDF Chunker Evolution Setup

### 1. Project Structure

```
pdf_chunker_evolution/
├── initial_program.py      # Starting PDF chunker implementation
├── evaluator.py           # Evaluation logic and metrics
├── config.yaml           # OpenEvolve configuration
├── requirements.txt      # Dependencies
├── test_pdfs/           # Sample PDFs for testing
│   ├── academic_paper.pdf
│   ├── technical_doc.pdf
│   └── mixed_content.pdf
└── README.md
```

### 2. Initial PDF Chunker Program

```python
# initial_program.py
"""
EVOLVE-BLOCK-START
Basic PDF chunker for RAG applications
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
    chunks = chunk_pdf("test_pdfs/academic_paper.pdf")
    return len(chunks), chunks[:3] if chunks else []

if __name__ == "__main__":
    count, sample = test_chunker()
    print(f"Generated {count} chunks")
    print("Sample chunks:", sample)
```

### 3. Evaluation Function

```python
# evaluator.py
"""
Evaluator for PDF chunker evolution
"""
import os
import time
import importlib.util
from typing import Dict, Any
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
        
        # Test PDFs
        test_pdfs = [
            "test_pdfs/academic_paper.pdf",
            "test_pdfs/technical_doc.pdf",
            "test_pdfs/mixed_content.pdf"
        ]
        
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
        chunking_score = min(1.0, total_chunks / 100)  # Normalize chunk count
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
```

### 4. Configuration File

```yaml
# config.yaml
# OpenEvolve configuration for PDF chunker evolution

# General settings
max_iterations: 500
checkpoint_interval: 25
log_level: "INFO"
random_seed: 42

# Evolution settings
diff_based_evolution: true
max_code_length: 8000

# LLM configuration
llm:
  models:
    - name: "gpt-4"
      weight: 0.7
    - name: "gpt-3.5-turbo"
      weight: 0.3

  evaluator_models:
    - name: "gpt-4"
      weight: 1.0

  api_base: "https://api.openai.com/v1/"
  api_key: null  # Set via OPENAI_API_KEY environment variable
  
  temperature: 0.8
  top_p: 0.95
  max_tokens: 4096
  timeout: 60
  retries: 3

# Prompt configuration
prompt:
  system_message: |
    You are an expert in PDF processing and RAG (Retrieval-Augmented Generation) systems.
    Your task is to improve PDF chunking algorithms that split PDF documents into
    meaningful chunks for use in RAG applications. Focus on:
    - Semantic coherence within chunks
    - Natural boundary detection (sentences, paragraphs)
    - Metadata preservation
    - Performance optimization
    - Handling different document types (academic, technical, mixed content)

  evaluator_system_message: |
    You are an expert code reviewer specializing in PDF processing and RAG systems.
    Evaluate code quality, efficiency, and adherence to best practices.

  num_top_programs: 3
  num_diverse_programs: 2
  use_template_stochasticity: true

# Database configuration
database:
  population_size: 200
  archive_size: 50
  num_islands: 3
  migration_interval: 20
  migration_rate: 0.15
  
  elite_selection_ratio: 0.15
  exploration_ratio: 0.3
  exploitation_ratio: 0.7
  
  feature_dimensions:
    - "combined_score"
    - "semantic_coherence"
  feature_bins: 8

# Evaluator configuration
evaluator:
  timeout: 120
  max_retries: 2
  cascade_evaluation: true
  cascade_thresholds:
    - 0.3
    - 0.6
    - 0.8
  parallel_evaluations: 2
  use_llm_feedback: true
  llm_feedback_weight: 0.1
```

### 5. Requirements File

```txt
# requirements.txt
openevolve
PyMuPDF==1.23.8
nltk==3.8.1
scikit-learn==1.3.0
numpy==1.24.3
openai==1.3.0
```

## Running the Evolution

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv pdf_chunker_env
source pdf_chunker_env/bin/activate  # On Windows: pdf_chunker_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Run Evolution

```python
# run_evolution.py
import asyncio
from openevolve import OpenEvolve

async def main():
    # Initialize OpenEvolve
    evolve = OpenEvolve(
        initial_program_path="initial_program.py",
        evaluation_file="evaluator.py",
        config_path="config.yaml"
    )
    
    # Run evolution
    best_program = await evolve.run(iterations=500)
    
    print("Evolution completed!")
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save the best program
    with open("best_pdf_chunker.py", "w") as f:
        f.write(best_program.code)

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Command Line Execution

```bash
python openevolve-run.py initial_program.py evaluator.py \
  --config config.yaml \
  --iterations 500
```

## Key Evolution Strategies

### 1. Multi-Objective Optimization

The evaluator measures multiple aspects:
- **Chunking Score**: Number and distribution of chunks
- **Semantic Coherence**: How well chunks maintain semantic meaning
- **Boundary Quality**: How well chunks break at natural boundaries
- **Metadata Quality**: Completeness of chunk metadata
- **Performance Score**: Processing speed

### 2. MAP-Elites Diversity

The system maintains diversity through:
- **Feature Map**: Tracks programs across different performance dimensions
- **Island Model**: Separate populations evolve independently
- **Migration**: Best solutions spread between islands

### 3. LLM-Guided Evolution

The LLM receives context about:
- Past successful programs and their metrics
- Common failure patterns
- Performance improvements over generations
- Specific areas for improvement

## Advanced Techniques

### 1. Artifacts Integration

```python
# Enhanced evaluator with artifacts
from openevolve.evaluation_result import EvaluationResult

def evaluate_with_artifacts(program_path: str) -> EvaluationResult:
    # ... evaluation logic ...
    
    # Capture artifacts for better LLM feedback
    artifacts = {
        "sample_chunks": str(chunks[:2]),
        "processing_time": str(total_time),
        "error_log": error_log if error_log else "None"
    }
    
    return EvaluationResult(
        metrics=metrics,
        artifacts=artifacts
    )
```

### 2. Custom Prompt Templates

```python
# Custom prompt template for PDF chunking
PDF_CHUNKING_PROMPT = """
You are evolving a PDF chunker for RAG applications. Consider these aspects:

1. **Semantic Boundaries**: Break at sentence/paragraph boundaries when possible
2. **Content Types**: Handle academic papers, technical docs, and mixed content
3. **Metadata**: Preserve page numbers, character positions, and source info
4. **Performance**: Optimize for speed while maintaining quality
5. **Robustness**: Handle malformed PDFs and edge cases

Current best approaches:
- Use PyMuPDF for reliable text extraction
- Implement sliding window with overlap
- Consider document structure (headers, sections)
- Balance chunk size vs. semantic coherence

Improve the chunk_pdf function based on the evaluation metrics.
"""
```

### 3. Cascade Evaluation

The system uses multi-stage evaluation:
1. **Stage 1**: Quick syntax and basic functionality check
2. **Stage 2**: Performance and chunking quality assessment
3. **Stage 3**: Full evaluation with all metrics

## Monitoring and Analysis

### 1. Checkpoint Analysis

```python
# Analyze evolution progress
import json
import os

def analyze_checkpoints(checkpoint_dir: str):
    checkpoints = sorted([d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint_')])
    
    for checkpoint in checkpoints[-5:]:  # Last 5 checkpoints
        info_path = os.path.join(checkpoint_dir, checkpoint, 'best_program_info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print(f"{checkpoint}: {info['metrics']['combined_score']:.4f}")

```

### 2. Visualization

```bash
# Start the evolution visualizer
python scripts/visualizer.py --path pdf_chunker_evolution/openevolve_output/
```

## Expected Evolution Patterns

### Early Generations (1-50)
- Basic functionality improvements
- Error handling and robustness
- Simple optimizations

### Middle Generations (51-200)
- Semantic boundary detection
- Metadata enhancement
- Performance optimizations

### Late Generations (201-500)
- Advanced techniques (NLP-based chunking)
- Multi-document type handling
- Fine-tuning of parameters

## Best Practices

1. **Start Simple**: Begin with a basic but functional chunker
2. **Diverse Test Data**: Use different types of PDFs (academic, technical, mixed)
3. **Balanced Metrics**: Don't over-optimize for a single metric
4. **Regular Checkpoints**: Save progress frequently for analysis
5. **Monitor Diversity**: Ensure the population doesn't converge too early
6. **Artifact Feedback**: Use artifacts to provide better LLM guidance

## Troubleshooting

### Common Issues

1. **Low Diversity**: Increase exploration ratio or add more islands
2. **Slow Convergence**: Adjust LLM temperature or add more diverse examples
3. **Poor Performance**: Check evaluation function and test data
4. **API Errors**: Verify API key and rate limits

### Debugging Tips

```python
# Enable debug logging
config.log_level = "DEBUG"

# Check program database state
print(f"Database size: {len(evolve.database.programs)}")
print(f"Best score: {evolve.database.get_best_program().metrics}")

# Analyze feature map
print(f"Feature map entries: {len(evolve.database.feature_map)}")
```

This guide provides a comprehensive framework for evolving PDF chunkers using OpenEvolve. The key is to start with a functional baseline and let the system discover increasingly sophisticated approaches through guided evolution. 