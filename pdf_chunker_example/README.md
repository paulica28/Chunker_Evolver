# PDF Chunker Evolution Example

This example demonstrates how to use OpenEvolve to evolve an optimal PDF chunker for RAG (Retrieval-Augmented Generation) applications.

## Overview

The goal is to evolve a PDF chunking algorithm that:
- Splits PDF documents into meaningful chunks
- Maintains semantic coherence within chunks
- Breaks at natural boundaries (sentences, paragraphs)
- Preserves metadata and document structure
- Optimizes for both quality and performance

## Files

- `initial_program.py`: Starting PDF chunker implementation
- `evaluator.py`: Evaluation logic and metrics
- `config.yaml`: OpenEvolve configuration
- `requirements.txt`: Dependencies
- `run_evolution.py`: Script to run the evolution

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Run the evolution**:
   ```bash
   python run_evolution.py
   ```

## Evaluation Metrics

The evaluator measures multiple aspects:

1. **Chunking Score**: Number and distribution of chunks
2. **Semantic Coherence**: How well chunks maintain semantic meaning
3. **Boundary Quality**: How well chunks break at natural boundaries
4. **Metadata Quality**: Completeness of chunk metadata
5. **Performance Score**: Processing speed

## Expected Evolution

### Early Generations (1-50)
- Basic functionality improvements
- Error handling and robustness
- Simple optimizations

### Middle Generations (51-150)
- Semantic boundary detection
- Metadata enhancement
- Performance optimizations

### Late Generations (151-200)
- Advanced techniques (NLP-based chunking)
- Multi-document type handling
- Fine-tuning of parameters

## Results

After evolution, you'll find:
- `best_pdf_chunker.py`: The evolved chunker
- `openevolve_output/`: Checkpoint data and logs
- Evolution metrics and performance data

## Customization

To customize the evolution:

1. **Modify the initial program** in `initial_program.py`
2. **Adjust evaluation criteria** in `evaluator.py`
3. **Tune evolution parameters** in `config.yaml`
4. **Add your own test PDFs** to the `test_pdfs/` directory

## Key Features

- **Multi-objective optimization**: Balances multiple quality metrics
- **MAP-Elites diversity**: Maintains diverse solution population
- **LLM-guided evolution**: Uses AI to generate improvements
- **Cascade evaluation**: Multi-stage evaluation for efficiency
- **Checkpointing**: Resume evolution from any point

## Troubleshooting

- **API errors**: Check your OpenAI API key and rate limits
- **Low diversity**: Increase exploration ratio in config
- **Slow convergence**: Adjust LLM temperature or add examples
- **Poor performance**: Check evaluation function and test data # Chunker_Evolver
