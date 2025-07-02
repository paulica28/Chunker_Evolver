#!/usr/bin/env python3
"""
Run PDF chunker evolution using OpenEvolve
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the openevolve directory to the path
openevolve_path = Path(__file__).parent.parent / "openevolve"
if openevolve_path.exists():
    sys.path.insert(0, str(openevolve_path))

from openevolve import OpenEvolve

async def main():
    """Run the PDF chunker evolution"""
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("Starting PDF chunker evolution...")
    print("This will evolve a PDF chunker for RAG applications")
    print("=" * 50)
    
    # Initialize OpenEvolve
    evolve = OpenEvolve(
        initial_program_path="initial_program.py",
        evaluation_file="evaluator.py",
        config_path="config.yaml"
    )
    
    # Run evolution
    print("Running evolution for 200 iterations...")
    best_program = await evolve.run(iterations=200)
    
    print("\n" + "=" * 50)
    print("Evolution completed!")
    print(f"Best program metrics:")
    for name, value in best_program.metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    
    # Save the best program
    with open("best_pdf_chunker.py", "w") as f:
        f.write(best_program.code)
    
    print(f"\nBest program saved to: best_pdf_chunker.py")
    print(f"Checkpoint data available in: openevolve_output/")
    
    # Show some sample output
    print("\nSample of best program:")
    lines = best_program.code.split('\n')
    for i, line in enumerate(lines[:20]):  # Show first 20 lines
        print(f"{i+1:3d}: {line}")
    if len(lines) > 20:
        print("...")

if __name__ == "__main__":
    asyncio.run(main()) 