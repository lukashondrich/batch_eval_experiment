#!/usr/bin/env python
"""
Main script to run the batch evaluation experiment.
"""
import argparse
import os
import time
from dotenv import load_dotenv
from src.evaluation import load_sample_data, run_all_evaluations, save_results

def main():
    """Run the batch evaluation experiment."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run batch evaluation experiment")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                        help="Model to use for evaluations (default: gpt-4o)")
    parser.add_argument("--trials", type=int, default=10, 
                        help="Number of trials to run (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.2, 
                        help="Temperature for generation (default: 0.0)")
    parser.add_argument("--repetition_penalty", type=float, default=0.0, 
                        help="Repetition penalty for generation (default: 0.0)")
    parser.add_argument("--output", type=str, default="data/raw/evaluation_results.json", 
                        help="Output file path (default: data/raw/evaluation_results.json)")
    args = parser.parse_args()
    
    # Ensure OpenAI API key is set
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        return 1
    
    # Load sample data
    print("Loading sample data...")
    try:
        sample_data = load_sample_data()
        print(f"Loaded {len(sample_data['samples'])} samples.")
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return 1
    
    # Run the experiment
    print(f"Running experiment with model: {args.model}, trials: {args.trials}, temperature: {args.temperature}, repetition_penalty: {args.repetition_penalty}")
    start_time = time.time()
    
    results = run_all_evaluations(
        sample_data, 
        trials=args.trials,
        model=args.model,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty
    )
    
    elapsed_time = time.time() - start_time
    print(f"Experiment completed in {elapsed_time:.2f} seconds.")
    
    # Save results
    output_path = save_results(results, args.output)
    print(f"Results saved to: {output_path}")
    
    # Summary statistics
    print("\nExperiment Summary:")
    print(f"- Model: {args.model}")
    print(f"- Trials: {args.trials}")
    print(f"- Temperature: {args.temperature}")
    print(f"- Repetition Penalty: {args.repetition_penalty}")
    print(f"- Samples: {len(sample_data['samples'])}")
    print(f"- Methods: baseline_batch, independent, filler_token_batch")
    print("\nNext steps:")
    print("- Run analysis: python -m src.analysis")
    print("- Create visualizations: python -m src.visualization")
    
    return 0

if __name__ == "__main__":
    exit(main())