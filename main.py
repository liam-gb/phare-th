"""
Main script to run the ICD-10 classification pipeline.
"""
import os
import argparse
from typing import Dict, Any, List

from src.load_data import load_icd10_data, load_test_dataset
from src.preprocessing import setup_bert, encode_icd10_descriptions
from src.utils import predict_codes
from src.evaluate import evaluate, print_evaluation_results, run_evaluation


def main() -> None:
    """Main function to run the ICD-10 classification pipeline."""
    parser = argparse.ArgumentParser(description="ICD-10 Classification Pipeline")
    parser.add_argument("--api_key", type=str, help="Claude API key")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--note", type=str, help="Process a single clinical note")
    args = parser.parse_args()
    
    # Get API key from args or environment
    print(f"Environment variables: {os.environ.get('CLAUDE_API_KEY')}")

    api_key = args.api_key or os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError(
            "Claude API key must be provided via --api_key argument or CLAUDE_API_KEY environment variable"
        )
    
    print("Loading ICD-10 data...")
    icd10_df = load_icd10_data()
    
    print("Setting up Clinical BERT model...")
    tokenizer, model = setup_bert()
    
    # Path for cached embeddings
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    embeddings_cache_path = os.path.join(cache_dir, "icd10_embeddings.pt")
    
    # Load or generate ICD-10 embeddings
    icd10_embeddings = encode_icd10_descriptions(icd10_df, tokenizer, model, cache_path=embeddings_cache_path)
    
    if args.note:
        # Process a single provided note
        print("Processing single note...")
        predicted_codes = predict_codes(
            args.note, icd10_embeddings, icd10_df, tokenizer, model, api_key
        )
        print(f"Predicted ICD-10 Codes: {predicted_codes}")
    else:
        # Run evaluation on test dataset
        print("Loading test dataset...")
        test_dataset = load_test_dataset()
        
        run_evaluation(
            test_dataset, 
            icd10_df, 
            icd10_embeddings, 
            tokenizer, 
            model, 
            api_key,
            predict_codes,
            args.num_samples
        )


if __name__ == "__main__":
    main()