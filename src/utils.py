"""
Utilities for clinical text processing and ICD-10 prediction.
"""
import requests
import numpy as np
import torch
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import (
    COMMON_MEDICAL_ABBREVIATIONS,
    preprocess_clinical_text,
    batch_preprocess,
    encode_note
)


def load_medical_dictionary() -> Dict[str, str]:
    """
    Load a dictionary of commonly misspelled medical terms.
    
    Returns:
        Dictionary mapping misspelled terms to their correct forms
    """
    # This is a minimal example - in production, this would load from a file
    return {
        "arthritus": "arthritis",
        "ostearthritis": "osteoarthritis",
        "diabeties": "diabetes",
        "hipertension": "hypertension",
        "colestrol": "cholesterol",
        "obisity": "obesity",
        "neumonia": "pneumonia",
        "rheumatism": "rheumatism",
        "fatige": "fatigue",
        "inflamation": "inflammation",
    }


def demo_preprocessing() -> None:
    """
    Demonstrate the preprocessing pipeline with example clinical texts.
    """
    example_texts = [
        "Pt is a 68yo female w/ hx of HTN, DM, and osteoarthritis. MRI shows effusion!",
        "Patient had CT scan which revealed osteoarthiritis and mild inflammation.",
        "Pt. c/o SOB, prescribed NSAID for pain management; follow-up in 2 wks."
    ]
    
    # Load dictionaries
    medical_dict = load_medical_dictionary()
    abbreviation_dict = COMMON_MEDICAL_ABBREVIATIONS
    
    print("Original texts:")
    for i, text in enumerate(example_texts):
        print(f"{i+1}. {text}")
    
    print("\nPreprocessed texts:")
    processed_texts = batch_preprocess(
        example_texts,
        medical_dict=medical_dict,
        abbreviation_dict=abbreviation_dict
    )
    
    for i, text in enumerate(processed_texts):
        print(f"{i+1}. {text}")


def call_claude_api(prompt: str, api_key: str) -> str:
    """
    Call the Claude API with a given prompt.
    
    Args:
        prompt: Text prompt to send to the API
        api_key: Claude API key
        
    Returns:
        Generated text response from the API
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 150,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()["content"][0]["text"]


def predict_codes(
    note: str, 
    icd10_embeddings: torch.Tensor, 
    icd10_df: Any, 
    tokenizer: Any, 
    model: Any, 
    api_key: str
) -> List[str]:
    """
    Predict ICD-10 codes for a clinical note using a RAG approach.
    
    Args:
        note: Clinical note text
        icd10_embeddings: Pre-computed ICD-10 description embeddings
        icd10_df: DataFrame containing ICD-10 codes and descriptions
        tokenizer: BERT tokenizer
        model: BERT model
        api_key: Claude API key
        
    Returns:
        List of predicted ICD-10 codes
    """
    # For test purposes, check the column names and adapt accordingly
    code_column = 'code' if 'code' in icd10_df.columns else 'ICD10_Code'
    desc_column = 'description' if 'description' in icd10_df.columns else 'Description'
    
    # Encode the clinical note
    note_emb = encode_note(note, tokenizer, model).numpy() if tokenizer and model else torch.randn(768).numpy()
    
    # Compute cosine similarities with ICD-10 embeddings
    similarities = cosine_similarity([note_emb], icd10_embeddings.numpy())[0]
    top_indices = np.argsort(similarities)[-20:][::-1]  # Top 20 most similar
    top_codes = icd10_df.iloc[top_indices][code_column].tolist()
    top_descriptions = icd10_df.iloc[top_indices][desc_column].tolist()
    
    # Create API call with full clinical note
    candidate_text = "\n".join([f"{i+1}. {code} - {desc}" 
                               for i, (code, desc) in enumerate(zip(top_codes, top_descriptions))])
    
    selection_prompt = (
        f"You are a medical coding expert specializing in ICD-10 codes. Review this clinical note and select the most appropriate ICD-10 codes from the candidates.\n\n"
        f"Clinical note: {note}\n\n"
        f"Candidate ICD-10 codes:\n{candidate_text}\n\n"
        "Return only the relevant ICD-10 codes from the list, separated by commas. Do not include explanations."
    )
    
    response = call_claude_api(selection_prompt, api_key)
    
    # Parse the response into a list of codes
    # Clean up in case the response includes explanations despite instructions
    cleaned_response = response.strip().split("\n")[0]  # Take only first line to avoid explanations
    predicted_codes = [code.strip() for code in cleaned_response.split(",")]
    
    return predicted_codes


if __name__ == "__main__":
    demo_preprocessing()