"""
Preprocessing module for clinical text data.
"""
import re
import os
from typing import Dict, List, Optional, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModel


def medical_spell_check(text: str, spell_checker=None, medical_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Correct spelling errors in medical terms.
    
    Args:
        text: Raw clinical text
        spell_checker: Medical-specific spell checker (if available)
        medical_dict: Custom dictionary of medical terms for spelling correction
        
    Returns:
        Spell-corrected text
    """
    # If a spell checker is provided, use it
    if spell_checker:
        return spell_checker(text)
    
    # If no spell checker available but a medical dictionary is provided
    # This is a simple implementation - a real implementation would be more sophisticated
    if medical_dict:
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Check if a misspelled version of the word exists in our dictionary
            # Use lowercase for dictionary lookup but preserve original case
            word_lower = word.lower()
            
            # Handle punctuation at the end of words
            word_clean = word_lower.rstrip('.,;:!?')
            punctuation = word[len(word_clean):] if len(word_clean) < len(word) else ''
            
            if word_clean in medical_dict:
                # Preserve original capitalization if possible
                if word[0].isupper() and len(medical_dict[word_clean]) > 0:
                    replacement = medical_dict[word_clean][0].upper() + medical_dict[word_clean][1:]
                else:
                    replacement = medical_dict[word_clean]
                corrected_words.append(replacement + punctuation)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    # If neither spell checker nor dictionary is available, return original text
    return text


def normalise_text(text: str) -> str:
    """
    Standardise text by converting to lowercase and removing special characters.
    Retains letters, numbers, spaces, and hyphens.
    
    Args:
        text: Input text to normalise
        
    Returns:
        Normalised text
    """
    # Convert to lowercase
    lower_text = text.lower()
    
    # Remove special characters except alphanumeric, spaces, and hyphens
    normalised_text = re.sub(r'[^a-z0-9\s-]', '', lower_text)
    
    # Replace multiple spaces with single space
    normalised_text = re.sub(r'\s+', ' ', normalised_text)
    
    return normalised_text.strip()


def standardise_abbreviations(text: str, abbreviation_dict: Dict[str, str]) -> str:
    """
    Expand medical abbreviations to their full forms based on provided dictionary.
    
    Args:
        text: Input text with abbreviations
        abbreviation_dict: Dictionary mapping abbreviations to their full forms
        
    Returns:
        Text with expanded abbreviations
    """
    words = text.split()
    standardised_words = []
    
    for word in words:
        # Convert to lowercase for abbreviation lookup
        word_lower = word.lower()
        
        # Check if word (lowercase) is in the abbreviation dictionary
        if word_lower in abbreviation_dict:
            # Replace with the expanded form
            standardised_words.append(abbreviation_dict[word_lower])
        else:
            # Keep the original word
            standardised_words.append(word)
    
    return ' '.join(standardised_words)


def preprocess_clinical_text(
    text: str,
    spell_checker=None, 
    medical_dict: Optional[Dict[str, str]] = None,
    abbreviation_dict: Optional[Dict[str, str]] = None
) -> str:
    """
    Apply the complete preprocessing pipeline to clinical text.
    
    Args:
        text: Raw clinical text
        spell_checker: Medical spell checker function (optional)
        medical_dict: Dictionary for spell correction (optional)
        abbreviation_dict: Dictionary of abbreviations and their full forms (optional)
        
    Returns:
        Preprocessed text ready for tokenisation
    """
    # Step 1: Abbreviation standardisation (before normalization to preserve word boundaries)
    if abbreviation_dict:
        expanded_text = standardise_abbreviations(text, abbreviation_dict)
    else:
        expanded_text = text
        
    # Step 2: Medical spell checking
    corrected_text = medical_spell_check(expanded_text, spell_checker, medical_dict)
    
    # Step 3: Text normalisation
    normalised_text = normalise_text(corrected_text)
    
    return normalised_text


def batch_preprocess(
    texts: List[str],
    spell_checker=None,
    medical_dict: Optional[Dict[str, str]] = None,
    abbreviation_dict: Optional[Dict[str, str]] = None,
    batch_size: int = 1000
) -> List[str]:
    """
    Process a large list of clinical texts in batches for efficiency.
    
    Args:
        texts: List of clinical texts to preprocess
        spell_checker: Medical spell checker function (optional)
        medical_dict: Dictionary for spell correction (optional)
        abbreviation_dict: Dictionary of abbreviations and their full forms (optional)
        batch_size: Number of texts to process in each batch
        
    Returns:
        List of preprocessed texts
    """
    processed_texts = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        processed_batch = [
            preprocess_clinical_text(
                text, 
                spell_checker, 
                medical_dict, 
                abbreviation_dict
            ) for text in batch
        ]
        processed_texts.extend(processed_batch)
    
    return processed_texts


def setup_bert() -> Tuple[AutoTokenizer, AutoModel]:
    """
    Initialize the Clinical BERT tokenizer and model.
    
    Returns:
        Tuple containing the tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return tokenizer, model


def encode_icd10_descriptions(icd10_df, tokenizer, model, cache_path=None) -> torch.Tensor:
    """
    Encode all ICD-10 descriptions into embeddings.
    
    Args:
        icd10_df: DataFrame containing ICD-10 codes and descriptions
        tokenizer: BERT tokenizer
        model: BERT model
        cache_path: Path to save/load embeddings cache (optional)
        
    Returns:
        Tensor of embeddings for each ICD-10 description
    """
    # Try to load cached embeddings if cache_path is provided
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached ICD-10 embeddings from {cache_path}")
        try:
            return torch.load(cache_path)
        except Exception as e:
            print(f"Error loading cached embeddings: {e}. Generating new embeddings.")
    
    print("Generating ICD-10 embeddings (this may take some time)...")
    descriptions = icd10_df["Description"].tolist()
    embeddings = []
    
    # Use tqdm for progress bar if available
    try:
        from tqdm import tqdm
        desc_iter = tqdm(descriptions)
    except ImportError:
        desc_iter = descriptions
        
    for desc in desc_iter:
        inputs = tokenizer(desc, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze()
        embeddings.append(emb)
    
    result = torch.stack(embeddings)
    
    # Save to cache if path is provided
    if cache_path:
        print(f"Saving ICD-10 embeddings to {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(result, cache_path)
        
    return result


def encode_note(note: str, tokenizer, model) -> torch.Tensor:
    """
    Encode a single clinical note into an embedding.
    
    Args:
        note: Clinical note text
        tokenizer: BERT tokenizer
        model: BERT model
        
    Returns:
        Embedding tensor for the note
    """
    inputs = tokenizer(note, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()


# Example medical abbreviation dictionary
COMMON_MEDICAL_ABBREVIATIONS = {
    "mri": "magnetic resonance imaging",
    "ct": "computed tomography",
    "nsaid": "non-steroidal anti-inflammatory drug",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "chf": "congestive heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "cad": "coronary artery disease",
    "bmi": "body mass index",
    "bp": "blood pressure",
    "hr": "heart rate",
    "pt": "patient",
    "yo": "year old",
    "w/": "with",
    "w/o": "without",
    "rx": "prescription",
    "dx": "diagnosis",
    "hx": "history",
    "fx": "fracture",
    "sob": "shortness of breath",
}