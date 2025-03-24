"""
Functions for loading the dataset and ICD-10 descriptions.
"""
import pandas as pd
from datasets import load_dataset
from typing import Dict, Any


def load_icd10_data() -> pd.DataFrame:
    """
    Load the ICD-10 descriptions from a CSV file.
    
    Returns:
        DataFrame containing ICD-10 codes and their descriptions
    """
    url = "https://raw.githubusercontent.com/ainativehealth/GoodMedicalCoder/main/ICD-10_formatted.csv"
    
    # Read the file as a single column (no separator)
    raw_df = pd.read_csv(url, header=None, names=['raw_data'])
    
    # Split the raw data into description and code
    raw_df[['Description', 'ICD10_Code']] = raw_df['raw_data'].str.split(r' \| ', expand=True)
    
    # Clean up the data (remove quotes if present)
    raw_df['Description'] = raw_df['Description'].str.replace('"', '')
    raw_df['ICD10_Code'] = raw_df['ICD10_Code'].str.replace('"', '')
    
    # Drop the original raw data column
    result_df = raw_df[['Description', 'ICD10_Code']]
    
    return result_df


def load_test_dataset() -> Dict[str, Any]:
    """
    Load the test split of the synthetic EHR dataset.
    
    Returns:
        Test dataset containing clinical notes and their associated ICD-10 codes
    """
    dataset = load_dataset("FiscaAI/synth-ehr-icd10cm-prompt")
    return dataset["test"]