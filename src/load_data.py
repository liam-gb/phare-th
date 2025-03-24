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
    icd10_df = pd.read_csv(url, sep="|")
    return icd10_df


def load_test_dataset() -> Dict[str, Any]:
    """
    Load the test split of the synthetic EHR dataset.
    
    Returns:
        Test dataset containing clinical notes and their associated ICD-10 codes
    """
    dataset = load_dataset("FiscaAI/synth-ehr-icd10cm-prompt")
    return dataset["test"]