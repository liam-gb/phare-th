import pandas as pd
from unittest.mock import patch
from src.load_data import load_icd10_data, load_test_dataset

def test_load_icd10_data():
    # Mock pd.read_csv to return a sample DataFrame
    sample_data = pd.DataFrame({'raw_data': ['"Desc1" | "A00"', '"Desc2" | "B00"']})
    with patch('pandas.read_csv', return_value=sample_data):
        df = load_icd10_data()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['Description', 'ICD10_Code']
        assert df.iloc[0]['Description'] == 'Desc1'
        assert df.iloc[0]['ICD10_Code'] == 'A00'

def test_load_test_dataset():
    # Mock load_dataset to return a sample DatasetDict
    from datasets import DatasetDict, Dataset
    sample_test_dataset = Dataset.from_dict({'user': ['note1'], 'codes': [['A00']]})
    sample_dataset = DatasetDict({'test': sample_test_dataset})
    
    with patch('datasets.load_dataset', return_value=sample_dataset):
        dataset = load_test_dataset()
        assert hasattr(dataset, '__getitem__')  # Check it's a Dataset-like object
        assert dataset[0]['user'] == 'note1'
        assert dataset[0]['codes'] == ['A00']