import torch
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.utils import load_medical_dictionary, call_claude_api, predict_codes

def test_load_medical_dictionary():
    medical_dict = load_medical_dictionary()
    assert isinstance(medical_dict, dict)
    assert 'arthritus' in medical_dict
    assert medical_dict['arthritus'] == 'arthritis'

@patch('requests.post')
def test_call_claude_api(mock_post):
    mock_post.return_value.json.return_value = {"content": [{"text": "A00, B00"}]}
    response = call_claude_api("Test prompt", "fake_key")
    assert response == "A00, B00"

@patch('src.utils.encode_note', return_value=torch.randn(768))
@patch('sklearn.metrics.pairwise.cosine_similarity', return_value=np.array([[0.9, 0.8, 0.7]]))
@patch('src.utils.call_claude_api', return_value="A00, B00")
def test_predict_codes(mock_api, mock_similarity, mock_encode):
    note = "Test note"
    icd10_embeddings = torch.randn(3, 768)
    icd10_df = pd.DataFrame({'code': ['A00', 'B00', 'C00'], 'description': ['Desc A', 'Desc B', 'Desc C']})
    
    # Create mock tokenizer and model to ensure encode_note is called
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    predicted = predict_codes(note, icd10_embeddings, icd10_df, mock_tokenizer, mock_model, "fake_key")
    assert predicted == ['A00', 'B00']
    mock_encode.assert_called_once()
    mock_similarity.assert_called_once()
    mock_api.assert_called_once()