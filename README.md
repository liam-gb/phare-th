# ICD-10 Code Prediction for Clinical Notes



## Overview



This project implements a system for automatically predicting ICD-10 diagnosis codes from clinical notes using a Retrieval-Augmented Generation (RAG) approach. The system combines:



1. **Clinical BERT Embeddings**: Uses Bio_ClinicalBERT to generate embeddings for both clinical notes and ICD-10 descriptions
2. **Similarity Matching**: Identifies candidate ICD-10 codes based on embedding similarity 
3. **Claude API Refinement**: Uses the Claude language model to make final code selections from candidates



## Approach



The system uses a modular architecture with these key components:


- **Data Loading**: Loads ICD-10 codes/descriptions and test clinical notes
- **Preprocessing**: Normalizes text, expands abbreviations, and corrects spelling
- **Embedding Generation**: Creates vector representations of text using Bio_ClinicalBERT
- **Retrieval**: Finds similar ICD-10 codes using vector similarity
- **Generation**: Filters and refines candidate codes using Claude API
- **Evaluation**: Measures precision, recall, and F1 score of predictions

This approach offers several advantages:
- No training required (leverages pre-trained models)
- Combines both embedding-based retrieval and LLM-based classification
- Handles the complexity of medical terminology and context

The approach is inspired by Lee and Lindsay, 2024: https://arxiv.org/pdf/2403.10822

Unfortunately, the ICD-10 encoding step took too long for me to run within 2 hours to test the pipeline end-to-end, so I created tests for each component in the /tests folder. 

Additionally, I had limited time to explore the data to design appropriate cleaning (and evaluate the impact of the cleaning on performance), so I included a sketch of some steps of the cleaning process in preprocessing.py. 

## Requirements

```python
datasets>=2.12.0
pandas>=1.5.3
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.2
requests>=2.31.0
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Obtain a Claude API key

## Usage

### Process a single clinical note:

```bash
python main.py --api_key YOUR_CLAUDE_API_KEY --note "Patient presents with shortness of breath..."
```

### Run evaluation on the test dataset:

```bash
python main.py --api_key YOUR_CLAUDE_API_KEY --num_samples 5
```

You can also set the API key as an environment variable:

```bash
export CLAUDE_API_KEY=your_api_key
python main.py --num_samples 10
```

## Project Structure

```
.
├── main.py              # Main script to run the pipeline
├── requirements.txt     # Python dependencies
└── src/
    ├── load_data.py     # Functions to load data
    ├── preprocessing.py # Text preprocessing functions
    ├── train.py         # Placeholder (no training needed)
    ├── utils.py         # Utility functions including API calls
    └── evaluate.py      # Evaluation metrics and reporting
```

## Limitations and Future Improvements

- **Performance**: Encoding all ICD-10 descriptions is memory-intensive and could be optimized
- **Clinical Preprocessing**: Could be enhanced with more domain-specific preprocessing
- **Evaluation**: Current metrics are basic; could add weighted F1 and confusion matrices
- **Model Training**: Could fine-tune BERT on the specific domain for better embeddings
- **Data Privacy**: A production system would need more secure handling of PHI

## References

- Bio_ClinicalBERT: [https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- ICD-10 Codes: GoodMedicalCoder dataset
- Claude API: [https://docs.anthropic.com/claude/reference/getting-started-with-the-api](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)