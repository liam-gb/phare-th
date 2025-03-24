from src.preprocessing import (
    medical_spell_check,
    normalise_text,
    standardise_abbreviations,
    preprocess_clinical_text
)

def test_medical_spell_check():
    medical_dict = {'arthritus': 'arthritis', 'diabeties': 'diabetes'}
    text = "Patient has arthritus and diabeties."
    corrected = medical_spell_check(text, medical_dict=medical_dict)
    assert corrected == "Patient has arthritis and diabetes."

def test_normalise_text():
    text = "Hello, World! This is a Test."
    normalized = normalise_text(text)
    assert normalized == "hello world this is a test"

def test_standardise_abbreviations():
    abbreviation_dict = {'pt': 'patient', 'hx': 'history'}
    text = "Pt has hx of hypertension."
    standardized = standardise_abbreviations(text, abbreviation_dict)
    assert standardized == "patient has history of hypertension."

def test_preprocess_clinical_text():
    medical_dict = {'arthritus': 'arthritis'}
    abbreviation_dict = {'pt': 'patient'}
    text = "Pt has arthritus."
    preprocessed = preprocess_clinical_text(text, medical_dict=medical_dict, abbreviation_dict=abbreviation_dict)
    assert preprocessed == "patient has arthritis"