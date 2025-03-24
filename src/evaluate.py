"""
Evaluation module for ICD-10 classification.
"""
from typing import List, Tuple, Dict, Any


def evaluate(predicted: List[str], true: List[str]) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1-score for predicted vs. true ICD-10 codes.
    
    Args:
        predicted: List of predicted ICD-10 codes
        true: List of true ICD-10 codes
        
    Returns:
        Tuple containing precision, recall, and F1-score
    """
    pred_set, true_set = set(predicted), set(true)
    tp = len(pred_set & true_set)  # True positives
    fp = len(pred_set - true_set)  # False positives
    fn = len(true_set - pred_set)  # False negatives
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1


def print_evaluation_results(predicted: List[str], true: List[str]) -> None:
    """
    Print evaluation results in a human-readable format.
    
    Args:
        predicted: List of predicted ICD-10 codes
        true: List of true ICD-10 codes
    """
    precision, recall, f1 = evaluate(predicted, true)
    
    print(f"Evaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print correctly predicted and missed codes
    true_set = set(true)
    pred_set = set(predicted)
    correct = true_set.intersection(pred_set)
    missed = true_set - pred_set
    false_positives = pred_set - true_set
    
    print(f"\nCorrectly predicted ({len(correct)}): {', '.join(sorted(correct))}")
    print(f"Missed ({len(missed)}): {', '.join(sorted(missed))}")
    print(f"False positives ({len(false_positives)}): {', '.join(sorted(false_positives))}")


def run_evaluation(
    dataset: Dict[str, Any],
    icd10_df: Any,
    icd10_embeddings: Any,
    tokenizer: Any,
    model: Any,
    api_key: str,
    predict_func: callable,
    num_samples: int = 5
) -> None:
    """
    Run evaluation on a subset of the test dataset.
    
    Args:
        dataset: Test dataset containing clinical notes and codes
        icd10_df: DataFrame containing ICD-10 codes and descriptions
        icd10_embeddings: Pre-computed embeddings for ICD-10 descriptions
        tokenizer: BERT tokenizer
        model: BERT model
        api_key: Claude API key
        predict_func: Function to predict codes given inputs
        num_samples: Number of samples to evaluate
    """
    total_precision, total_recall, total_f1 = 0, 0, 0
    
    print(f"Evaluating on {num_samples} samples...\n")
    
    for i in range(min(num_samples, len(dataset))):
        note = dataset[i]["user"]
        true_codes = dataset[i]["codes"]
        
        print(f"Example {i+1}:")
        print(f"Clinical Note: {note[:200]}..." if len(note) > 200 else f"Clinical Note: {note}")
        print(f"True Codes: {true_codes}")
        
        predicted_codes = predict_func(
            note, icd10_embeddings, icd10_df, tokenizer, model, api_key
        )
        
        print(f"Predicted Codes: {predicted_codes}")
        precision, recall, f1 = evaluate(predicted_codes, true_codes)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    # Calculate averages
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_f1 = total_f1 / num_samples
    
    print(f"Overall Results (avg of {num_samples} samples):")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")