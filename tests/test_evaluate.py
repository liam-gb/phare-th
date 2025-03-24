from src.evaluate import evaluate

def test_evaluate():
    predicted = ['A00', 'B00']
    true = ['A00', 'C00']
    precision, recall, f1 = evaluate(predicted, true)
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5

def test_evaluate_no_overlap():
    predicted = ['D00']
    true = ['A00']
    precision, recall, f1 = evaluate(predicted, true)
    assert precision == 0
    assert recall == 0
    assert f1 == 0

def test_evaluate_empty_predicted():
    predicted = []
    true = ['A00']
    precision, recall, f1 = evaluate(predicted, true)
    assert precision == 0
    assert recall == 0
    assert f1 == 0

def test_evaluate_empty_true():
    predicted = ['A00']
    true = []
    precision, recall, f1 = evaluate(predicted, true)
    assert precision == 0
    assert recall == 0
    assert f1 == 0