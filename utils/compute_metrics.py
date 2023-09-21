# from evaluate import evaluator
# import evaluate
from datasets import load_metric

def compute_metrics(pred):
    squad_labels = pred.label_ids
    squad_preds = pred.predictions.argmax(-1)

    # Calculate Exact Match (EM)
    em = sum([1 if p == l else 0 for p, l in zip(squad_preds, squad_labels)]) / len(squad_labels)

    # Calculate F1-score
    f1 = f1_score(squad_labels, squad_preds, average='macro')

    return {
        'exact_match': em,
        'f1': f1
    }