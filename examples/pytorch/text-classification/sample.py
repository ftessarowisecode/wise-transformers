#

from sklearn.metrics import precision_recall_fscore_support

def compute_global_metrics(predictions, labels):
    total_samples = 0
    correct_samples = 0
    all_preds = []
    all_labels = []

    # Iterate through each level
    for level_pred, level_label in zip(predictions, labels):
        # Check if the lengths of predictions and labels at this level are the same
        if len(level_pred) != len(level_label):
            raise ValueError("Mismatch in the number of predictions and labels at a level.")

        # Iterate through each example at this level
        for pred, label in zip(level_pred, level_label):
            # Check if all levels in the prediction match the corresponding levels in the label
            if all(p == l for p, l in zip(pred, label)):
                correct_samples += 1

            all_preds.extend(pred)
            all_labels.extend(label)

            total_samples += 1

    # Calculate precision, recall, and F1 score globally
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    # Calculate global accuracy
    global_accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

    return global_accuracy, precision, recall, f1


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    assert len(logits) == 2
    result_dict = {}
    argmax_predictions = []
    # compute for each level metrics
    for i, (level_logits, level_labels) in enumerate(zip(logits, labels)):
        predictions = np.argmax(level_logits, axis=-1)
        argmax_predictions.append(predictions)

        precision = metric1.compute(predictions=predictions, references=level_labels, average="macro")["precision"]
        recall = metric2.compute(predictions=predictions, references=level_labels, average="macro")["recall"]
        f1 = metric3.compute(predictions=predictions, references=level_labels, average="macro")["f1"]
        accuracy = metric.compute(predictions=predictions, references=level_labels)["accuracy"]
        result_dict.update({f"accuracy_L{i}": accuracy})
        result_dict.update({f"f1_L{i}": f1})
        result_dict.update({f"recall_L{i}": recall})
        result_dict.update({f"precision_L{i}": precision})
    accuracy, precision, recall, f1 = compute_global_metrics(argmax_predictions, labels)
    result_dict.update({f"accuracy_L{i}": accuracy})
    result_dict.update({f"f1": f1})
    result_dict.update({f"recall": recall})
    result_dict.update({f"precision": precision})
    
    return result_dict