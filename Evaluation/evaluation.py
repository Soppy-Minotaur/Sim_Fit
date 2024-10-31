from collections import Counter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
import numpy as np
import pandas as pd
import os



def calculate_metrics(predicted_indices, target_indices):

    """
    Takes prediction and label indices and compute evaluation metrics.

    Returns a dictionary containing the following keys:
    - accuracy  
    - kappa
    - mcc
    - macro_precision
    - macro_recall
    - macro_f1
    """
   
    accuracy = np.mean(target_indices == predicted_indices)
    kappa = cohen_kappa_score(target_indices, predicted_indices)
    mcc = matthews_corrcoef(target_indices, predicted_indices)
    macro_precision = precision_score(target_indices, predicted_indices, average='macro')
    macro_recall = recall_score(target_indices, predicted_indices, average='macro')
    macro_f1 = f1_score(target_indices, predicted_indices, average='macro')

    results = {"accuracy": accuracy, "kappa": kappa, "mcc": mcc, "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1}

    return results

def calculate_classwise_metrics(target_indices, predicted_indices, class_set, output_path):

    class_counts = Counter(target_indices)
    class_metrics = []

    for class_index in tqdm(class_counts.keys(), desc="Computing metrics"):
        # calculate metrics per class
        true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
        for true_label, pred_label in zip(target_indices, predicted_indices):
            if true_label == class_index and pred_label == class_index:
                true_positive += 1
            elif true_label == class_index and pred_label != class_index:
                false_negative += 1
            elif true_label != class_index and pred_label == class_index:
                false_positive += 1
            else:
                true_negative += 1
        
        if true_positive + false_positive == 0:
            precision = 0
        else:
            precision = true_positive / (true_positive + false_positive)
        if true_positive + false_negative == 0:
            recall = 0
        else:
            recall = true_positive / (true_positive + false_negative)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        class_name = class_set[class_index]
        # append the class's data
        class_metrics.append((class_name, class_counts[class_index], precision, recall, f1))
        
    # Create a pandas DataFrame from the class_metrics list
    print("Classwise metrics calculated.")
    classwise_metric_df = pd.DataFrame(class_metrics, columns=['Static Code Description', 'No. of Test Examples', 'Precision', 'Recall', 'F1'])
    sorted_classwise_metric_df = classwise_metric_df.sort_values(by='No. of Test Examples', ascending=False)
    os.makedirs(output_path, exist_ok=True)
    sorted_classwise_metric_df.to_csv(os.path.join(output_path,"classwise_metrics.csv"), index=False)

    return sorted_classwise_metric_df


