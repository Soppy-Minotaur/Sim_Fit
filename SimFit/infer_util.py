import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm

def encode(model_path:str, test_dataset, text_col, class_col):
    """
    Encodes a given dataset, either from within the model folder or from a specified dataset.

    Returns inputs_emb, targets_set_emb, inputs, targets, targets_set, target_indices
    
    """
    inf_model = SentenceTransformer(model_path)

    if not test_dataset:
        test_df = pd.read_csv(os.path.join(model_path,"dataset","test_dataset.csv"))
    else:
        test_df = pd.read_csv(test_dataset)

    # Preparing a unique set of classes, and replacing the original target list with one containing indices referencing the class set
    targets = test_df[class_col].to_list()
    inputs = test_df[text_col].to_list() 
    targets_set, labels_arr = np.unique(targets, return_inverse=True)
    target_indices = labels_arr.tolist()

    # Computing embedding for both lists
    targets_set_emb = inf_model.encode(targets_set, show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=True)
    print('Targets encoded')
    inputs_emb = inf_model.encode(inputs, show_progress_bar=True,convert_to_numpy=True,normalize_embeddings=True)
    print('Input sentences encoded')

    return inputs_emb, targets_set_emb, inputs, targets, targets_set, target_indices

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



def top_k_accuracy(inputs_emb, targets_set_emb, labels, k_range:list, sim_function = cosine_similarity):
    """
    k_range is the range of k values we want to calculate accuracies for. It should be a list of all k values to test.
    """
    similarity_scores = sim_function(inputs_emb, targets_set_emb)
    result = []

    # Iterate over each row
    for k in k_range:
        result_per_k = np.zeros(similarity_scores.shape[0])
        for i in range(similarity_scores.shape[0]):
            # Get the current row
            row = similarity_scores[i]
            # Find the k largest values in the row
            # We use np.partition to get the k largest elements at the end of the array
            top_k_values = np.partition(row, -k)[-k:]
            # Check if the value at the given index is in these top k values
            if row[labels[i]] in top_k_values:
                result_per_k[i] = True
            else:
                result_per_k[i] = False
        result.append(np.mean(result_per_k)) # result is a list of all the top k accuracy for each k value

    return result

def calculate_classwise_metrics(targets, predicted_classes, train_dataset, val_dataset, test_dataset, model_path, class_col = "NOVCodeDescription"):
   
    class_counts = Counter(targets)
    class_metrics = []
    train_df = pd.read_csv(train_dataset)
    val_df = pd.read_csv(val_dataset)
    test_df = pd.read_csv(test_dataset)

    for class_label in tqdm(class_counts.keys(), desc="Computing metrics"):
        # calculate metrics per class
        true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
        for true_label, pred_label in zip(targets, predicted_classes):
            if true_label == class_label and pred_label == class_label:
                true_positive += 1
            elif true_label == class_label and pred_label != class_label:
                false_negative += 1
            elif true_label != class_label and pred_label == class_label:
                false_positive += 1
            else:
                true_negative += 1
        
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)


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

        # append the class's data
        train_count = train_df[class_col].value_counts().get(class_label, 0)
        val_count = val_df[class_col].value_counts().get(class_label, 0)
        test_count = test_df[class_col].value_counts().get(class_label, 0)
        dataset_count = train_count + val_count + test_count
        class_metrics.append((class_label, train_count, val_count, class_counts[class_label], dataset_count, accuracy, precision, recall, f1))
        
        
        
    # Create a pandas DataFrame from the class_metrics list
    print("Classwise metrics calculated.")
    classwise_metric_df = pd.DataFrame(class_metrics, columns=['Road Code Definitions', 'No. in Training', 'No. in Validation','No. in Testing', 'No. in Dataset','Accuracy', 'Precision', 'Recall', 'F1'])
    sorted_classwise_metric_df = classwise_metric_df.sort_values(by='F1', ascending=False)
    output_path = os.path.join(model_path,'classwise_testing_results')
    os.makedirs(output_path, exist_ok=True)
    sorted_classwise_metric_df.to_csv(os.path.join(output_path,"classwise_metrics.csv"), index=False)

    return sorted_classwise_metric_df