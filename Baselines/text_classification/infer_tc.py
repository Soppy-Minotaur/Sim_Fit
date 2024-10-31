
import numpy as np
import torch
import os
from transformers import TrainingArguments, Trainer
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
import csv
from Evaluation.evaluation import calculate_metrics

class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

def evaluate_tc(model,tokenizer,output_path,test_dataset,class_set,text_col="DetailsofViolation",class_col="NOVCodeDescription"):
    """
    Infer with a selected traditioal classification model. 

    If infering with a custom dataset not within model folder, specify test_dataset, text_col, and class_col.
    """


    test_df = pd.read_csv(test_dataset)
    class_set_df = pd.read_csv(class_set)

    inputs = test_df[text_col].to_list()
    target_indices = test_df[class_col].to_list()
    target_indices = list(map(int, target_indices))
    class_set = class_set_df["class_name"].to_numpy()
    test_encodings = tokenizer(inputs, truncation=True, padding=True)
    print("testing set tokenized")
    
    test_dataset = Dataset(test_encodings, target_indices)

    # Create a results folder
    output_dir = os.path.join(output_path,'results')
    os.makedirs(output_dir, exist_ok=True)

    # Create logging folder
    logging_folder_path = os.path.join(output_path,'test_logs')
    os.makedirs(logging_folder_path, exist_ok=True)

    train_args = TrainingArguments(output_dir=output_dir,per_device_eval_batch_size=64,logging_dir= logging_folder_path,do_eval=True)
    trainer = Trainer(model=model, args=train_args, eval_dataset=test_dataset)
    predictions_prob, _, _ = trainer.predict(test_dataset, metric_key_prefix="predict")
    predicted_indices = np.argmax(predictions_prob, axis=1)
    predicted_classes = class_set[predicted_indices]
    target_classes = class_set[target_indices]
    metrics = calculate_metrics(predicted_indices, target_indices)

    results = {"Metrics":metrics,"Inputs":inputs,"Predicted Classes":predicted_classes,"Target Classes":target_classes}

    return results

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

    metrics = {"accuracy": accuracy, "kappa": kappa, "mcc": mcc, "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1}

    return metrics

def write_results(training_params:dict, results:dict, output_path:str):
    """
    Write the training parameters and evaluation results to a CSV file.
    """
    with open(output_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=training_params.keys() + results.keys())
        if not os.path.isfile(output_path):
            writer.writeheader()
        writer.writerows(training_params + results)
