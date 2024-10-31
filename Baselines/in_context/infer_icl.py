from typing import Optional
import numpy as np
import torch
import os
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
import os
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
import csv
from tqdm import tqdm
from .function_map import tokenizer_mapping, model_mapping
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
        

def evaluate_icl_no_training(model_name, train_dataset, test_dataset, shot, text_col="DetailsofViolation", class_col="NOVCodeDescription"):
    

    # Generate prompts
    train_df = pd.read_csv(train_dataset)
    class_set = train_df[class_col].unique()
    class_set_dict = {class_name: class_index for class_index, class_name in enumerate(class_set)} # class_set_dict[class_name] = class_index
    train_df["Class Index"] = train_df[class_col].map(class_set_dict)

    def generate_prompts(train_df, class_col, text_col, shot):
        prompts = []
        if shot > 0:
            for cls in class_set:
                class_samples = train_df[train_df[class_col] == cls][text_col].sample(n=shot, random_state=1).tolist()
                for i in range(len(class_samples)):
                    class_samples[i] = f"Input: {class_samples[i]}\nClass: {cls}"
                prompts += class_samples
            random.shuffle(prompts)
            prompts_str = ""
            for i in range(len(prompts)):
                prompts_str += f"Example {i+1}:\n{prompts[i]}\n\n"
        else:
            prompts_str = ""
            prompts_str += "These are all the possible classes:\n"
            i = 0
            for cls in class_set:
                i += 1
                prompts_str += f"Class {i}: {cls}\n"   
        return prompts_str

    def add_prompts(row, prompts):
        input = row[text_col]
        prompts += f"Now, classify the following input according to the possible classes:\nInput: {input}\nClass:"
        return prompts 

    test_df = pd.read_csv(test_dataset)
    test_df["Class Index Column"] = test_df[class_col].map(class_set_dict)
    target_classes = test_df[class_col].to_list()
    target_indices = test_df["Class Index Column"].to_list()
    prompts = generate_prompts(train_df, class_col, text_col, shot)
    tqdm.pandas()
    test_df['Prompt'] = test_df.progress_apply(lambda row: add_prompts(row,prompts), axis=1)
    test_prompts = test_df['Prompt'].to_list()

    tokenizer = tokenizer_mapping(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = model_mapping(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_size = 32  
    predicted_classes = []

    # Process prompts in batches
    for i in tqdm(range(0, len(test_prompts), batch_size)):
        batch_prompts = test_prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(inputs['input_ids'], max_new_tokens=200)
        batch_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        predicted_classes.extend(batch_responses)

    predicted_indices = map(lambda x: class_set_dict[x], predicted_classes)

    metrics = calculate_metrics(predicted_indices, target_indices)

    results = {"Metrics":metrics,"Inputs":inputs,"Predicted Classes":predicted_classes,"Target Classes":target_classes,
               "Class Set":class_set,"Predicted Indices":predicted_indices,"Target Indices":target_indices}

    return results

    


    
    





def evaluate_icl(model, output_path, tokenizer,test_dataset,class_set,text_col="DetailsofViolation",class_col="NOVCodeDescription"):
    """
    Infer with a selected traditioal classification model. 

    If infering with a custom dataset not within model folder, specify test_dataset, text_col, and class_col.
    """
  
    test_df = pd.read_csv(test_dataset)
    class_set_df = pd.read_csv(class_set)

    inputs = test_df[text_col].to_list()
    target_indices = test_df[class_col].to_list()
    class_set = class_set_df[class_col].to_numpy()
    test_encodings = tokenizer(inputs, truncation=True, padding=True)
    
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
