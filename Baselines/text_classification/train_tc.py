import os
import pandas as pd
from function_map import tokenizer_class_mapping, tokenizer_name_mapping, model_class_mapping, model_name_mapping
import csv
import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_metric
from datasets import load_dataset
from evaluate import evaluator
from transformers import AutoModelForSequenceClassification, pipeline
import evaluate
import re

# Create the dataset class and dataset objects
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
        
def train_tc(
        train_dataset,
        val_dataset,
        test_dataset,
        tokenizer,
        model,
        output_path, 
        num_epochs,
        learning_rate,
        warmup_ratio,
        train_batch_size,
        eval_batch_size=64,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_steps=None,
        text_col="DetailsofViolation",
        class_col="NOVCodeDescription"
        ):
    


    """
    tokenizer_class should be one of the following: BertTokenizer,RobertaTokenizer,XLNetTokenizer
    tokenizer_name should be one of the following: bert-base-uncased,roberta-base,xlnet-base-cased
    model_class should be one of the following: BertForSequenceClassification,RobertaForSequenceClassification,XLNetForSequenceClassification
    model_name should be one of the following: bert-base-uncased,roberta-base,xlnet-base-cased
    """
    
    datasets_dict = process_datasets(train_dataset, val_dataset, test_dataset, output_path, text_col, class_col)

    train_encodings = tokenizer(datasets_dict["train_texts"], truncation=True, padding=True)
    val_encodings = tokenizer(datasets_dict["val_texts"], truncation=True, padding=True)
    test_encodings = tokenizer(datasets_dict["test_texts"], truncation=True, padding=True)  
    train_dataset = Dataset(train_encodings, datasets_dict["train_labels"])
    val_dataset = Dataset(val_encodings, datasets_dict["val_labels"])
    test_dataset = Dataset(test_encodings, datasets_dict["test_labels"])
    num_classes = len(set(datasets_dict["train_labels"]))

    results_folder_path = os.path.join(output_path,'results')
    os.makedirs(results_folder_path, exist_ok=True)
    logging_folder_path = os.path.join(output_path,'logs')
    os.makedirs(logging_folder_path, exist_ok=True)

    warmup_steps = int(warmup_ratio * num_epochs * len(train_dataset) / train_batch_size)

    training_args = TrainingArguments(
    output_dir=results_folder_path,        
    num_train_epochs=num_epochs,        
    learning_rate=learning_rate,     
    per_device_train_batch_size=train_batch_size, 
    per_device_eval_batch_size=eval_batch_size,  
    warmup_steps=warmup_steps,               
    weight_decay=weight_decay,              
    logging_dir=logging_folder_path,     
    logging_steps=logging_steps,
    evaluation_strategy="steps",
    eval_steps=evaluation_steps,
    save_on_each_node=False, # there was an issue with saving checkpoints, when changing name from tmp to not tmp
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_steps = evaluation_steps)

    trainer = Trainer(model=model,args=training_args,train_dataset=train_dataset,eval_dataset=val_dataset)

    with open(os.path.join(output_path, 'training_params.csv'), 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['Model','Pretrained on','Tokenizer Model',
                         'Epochs','Train Batch Size','Eval Batch Size',
                         'Warmup Steps','Weight Decay','Number of Classes','Training Dataset'])

        # Write the data
        writer.writerow([os.path.basename(output_path),model_name,tokenizer_name, 
                        num_epochs, train_batch_size,eval_batch_size, warmup_steps, weight_decay,
                        num_classes, train_dataset])
        
    trainer.train()
    trainer.save_model(output_path)



def process_datasets(train_dataset, val_dataset, test_dataset, output_path,text_col, class_col):
    train_df = pd.read_csv(train_dataset)
    val_df = pd.read_csv(val_dataset)
    test_df = pd.read_csv(test_dataset)
    train_df[class_col], class_set = pd.factorize(train_df[class_col])
    class_set_dict = {value: index for index, value in enumerate(class_set)}

    val_df[class_col] = val_df[class_col].map(class_set_dict).fillna(-1) # Use fillna(-1) to handle any unmatched categories
    test_df[class_col] = test_df[class_col].map(class_set_dict).fillna(-1)

    # Create a dataset folder and save the training, validation and testing data
    dataset_output_dir = os.path.join(output_path,'dataset')
    os.makedirs(dataset_output_dir, exist_ok=True)
    class_set_df = pd.DataFrame(class_set, columns=[class_col])
    train_df.to_csv(os.path.join(dataset_output_dir, 'train_dataset.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_output_dir, 'test_dataset.csv'), index=False)
    val_df.to_csv(os.path.join(dataset_output_dir, 'val_dataset.csv'), index=False)
    class_set_df.to_csv(os.path.join(dataset_output_dir, 'class_set.csv'), index=False)

    datasets_dict = {"train_texts": train_df[text_col].tolist(),"train_labels": train_df[class_col].tolist(),
                    "val_texts": val_df[text_col].tolist(),"val_labels": val_df[class_col].tolist(),
                    "test_texts": test_df[text_col].tolist(),"test_labels": test_df[class_col].tolist(),
                    "class_set_dict": class_set_dict}
    
    return datasets_dict

    