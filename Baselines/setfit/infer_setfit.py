from datasets import load_dataset
from setfit import SetFitModel, Trainer
from typing import Optional
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
import gc
import torch

def get_mapping_labels(train_dataset, class_col="NOVCodeDescription"):
    train_df = pd.read_csv(train_dataset)
    train_df[class_col], class_set = pd.factorize(train_df[class_col])
    class_set_dict = {value: index for index, value in enumerate(class_set)}
    return class_set_dict

def map_labels(example, class_set_dict):
        example['label'] = class_set_dict[example['label']]
        return example


def compute_metrics_setfit(y_pred, y_test):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    matthews_corr = matthews_corrcoef(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cohen_kappa': cohen_kappa,
        'matthews_corrcoef': matthews_corr
    }

def evaluate_setfit(model_path:str, class_set_dict, test_dataset:Optional[str]=None, text_col:str="DetailsofViolation", class_col:str="NOVCodeDescription"):
    dataset = load_dataset('csv', data_files={'test': test_dataset})
    dataset = dataset.rename_column(class_col, 'label')  # 'labels' is the default name expected by most models
    dataset = dataset.rename_column(text_col, 'text') 

    # Map string labels to integers for both train and test datasets
    dataset = dataset.map(lambda x: map_labels(x, class_set_dict))
    test_dataset = dataset['test'] 
    model = SetFitModel.from_pretrained(model_path)

    trainer = Trainer(eval_dataset=test_dataset,model=model, metric=compute_metrics_setfit)

    metrics = trainer.evaluate()

    del trainer
    del model
    del test_dataset
    del dataset
  
   
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
    gc.collect()
    torch.cuda.empty_cache()

    return metrics