from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from optuna import Trial
from typing import Dict, Union, Any
import torch
import csv
import pandas as pd
import os
import gc

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

def model_init(params: Dict[str, Any]) -> SetFitModel:
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5", **params)

def hp_space(trial: Trial) -> Dict[str, Union[float, int, str]]:
    return {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-3, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "seed": trial.suggest_int("seed", 1, 40),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
    }

def get_mapping_labels(train_dataset, class_col="NOVCodeDescription"):
    train_df = pd.read_csv(train_dataset)
    train_df[class_col], class_set = pd.factorize(train_df[class_col])
    class_set_dict = {value: index for index, value in enumerate(class_set)}
    return class_set_dict

def map_labels(example, class_set_dict):
        example['label'] = class_set_dict[example['label']]
        return example


def hyperparam_search(train_dataset, val_dataset, class_set_dict, n_trials, output_path, model_init=model_init, hp_space=hp_space,
                          text_col="DetailsofViolation", class_col="NOVCodeDescription"):
    
    dataset = load_dataset('csv', data_files={'train': train_dataset, 'test': val_dataset})
    dataset = dataset.rename_column(class_col, 'label')  # 'labels' is the default name expected by most models
    dataset = dataset.rename_column(text_col, 'text') 

    # Map string labels to integers for both train and test datasets
    dataset = dataset.map(lambda x: map_labels(x, class_set_dict))
    train_dataset = dataset['train']
    val_dataset = dataset['test'] 

    trainer = Trainer(train_dataset=train_dataset,eval_dataset=val_dataset,model_init=model_init)


    best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=n_trials)
    print(best_run)
  
    """# Access the study object from the trainer
    study = best_run.study

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Trial", "Body Learning Rate", "Num Epochs", "Batch Size", "Seed", "Max Iter", "Solver", "Score"])
        for trial in study.trials:
            writer.writerow([
                trial.number,
                trial.params.get("body_learning_rate"),
                trial.params.get("num_epochs"),
                trial.params.get("batch_size"),
                trial.params.get("seed"),
                trial.params.get("max_iter"),
                trial.params.get("solver"),
                trial.value  # The evaluation score (objective)
            ])"""

def train_setfit(train_dataset, val_dataset, class_set_dict, output_path, model, max_iter, solver, body_learning_rate, head_learning_rate, num_epochs, batch_size, 
                 text_col="DetailsofViolation", class_col="NOVCodeDescription"):
    """
    model can be "BAAI/bge-small-en-v1.5"
    
    """
    dataset = load_dataset('csv', data_files={'train': train_dataset, 'test': val_dataset})
    dataset = dataset.rename_column(class_col, 'label')  # 'labels' is the default name expected by most models
    dataset = dataset.rename_column(text_col, 'text') 

    # Map string labels to integers for both train and test datasets
    dataset = dataset.map(lambda x: map_labels(x, class_set_dict))
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    model = SetFitModel.from_pretrained(model, head_params={"max_iter": max_iter,"solver": solver})
    body_learning_rate = tuple(map(float, body_learning_rate.split(',')))
    num_epochs = tuple(map(int, num_epochs.split(',')))
    batch_size = tuple(map(int, batch_size.split(',')))
    training_args = TrainingArguments(   
    num_epochs=num_epochs,      
    body_learning_rate=body_learning_rate,  
    head_learning_rate = head_learning_rate,
    batch_size=batch_size)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        args=training_args, 
        )

    with open(os.path.join(output_path, 'training_params.csv'), 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['Model','Pretrained on','max_iter',
                         'solver','body_learning_rate','num_epochs',
                         'batch_size','train_dataset','val_dataset'])

        # Write the data
        writer.writerow([os.path.basename(output_path),model,max_iter, solver, body_learning_rate, num_epochs, batch_size, train_dataset, val_dataset])
    trainer.train()
    model.save_pretrained(output_path)
    metrics = trainer.evaluate()
    print(metrics)

    
    del metrics
    del trainer
    del model
    del train_dataset
    del val_dataset
    del dataset
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
 
    gc.collect()
    torch.cuda.empty_cache()

