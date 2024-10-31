import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from .infer import encode
from .train_util import generate_negative_pairs
from .function_map import ce_model_mapping, ce_eval_function_mapping, ce_loss_function_mapping
import csv
from typing import Optional, Tuple
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader
from .losses.CECrossEntropyLoss import CECrossEntropyLoss
from .losses.CEContrastiveLoss import CEContrastiveLoss
from .evaluators.CEAccuracyEvaluator import CEAccuracyEvaluator


def train_ce(train_dataset:str,val_dataset:str,model,output_path,train_loss_class,evaluator,margin,scale,learning_rate,batch_size,warmup_ratio,num_epochs,
             text_col="DetailsofViolation",class_col="NOVCodeDescription",classes:Optional[Tuple[str]]=None, train_neg_ratio:Optional[int] = "Follow Bi-Encoder"):
    
    """
    Trains a Cross-Encoder model given the base Bi-Encoder model. 

    If no custom dataset, assume dataset present in bi-encoder model folder. 
    """

    train_df = pd.read_csv(train_dataset)
    train_examples = []
    n_examples = len(train_df)
    if train_loss_class is CEContrastiveLoss:
        train_loss = train_loss_class(margin=margin)
    elif train_loss_class is CECrossEntropyLoss:
        train_loss = train_loss_class(scale=scale)
    else:   
        train_loss = train_loss_class()

    if not isinstance(train_loss, CECrossEntropyLoss):
        for i in range(n_examples):
            train_examples.append(InputExample(texts=[train_df.iloc[i][text_col], train_df.iloc[i][class_col]], label=train_df.iloc[i]['Labels']))
    else:
        for i in range(n_examples):
            label = train_df.iloc[i][class_col]
            indexed_label = classes.index(label)
            train_examples.append(InputExample(texts=[train_df.iloc[i][text_col]], label=indexed_label))
    
    val_df = pd.read_csv(val_dataset)
    val_examples = []
    vn_examples = len(val_df)
    if evaluator is not CEAccuracyEvaluator:
        for i in range(vn_examples):
            val_examples.append(InputExample(texts=[val_df.iloc[i][text_col], val_df.iloc[i][class_col]], label=val_df.iloc[i]['Labels']))
        evaluator = evaluator.from_input_examples(val_examples,write_csv =True)
    else:
        for i in range(vn_examples):
            label = val_df.iloc[i][class_col]
            indexed_label = classes.index(label)
            val_examples.append(InputExample(texts=[val_df.iloc[i][text_col]], label=indexed_label))
        evaluator = evaluator.from_input_examples(val_examples, classes=classes, write_csv=True)
        
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    warmup_steps = int(len(train_dataloader) * num_epochs * warmup_ratio) 
    eval_steps = 0 # evaluate every epoch
    random_state = 42
    with open(os.path.join(output_path, 'training_params.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Trained CE Model',
                          'From Pre-trained CE Model',
                          'Negative to Positive Ratio',
                          'Train Loss Function',
                          'Margin',
                          'Scale',
                          'Validation Loss Function',
                          'Learning Rate',
                          'Batch Size',
                          'Warmup Ratio',
                          "Random State",
                          'Number of Epochs'])

        # Write the data
        writer.writerow([os.path.basename(output_path),
                          model,
                          train_neg_ratio,
                          train_loss,
                          margin,
                          scale,
                          evaluator,
                          learning_rate,
                          batch_size,
                          warmup_ratio,
                          random_state,
                          num_epochs])


        if isinstance(train_loss, CECrossEntropyLoss):
            model.CrossEntropyfit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            evaluation_steps=eval_steps,
            epochs=num_epochs,
            loss_fct = train_loss,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True,
            save_best_model=True)
        else:
            model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            evaluation_steps=eval_steps,
            epochs=num_epochs,
            loss_fct = train_loss,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True,
            save_best_model=True
            )

            
   
