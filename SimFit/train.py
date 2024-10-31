
import pandas as pd
from sentence_transformers import InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss,CoSENTLoss,ContrastiveLoss
from .losses.SentenceCrossEntropyLoss import SentenceCrossEntropyLoss
from .evaluators.SentenceCrossEntropyEvaluator import SentenceCrossEntropyEvaluator
from .evaluators.SentenceAccuracyEvaluator import SentenceAccuracyEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import SimilarityFunction
import os
import csv
from .train_util import generate_negative_pairs
from .function_map import model_mapping, loss_function_mapping, eval_function_mapping

def train(train_dataset,val_dataset,test_dataset,model,output_path,train_neg_samp_ratio,val_neg_samp_ratio,
            learning_rate,batch_size,scale,warmup_ratio,num_epochs,evaluator,train_loss_class,margin,
            text_col="DetailsofViolation",class_col="NOVCodeDescription"):
    """
    Given all training parameters, train a model and save the evaluation results to a CSV file.

    train_loss_class can be one of the following:
    1. MultipleNegativesRankingLoss
    2. CoSENTLoss
    3. ContrastiveLoss (requires additional argument margin = margin_value)

    evaluator can be one of the following: 
    1. CosineSimilarityEvaluator
    2. EmbeddingSimilarityEvaluator

    Must specify text and class columns for custom datasets
    """
            
    train_df = pd.read_csv(train_dataset)
    val_df = pd.read_csv(val_dataset)
    print("val_df: ",val_df.shape[0])
    test_df = pd.read_csv(test_dataset)
    shots = train_df.shape[0] / 83

    # Save datasets used to model folder
    dataset_output_dir = os.path.join(output_path, 'dataset')
    os.makedirs(dataset_output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(dataset_output_dir, 'train_dataset.csv'), index=False)
    test_df.to_csv(os.path.join(dataset_output_dir, 'test_dataset.csv'), index=False)
    val_df.to_csv(os.path.join(dataset_output_dir, 'val_dataset.csv'), index=False)

    # Preprare training data according to whether negative samples are needed
    train_examples = []

    # create loss function instance
    if train_loss_class == ContrastiveLoss:
        train_loss = train_loss_class(model,margin=margin)
    elif train_loss_class == SentenceCrossEntropyLoss:
        train_loss = train_loss_class(model,scale=scale)
    else:
        train_loss = train_loss_class(model)
    
    # create input examples for loss function
    if not isinstance(train_loss, (MultipleNegativesRankingLoss, SentenceCrossEntropyLoss)):
        generate_negative_pairs(train_dataset, os.path.join(dataset_output_dir, 'train_dataset_wneg.csv'), train_neg_samp_ratio, "0")
        train_df = pd.read_csv(os.path.join(dataset_output_dir, 'train_dataset_wneg.csv'))
        n_examples = len(train_df)
        for i in range(n_examples):
            train_examples.append(InputExample(texts=[train_df.iloc[i][text_col], train_df.iloc[i][class_col]], label=train_df.iloc[i]['Labels']))
    elif isinstance(train_loss, SentenceCrossEntropyLoss):
        classes = tuple(train_df[class_col].unique())
        n_examples = len(train_df)
        for i in range(n_examples):
            label = train_df.iloc[i][class_col]
            indexed_label = classes.index(label)
            train_examples.append(InputExample(texts=[train_df.iloc[i][text_col]], label=indexed_label))
    else:
        n_examples = len(train_df)
        for i in range(n_examples):
            train_examples.append(InputExample(texts=[train_df.iloc[i][text_col], train_df.iloc[i][class_col]]))
    print("train_examples length: ",len(train_examples))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size) 
    train_sim_func = SimilarityFunction.COSINE
  
    # Prepare validation data where there must be negative samples for contrast
    generate_negative_pairs(val_dataset, os.path.join(dataset_output_dir, 'val_dataset_wneg.csv'), val_neg_samp_ratio, "0") # gemerate this just in case Cross Encoder needs it downstream
    val_sim_func = SimilarityFunction.EUCLIDEAN # SimilarityFunction.EUCLIDEAN
    if evaluator is not SentenceCrossEntropyEvaluator and evaluator is not SentenceAccuracyEvaluator:
        val_df = pd.read_csv(os.path.join(dataset_output_dir, 'val_dataset_wneg.csv'))
        val_score_labels = val_df["Labels"].tolist()
        val_classes = val_df[class_col].tolist()
        val_inputs = val_df[text_col].tolist()
        evaluator = evaluator(val_classes, val_inputs, val_score_labels, write_csv=True, show_progress_bar=True, main_similarity=val_sim_func)
    else:
        val_df = pd.read_csv(os.path.join(dataset_output_dir, 'val_dataset.csv'))
        val_labels = val_df[class_col].tolist()
        val_inputs = val_df[text_col].tolist()
        val_classes = train_df[class_col].unique().tolist()
        evaluator = evaluator(val_labels, val_inputs, val_classes, write_csv=True, show_progress_bar=True, main_similarity=val_sim_func)
    eval_steps = 0 # evaluate once per epoch

    # List out training parameters for recording
    warmup_steps = int(len(train_dataloader) * num_epochs * warmup_ratio) 
    trained_model = os.path.basename(output_path)
    random_state = 42
    
    # Write training parameters to CSV
    with open(os.path.join(output_path, 'training_params.csv'), 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(["Trained Model", "Pretrained on", "No. Shots", "Training Loss", "Scale",
                        "Margin","Evaluator","Train Neg Ratio","Val Neg Ratio","Learning Rate",
                        "Batch Size", "Warmup Ratio", "Num Epochs", "Random State"])

        # Write the data
        writer.writerow([trained_model,model,shots, str(train_loss)+ "("+str(train_sim_func)+")", scale, margin,str(evaluator)+"("+str(val_sim_func)+")",
                         train_neg_samp_ratio, val_neg_samp_ratio, learning_rate, batch_size, warmup_ratio, num_epochs,
                         random_state])

    if not isinstance(train_loss, SentenceCrossEntropyLoss):
        model.fit(train_objectives=[(train_dataloader, train_loss)],evaluator=evaluator,evaluation_steps=eval_steps, warmup_steps=warmup_steps, 
          epochs=num_epochs, show_progress_bar=True,
            output_path=output_path,save_best_model=True,optimizer_params={'lr': learning_rate})
    else:
        model.CrossEntropyfit(train_objectives=[(train_dataloader, train_loss)],classes=classes,evaluator=evaluator,evaluation_steps=eval_steps, warmup_steps=warmup_steps, 
          epochs=num_epochs, show_progress_bar=True,
            output_path=output_path,save_best_model=True,optimizer_params={'lr': learning_rate})



def main():
     # Read parameters from CSV
    train_configs_df = pd.read_csv('c:\Cambridge\Journal_Codes\Training_Schedule\training_configs.csv')

    # Iterate over each row and pass parameters to train function
    for index, row in train_configs_df.iterrows():
        train(train_dataset=row['train_dataset'],
              val_dataset=row['val_dataset'],
              test_dataset=row['test_dataset'],
              model=row['model'],
              output_path=row['output_path'],
              train_neg_samp_ratio=int(row['train_neg_samp_ratio']),
              val_neg_samp_ratio=int(row['val_neg_samp_ratio']),
            learning_rate=float(row['learning_rate']),
            batch_size=int(row['batch_size']),
            warmup_ratio=float(row['warmup_ratio']),
            num_epochs=int(row['num_epochs']),
            evaluator=row['evaluator'],
            eval_steps=int(row['eval_steps']),
            train_loss_class=row['train_loss_class'],
            margin=float(row['margin']))
     
if __name__ == '__main__':
    main()