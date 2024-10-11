import pandas as pd
import os

def create_prompt(text:str,classes:list, prompt_type:str):

    classes = ', '.join(classes)

    if prompt_type == "generic":
        prompt = f"Pick one category for the following text. The text is: {text}. The options are: {classes}."

    elif prompt_type == "task":
        prompt = f"Classify the input text into one of the following categories. The text is: {text}. The categories are: {classes}."

    elif prompt_type == "domain":
        prompt = f"Select the road code that is being violated in each instance. The instance is: {text}. The road codes are: {classes}."

    return prompt

def create_prompt_dataset(dataset,output_path,prompt_type="task",text_col="DetailsofViolation",class_col="NOVCodeDescription"):
    dataset_df = pd.read_csv(dataset)
    classes = dataset_df[class_col].unique().tolist()
    for i in range(dataset_df.shape[0]):
        dataset_df.iloc[i][text_col] = create_prompt(str(dataset_df.iloc[i][text_col]),classes, prompt_type)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_df.to_csv(output_path, index=False)

