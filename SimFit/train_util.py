import csv
import os
import random
import pandas as pd


def generate_negative_pairs(input_file, output_file, neg_samp_ratio=4, neg_rep="0"):
    """
    Generates negative pairings according to neg_samp_ratio.
    A new column represents negative pairs with neg_rep, positive with "1"
    """
    try:
        # Read the input CSV file
        data = []
        with open(input_file, 'r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i > 0:  # Skip the header row
                    data.append(row)

        # Generate positive pairs
        positive_pairs = [[class_, item, '1'] for class_, item in data]
        negative_pairs = []

        # Calculate the desired number of negative pairs
        desired_negative_pairs = neg_samp_ratio * len(positive_pairs)

        # Generate negative pairs randomly
        while len(negative_pairs) < desired_negative_pairs:
            class_a, item_a = random.choice(data)
            class_b, item_b = random.choice(data)
            if class_a != class_b:
                negative_pairs.append([class_a, item_b, neg_rep])

        # Combine the positive and negative pairs
        updated_data = positive_pairs + negative_pairs
        random.shuffle(updated_data)
        headings = ["NOVCodeDescription","DetailsofViolation","Labels"]
        # Shuffle the updated data
        random.shuffle(updated_data)

        # Write the updated data to the output CSV file
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headings)
            writer.writerows(updated_data)

        print(f"Generated negative pairs successfully. Output written to {output_file}.")

    except FileNotFoundError:
        print("Input file not found.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def create_next_model_folder(output_directory:str, template_name:str):
    """
    Automatically creates a new model folder in the models directory and names it. Outputs the new model folder directory,
    and the model id, which is the same as the new folder name.

    template name should be one of the three: 'sf_model_', 'sr_model_', 'tc_model_'
    """

    all_folders = os.listdir(output_directory)
    model_folders = [folder for folder in all_folders if folder.startswith(template_name)]
    iteration_numbers = [int(folder.split('_')[-1]) for folder in model_folders if folder.split('_')[-1].isdigit()]
    highest_iteration = max(iteration_numbers, default=0)
    model_name = f'{template_name}{highest_iteration + 1}'

    # Create the new folder
    model_output_path = os.path.join(output_directory, model_name)
    os.makedirs(model_output_path, exist_ok=True)
    print(f"New folder, {model_output_path}, was created successfully.")
    
    return model_output_path, model_name


def write_results(training_params:dict, results:dict, output_path:str):
    """
    Write the training parameters and evaluation results to a CSV file.
    """
    with open(output_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=training_params.keys() + results.keys())
        if not os.path.isfile(output_path):
            writer.writeheader()
        writer.writerows(training_params + results)


def fetch_biencoder_datasets(biencoder_model_path,ce_output_path):
    
    # Preparing Bi-Encoder data
    original_train_df = pd.read_csv(os.path.join(biencoder_model_path,"dataset","train_dataset.csv"))
    shots = original_train_df.shape[0] / 83
    
    if os.path.join(biencoder_model_path,"dataset","train_dataset_wneg.csv").is_file():
        train_dataset = os.path.join(biencoder_model_path,"dataset","train_dataset_wneg.csv")

    else:
        generate_negative_pairs(os.path.join(biencoder_model_path,"dataset", 'train_dataset.csv'), os.path.join(ce_output_path,'ce_train_dataset_wneg.csv'), train_neg_ratio)
        train_dataset = os.path.join(ce_output_path,"ce_train_dataset_wneg.csv")
    
    val_dataset = os.path.join(biencoder_model_path,"dataset","val_dataset_wneg.csv")
    test_dataset = os.path.join(biencoder_model_path,"dataset","test_dataset_wneg.csv")
    
    return train_dataset, val_dataset, test_dataset, shots