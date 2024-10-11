import os
import csv

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
    file_exists = os.path.isfile(output_path)
    with open(output_path, mode='a', newline='') as file:
        fieldnames = list(training_params.keys()) + list(results.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({**training_params, **results})