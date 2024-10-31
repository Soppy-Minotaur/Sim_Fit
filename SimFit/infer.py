import pandas as pd
import numpy as np
import os
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from .infer_util import encode
from Evaluation.evaluation import calculate_metrics

def evaluate(model_path:str, test_dataset:Optional[str]=None, text_col:str="DetailsofViolation", class_col:str="NOVCodeDescription"):
    """
    Infer with a selected Sim FIt model. 

    If infering with a custom dataset not within model folder, specify test_dataset, text_col, and class_col.
    """
 
    inputs_emb, targets_set_emb, inputs, target_classes, targets_set, target_indices = encode(model_path,test_dataset=test_dataset,
                                                                                           text_col=text_col,class_col=class_col)
    predicted_indices, predicted_classes = predict(inputs_emb, targets_set_emb, targets_set)
    metrics = calculate_metrics(predicted_indices, target_indices)

    results = {"Metrics":metrics,"Inputs":inputs,"Predicted Classes":predicted_classes,"Target Classes":target_classes,
               "Class Set":targets_set,"Predicted Indices":predicted_indices,"Target Indices":target_indices}

    return results

def predict(inputs_emb, targets_set_emb, targets_set, sim_function = cosine_similarity):
    """
    Function to take in embeddings of inputs and targets, and output the predicted classes indices and texts for each input.
    """

    # Computing similarity scores (each row one input, each column one class)
    similarity_scores = sim_function(inputs_emb, targets_set_emb)
    # Outputing a list of which index on each row has the highest value
    predictions = np.argmax(similarity_scores, axis=1)

    return predictions, targets_set[predictions]



def all_inputs_of_class(inputs,targets,sorted_df,class_name:str=None,class_index:int=None):
    """
    For a given class, outputs all sentences belonging to that class. Note, the input index should be 
    referencing the sorted metrics list based on class frequency

    Outputs:
    1. A list of all sentences belonging to a given class.
    """
    if class_name == None and class_index != None:
        # when a class index integer is given
        class_list = sorted_df["Static Code Description"].to_list()
        class_name = class_list[class_index]
    
    matching_sentences = []
    # Iterate through both lists
    for sentence, category in zip(inputs, targets):
        if category == class_name:
            matching_sentences.append(sentence)
    
    # Print the matching sentences
    print(f"Sentences belonging to {class_name}:")
    for sentence in matching_sentences:
        print(sentence)
    return matching_sentences

def all_assigned_to_class(inputs,targets_set,predictions,sorted_df,class_name:str=None,class_index:int=None):
    """
    For a given class, outputs all sentences predicted to be belonging to that class

    Outputs:
    1. A list of all sentences predicted to be belonging to given class.
    """
    if class_name == None and class_index != None:
        # when a class index integer is given, get class name from the sorted class list
        class_list = sorted_df["Static Code Description"].to_list()
        class_name = class_list[class_index]
    
    matching_sentences = []
    # find the index of the class name in the unnique list of classes
    index = targets_set.index(class_name)
    for sentence,category in zip(inputs,predictions):
        if category == index:
            matching_sentences.append(sentence)

    return matching_sentences



def compute_input_specific_scores(similarity_scores,inputs,targets_set,input_index:int=None,input_name:str=None,save_input_specifc_results:bool=True,model_path:str=None):
    """
    Outputs similarity scores with all classes for a given input, and saves it as a csv (optional).
    Input speficied by either the name (str) or index (int).

    Outputs:
    1. A sorted pandas dataframe containing similarity scores for every class for the specified input
    """

    def find_index_of_ith_largest(arr, i):
        indices = np.argsort(arr)  # Get the indices that would sort the array
        ith_largest_index = indices[-i-1]  # Get the index of the ith largest value
        return ith_largest_index
    
    if input_name != None and input_index == None:
        # the case when input name is given instead of the index
        input_index = inputs.index(input_name)

    ith_sim = similarity_scores[input_index,:]
    output_list = []
    for j in range(len(ith_sim)):
        index = find_index_of_ith_largest(ith_sim, j)
        output_list.append((ith_sim[index], targets_set[index]))

    # Create a pandas DataFrame from the class_metrics list
    sim_df = pd.DataFrame(output_list, columns=['Similarity Score', 'Static Code Description'])
    print(f"Similarity scores retrieved for input number {input_index}.")
    if save_input_specifc_results == True:

        # Define the output path
        output_path = os.path.join(model_path,'testing_results')
        os.makedirs(output_path, exist_ok=True)

        # Save the DataFrame to the specified CSV file
        sim_df.to_csv(os.path.join(output_path,f"sim_scores_for_input_{input_index}.csv"), index=False)

        print(f"Similarity scores for input number {input_index} saved.")
   
    return sim_df
