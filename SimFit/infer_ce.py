from typing import Optional
import numpy as np
from .infer_util import encode
from Evaluation.evaluation import calculate_metrics
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def evaluate_ce(model, biencoder_model_path, candidate_num, test_dataset:Optional[str] = None,
              text_col:str = "DetailsofViolation", class_col:str = "NOVCodeDescription"):

    inputs_emb, targets_set_emb, inputs, target_classes, targets_set, target_indices = encode(biencoder_model_path,test_dataset=test_dataset,
                                                                                           text_col=text_col,class_col=class_col)
    ce_predicted_indices, ce_predicted_classes = predict_ce(model, inputs_emb, targets_set_emb, inputs, targets_set, candidate_num)
    metrics = calculate_metrics(ce_predicted_indices, target_indices)
    results = {"Metrics":metrics,"Inputs":inputs,"Predicted Classes":ce_predicted_classes,"Target Classes":target_classes}

    return results


def predict_ce(model, inputs_emb, targets_set_emb, inputs, targets_set, candidate_num, sim_function = cosine_similarity):
    similarity_scores = sim_function(inputs_emb, targets_set_emb)
    neg_candidate_number = (-1) * candidate_num
    candidate_indices = np.argsort(similarity_scores, axis=1)[:, neg_candidate_number:]

    test_examples = []
    for i in range(len(inputs)):
        for j in candidate_indices[i,:]:
            test_examples.append([inputs[i],targets_set[j]])

    scores = model.predict(test_examples, convert_to_numpy=True, show_progress_bar=True)
    scores_2D = scores.reshape(-1, candidate_num)
    candidate_predictions = np.argmax(scores_2D, axis=1)
    rows = np.arange(candidate_indices.shape[0])
    ce_predicted_indices = candidate_indices[rows, candidate_predictions]
    return ce_predicted_indices, targets_set[ce_predicted_indices]
