from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.losses import MultipleNegativesRankingLoss,CoSENTLoss,ContrastiveLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from evaluators.CosineSimilarityEvaluator import CosineSimilarityEvaluator
from losses.CEContrastiveLoss import CEContrastiveLoss
from torch import nn
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from evaluators.CECosineSimilarityEvaluator import CECosineSimilarityEvaluator 


def model_mapping(model):
    if model == 'sentence-transformers/all-mpnet-base-v2':
        return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    elif model == 'mixedbread-ai/mxbai-embed-large-v1':
        return SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    elif model == 'thenlper/gte-large':
        return SentenceTransformer("thenlper/gte-large")
    elif model == 'sentence-transformers/stsb-mpnet-base-v2':
        return SentenceTransformer("sentence-transformers/stsb-mpnet-base-v2")
    elif model == 'sentence-transformers/stsb-roberta-base-v2':
        return SentenceTransformer("sentence-transformers/stsb-roberta-base-v2")


def loss_function_mapping(train_loss_class): 
    if train_loss_class == 'MultipleNegativesRankingLoss':
        return MultipleNegativesRankingLoss
    elif train_loss_class == 'ContrastiveLoss':
        return ContrastiveLoss
    elif train_loss_class == 'CoSENTLoss':
        return CoSENTLoss

def eval_function_mapping(evaluator): 
    if evaluator == 'EmbeddingSimilarity':
        return EmbeddingSimilarityEvaluator
    elif evaluator == 'CosineSimilarity':
        return CosineSimilarityEvaluator
 

def ce_model_mapping(model):
    if model == 'bert-base-uncased':
        return CrossEncoder("bert-base-uncased", num_labels=1)
    elif model == 'roberta-base':
        return CrossEncoder("roberta-base", num_labels=1)
    elif model == 'xlnet-base-cased':
        return CrossEncoder("xlnet-base-cased", num_labels=1)

def ce_loss_function_mapping(train_loss_class): 
    if train_loss_class == 'CEContrastiveLoss':
        return CEContrastiveLoss
    elif train_loss_class == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss

def ce_eval_function_mapping(evaluator):
    if evaluator == 'CECorrelationEvaluator':
        return CECorrelationEvaluator
    elif evaluator == 'CECosineSimilarityEvaluator':
        return CECosineSimilarityEvaluator