from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.losses import MultipleNegativesRankingLoss,CoSENTLoss,ContrastiveLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from .losses.SentenceCrossEntropyLoss import SentenceCrossEntropyLoss
from .losses.CrossEntropySentenceTransformer import CrossEntropySentenceTransformer
from .evaluators.CosineSimilarityEvaluator import CosineSimilarityEvaluator
from .evaluators.SentenceCrossEntropyEvaluator import SentenceCrossEntropyEvaluator
from .evaluators.SentenceAccuracyEvaluator import SentenceAccuracyEvaluator
from .losses.CrossEntropyCrossEncoder import CrossEntropyCrossEncoder
from .losses.CEContrastiveLoss import CEContrastiveLoss
from .losses.CECrossEntropyLoss import CECrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from .evaluators.CECosineSimilarityEvaluator import CECosineSimilarityEvaluator 
from .evaluators.CEAccuracyEvaluator import CEAccuracyEvaluator
from torch import nn



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
    else:
        return SentenceTransformer(model)
    
def cross_entropy_model_mapping(model):
    if model == 'sentence-transformers/all-mpnet-base-v2':
        return CrossEntropySentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    elif model == 'mixedbread-ai/mxbai-embed-large-v1':
        return CrossEntropySentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    elif model == 'thenlper/gte-large':
        return CrossEntropySentenceTransformer("thenlper/gte-large")
    elif model == 'sentence-transformers/stsb-mpnet-base-v2':
        return CrossEntropySentenceTransformer("sentence-transformers/stsb-mpnet-base-v2")
    elif model == 'sentence-transformers/stsb-roberta-base-v2':
        return CrossEntropySentenceTransformer("sentence-transformers/stsb-roberta-base-v2")
    else:
        return CrossEntropySentenceTransformer(model)


def loss_function_mapping(train_loss_class): 
    if train_loss_class == 'MultipleNegativesRankingLoss':
        return MultipleNegativesRankingLoss
    elif train_loss_class == 'ContrastiveLoss':
        return ContrastiveLoss
    elif train_loss_class == 'CoSENTLoss':
        return CoSENTLoss
    elif train_loss_class == 'SentenceCrossEntropyLoss':
        return SentenceCrossEntropyLoss

def eval_function_mapping(evaluator): 
    if evaluator == 'EmbeddingSimilarity':
        return EmbeddingSimilarityEvaluator
    elif evaluator == 'CosineSimilarity':
        return CosineSimilarityEvaluator
    elif evaluator == 'SentenceCrossEntropy':
        return SentenceCrossEntropyEvaluator
    elif evaluator == 'SentenceAccuracyEvaluator':
        return SentenceAccuracyEvaluator
 

def ce_model_mapping(model):
    if model == 'bert-base-uncased':
        return CrossEncoder("bert-base-uncased", num_labels=1)
    elif model == 'roberta-base':
        return CrossEncoder("roberta-base", num_labels=1)
    elif model == 'xlnet-base-cased':
        return CrossEncoder("xlnet-base-cased", num_labels=1)
    elif model == 'stsb-roberta-base':
        return CrossEncoder("cross-encoder/stsb-roberta-base", num_labels=1)
    elif model == 'quora-roberta-base':
        return CrossEncoder("cross-encoder/quora-roberta-base", num_labels=1)
    elif model == 'mixedbread-ai/mxbai-rerank-base-v1':
        return CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1", num_labels=1)
    else:
        return CrossEncoder(model, num_labels=1)

def ce_cross_entropy_model_mapping(model, classes, subsample):
    if model == 'bert-base-uncased':
        return CrossEntropyCrossEncoder("bert-base-uncased", classes=classes, subsample=subsample, num_labels=1)
    elif model == 'roberta-base':
        return CrossEntropyCrossEncoder("roberta-base", classes=classes, subsample=subsample, num_labels=1)
    elif model == 'xlnet-base-cased':
        return CrossEntropyCrossEncoder("xlnet-base-cased", classes=classes, subsample=subsample, num_labels=1)
    elif model == 'stsb-roberta-base':
        return CrossEntropyCrossEncoder("cross-encoder/stsb-roberta-base", classes=classes, subsample=subsample, num_labels=1)
    elif model == 'quora-roberta-base':
        return CrossEntropyCrossEncoder("cross-encoder/quora-roberta-base", classes=classes, subsample=subsample, num_labels=1)
    elif model == 'mixedbread-ai/mxbai-rerank-base-v1':
        return CrossEntropyCrossEncoder("mixedbread-ai/mxbai-rerank-base-v1", classes=classes, subsample=subsample, num_labels=1)
    else:
        return CrossEntropyCrossEncoder(model, classes=classes, subsample=subsample, num_labels=1)

def ce_loss_function_mapping(train_loss_class): 
    if train_loss_class == 'CEContrastiveLoss':
        return CEContrastiveLoss
    elif train_loss_class == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss
    elif train_loss_class == 'CECrossEntropyLoss':
        return CECrossEntropyLoss

def ce_eval_function_mapping(evaluator):
    if evaluator == 'CECorrelationEvaluator':
        return CECorrelationEvaluator
    elif evaluator == 'CECosineSimilarityEvaluator':
        return CECosineSimilarityEvaluator
    elif evaluator == 'CEAccuracyEvaluator':
        return CEAccuracyEvaluator