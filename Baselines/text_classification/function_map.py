from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification

def model_class_mapping(model):
    if model == 'BertForSequenceClassification':
        return BertForSequenceClassification
    elif model == 'RobertaForSequenceClassification':
        return RobertaForSequenceClassification
    elif model == 'XLNetForSequenceClassification':
        return XLNetForSequenceClassification


def tokenizer_class_mapping(tokenizer):
    if tokenizer == 'BertTokenizer':
        return BertTokenizer
    elif tokenizer == 'RobertaTokenizer':
        return RobertaTokenizer
    elif tokenizer == 'XLNetTokenizer':
        return XLNetTokenizer

