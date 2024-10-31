from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import LlamaForCausalLM, LlamaTokenizer

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
    
def model_mapping(model_name):
    if model_name == 'meta-llama/Llama-2-7b-hf':
        return LlamaForCausalLM.from_pretrained(model_name)
   
    
def tokenizer_mapping(model_name):
    if model_name == 'meta-llama/Llama-2-7b-hf':
        return LlamaTokenizer.from_pretrained(model_name)
    

