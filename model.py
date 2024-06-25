import spacy
from spacy import displacy
import pprint
#from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# import ast
# import time
# import seaborn as sns
# import matplotlib.pyplot as plt
#import json
import torch
import transformers as ppb
#import warnings
#warnings.filterwarnings('ignore')
import numpy as np



def extract_skills_resume(text):
    nlp = spacy.load("new_model_resume")

    doc = nlp(text)


    entities = {label: [] for label in nlp.get_pipe("ner").labels}
    for ent in doc.ents:
            if ent.label_ in nlp.get_pipe("ner").labels:
                entities[ent.label_].append(ent.text)

    from nltk.tokenize import word_tokenize
    token_list = []
    for item in entities["SKILL"]:
        text = item
        tokens = word_tokenize(text)
        token_list = token_list + tokens

    token_list_resume_correct=[]
    for i in token_list:
        if i.isalpha():
            token_list_resume_correct.append(i)
    token_list_correct = list(set(token_list_resume_correct))

    return token_list_correct

def extract_skills_jd(text):
    nlp = spacy.load("new_model_jd")

    doc = nlp(text)


    entities = {label: [] for label in nlp.get_pipe("ner").labels}
    for ent in doc.ents:
            if ent.label_ in nlp.get_pipe("ner").labels:
                entities[ent.label_].append(ent.text)

    from nltk.tokenize import word_tokenize
    token_list = []
    for item in entities["SKILL"]:
        text = item
        tokens = word_tokenize(text)
        token_list = token_list + tokens

    token_list_resume_correct=[]
    for i in token_list:
        if i.isalpha():
            token_list_resume_correct.append(i)
    token_list_correct = list(set(token_list_resume_correct))

    return token_list_correct
# For BERT:
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')


# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def get_sw_embeddings(word):
   
    # Tokenize the word
    encoded_input = tokenizer(word, return_tensors='pt')
    
    # Get the token IDs and attention mask
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
    
    # Remove the [CLS] and [SEP] tokens
    token_embeddings = last_hidden_states[0][1:-1]
    
    # If the word was split into subwords, average their embeddings
    word_embedding = token_embeddings.mean(dim=0)
    
    return word_embedding.numpy()


def sw_semantic_similarity_from_bert(job,resume):
    """calculate similarity with bertbaseuncased"""
    score = []
    match_count = 0
    sim_count = 0
    for i in job:
        sim_score = []
        for j in resume:
            job_emb = get_sw_embeddings(i).reshape(1,-1)
            resume_emb = get_sw_embeddings(j).reshape(1,-1)
            sim_score.append(cosine_similarity(job_emb,resume_emb))
        if np.array(sim_score).max()>0.6:
          score.append(np.array(sim_score).max())
          if  np.array(sim_score).max() >= .98:
            match_count+=1 
          else:
            sim_count += 1
                 

    score_sum = np.array(score).sum()/len(job)

   
    return round(score_sum,3), sim_count, match_count

