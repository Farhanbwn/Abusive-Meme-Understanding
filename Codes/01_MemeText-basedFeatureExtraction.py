#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import re
import emoji
import pickle
import torch
from tqdm import tqdm
from transformers import *
import os

# Create output directory if it doesn't exist
os.makedirs("AllFeatures", exist_ok=True)

# CONFIGURATION
DATASET_PATH = "/content/drive/MyDrive/Project_7th_sem/BanglaAbuseMeme/Dataset/BanglaAbuseMeme/BanglaAbuseMeme_annotation_with_captions.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# TEXT PREPROCESSING FUNCTIONS

puncts = [">", "+", ":", ";", "*", "'", "_", "●", "■", "•", "-", ".", "''", "``", 
          "'", "|", "​", "!", ",", "@", "?", "\u200d", "#", "(", ")", "|", "%", 
          "।", "=", "``", "&", "[", "]", "/", "”","'", "'", "'", '0', '1', 
          '2', '3', '4', '5', '6', '7', '8', '9']

def valid_bengali_letters(char):
    return char not in puncts

def get_replacement(char):
    if valid_bengali_letters(char):
        return char
    newlines = [10, 2404, 2405, 2551, 9576]
    if ord(char) in newlines: 
        return ' '
    return ' '

def get_valid_lines(line):
    copy_line = ''
    for letter in line:
        copy_line += get_replacement(letter)
    return copy_line

def preprocess_sent(sent):
    sent = re.sub(r"http\S+", " ", get_valid_lines(sent.lower()))
    sent = re.sub(r"@\S+", "@user", sent)
    sent = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "", sent)
    sent = emoji.demojize(sent)
    sent = re.sub(r"[:\*]", " ", sent)
    sent = re.sub(r"[<\*>]", " ", sent)
    sent = sent.replace("&amp;", " ")
    sent = sent.replace("ðŸ¤§", " ")
    sent = sent.replace("\n", " ")
    sent = sent.replace("ðŸ˜¡", " ")
    return sent.strip()

# LOAD DATASET

print("Loading dataset...")
try:
    allData = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded successfully! Shape: {allData.shape}")
    print(f"Columns: {allData.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(allData.head())
except FileNotFoundError:
    print(f"ERROR: Dataset file '{DATASET_PATH}' not found!")
    print("Please ensure the CSV file is in the same directory as this script.")
    exit(1)

# FEATURE EXTRACTION FUNCTION

def extract_embeddings(model, tokenizer, data, model_name, normalize_fn=None, use_cuda=True):
    embeddings = {}
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"\nExtracting embeddings using {model_name}...")
    
    for index, row in tqdm(allData.iterrows(), total=allData.shape[0]):
        try:
            new_sentence = preprocess_sent(str(row['caption']))
            
            if normalize_fn:
                new_sentence = normalize_fn(new_sentence)
            
            encoded_input = tokenizer(new_sentence, return_tensors='pt', 
                                     truncation=True, max_length=512)
            
            with torch.no_grad():
                if use_cuda and torch.cuda.is_available():
                    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
                
                output = model(**encoded_input)
                
                if hasattr(output, 'last_hidden_state'):
                    embedding = output.last_hidden_state[0][0].cpu().numpy()
                else:
                    embedding = output[0][0][0].cpu().numpy()
                
                embeddings[row['Ids']] = embedding
                
                del output
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"Error processing ID {row['Ids']}: {str(e)}")
            continue
    
    output_path = f"AllFeatures/{model_name}.p"
    with open(output_path, "wb") as fp:
        pickle.dump(embeddings, fp)
    
    print(f"Saved {len(embeddings)} embeddings to {output_path}")
    return embeddings

# 1. BANGLA BERT EMBEDDINGS

print("\n" + "="*50)
print("EXTRACTING BANGLA BERT EMBEDDINGS")
print("="*50)

try:
    from normalizer import normalize
    
    bangla_model = ElectraModel.from_pretrained("csebuetnlp/banglabert")
    bangla_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
    
    banglaBERTEmbedding = extract_embeddings(
        bangla_model, 
        bangla_tokenizer, 
        allData, 
        "banglaBERTEmbedding",
        normalize_fn=normalize,
        use_cuda=(DEVICE == "cuda")
    )
    
    del bangla_model, bangla_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except ImportError:
    print("WARNING: 'normalizer' package not found.")
    print("Install it with: pip install git+https://github.com/csebuetnlp/normalizer")
    print("Skipping BanglaBERT embeddings...")
except Exception as e:
    print(f"Error with BanglaBERT: {str(e)}")

# 2. M-BERT EMBEDDINGS

print("\n" + "="*50)
print("EXTRACTING M-BERT EMBEDDINGS")
print("="*50)

try:
    from transformers import BertTokenizer, BertModel
    
    mbert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    mbert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
    
    mBERTEmbedding = extract_embeddings(
        mbert_model, 
        mbert_tokenizer, 
        allData, 
        "mBERTEmbedding_bn_memes",
        use_cuda=(DEVICE == "cuda")
    )
    
    del mbert_model, mbert_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"Error with m-BERT: {str(e)}")

# 3. MuRIL EMBEDDINGS

print("\n" + "="*50)
print("EXTRACTING MuRIL EMBEDDINGS")
print("="*50)

try:
    muril_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    muril_model = BertModel.from_pretrained("google/muril-base-cased")
    
    murilBERTEmbedding = extract_embeddings(
        muril_model, 
        muril_tokenizer, 
        allData, 
        "MuRILEmbedding_bn_memes",
        use_cuda=(DEVICE == "cuda")
    )
    
    del muril_model, muril_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"Error with MuRIL: {str(e)}")

# 4. XLM-ROBERTA EMBEDDINGS

print("\n" + "="*50)
print("EXTRACTING XLM-ROBERTA EMBEDDINGS")
print("="*50)

try:
    xlm_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    xlm_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
    
    xlmBERTEmbedding = extract_embeddings(
        xlm_model, 
        xlm_tokenizer, 
        allData, 
        "xlmBERTEmbedding_bn_memes",
        use_cuda=(DEVICE == "cuda")
    )
    
    del xlm_model, xlm_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"Error with XLM-RoBERTa: {str(e)}")

print("\n" + "="*50)
print("FEATURE EXTRACTION COMPLETE!")
print("="*50)
print("\nAll embeddings saved in 'AllFeatures/' directory")