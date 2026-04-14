#!/usr/bin/env python
# coding: utf-8

"""
Multimodal Meme Classification - Concatenation Approach
Combines text and image features by concatenating them together
"""

import torch
import json
import random
import time
import datetime
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import *
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# CONFIGURATION
FOLD_FILE = "/content/drive/MyDrive/Project_7th_sem/BanglaAbuseMeme/Codes/FoldWiseDetail.p"  # Use the file we created
OUTPUT_FOLDER = "/content/drive/MyDrive/Project_7th_sem/BanglaAbuseMeme/Codes/ResultFolder"
FEATURES_FOLDER = "/content/drive/MyDrive/Project_7th_sem/BanglaAbuseMeme/Codes/AllFeatures"
N_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# GPU Setup
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print(f'Using GPU: {torch.cuda.get_device_name()}')
else:
    print('No GPU available, using CPU instead.')
    device = torch.device("cpu")

# Set random seeds
def fix_the_random(seed_val=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

fix_the_random(2021)

# Evaluation Metrics
def evalMetric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mf1Score = f1_score(y_true, y_pred, average='macro')
    f1Score = f1_score(y_true, y_pred)
    
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        area_under_c = auc(fpr, tpr)
    except:
        area_under_c = 0.0
    
    recallScore = recall_score(y_true, y_pred)
    precisionScore = precision_score(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        'mF1Score': mf1Score,
        'f1Score': f1Score,
        'auc': area_under_c,
        'precision': precisionScore,
        'recall': recallScore
    }

# Helper Functions
def getFeaturesandLabel(X, y, text_features, image_features):
    """Extract text and image features for given sample IDs"""
    X_text_data = []
    X_image_data = []
    
    for i in X:
        # Handle CLIP features (dict with 'text' and 'image' keys)
        try:
            if isinstance(text_features[i], dict) and 'text' in text_features[i]:
                X_text_data.append(text_features[i]['text'])
            else:
                X_text_data.append(text_features[i])
        except KeyError:
            print(f"Warning: {i} not found in text features")
            
        try:
            if isinstance(image_features[i], dict) and 'image' in image_features[i]:
                X_image_data.append(image_features[i]['image'])
            else:
                X_image_data.append(image_features[i])
        except KeyError:
            print(f"Warning: {i} not found in image features")
    
    X_text_data = torch.tensor(X_text_data, dtype=torch.float32)
    X_image_data = torch.tensor(X_image_data, dtype=torch.float32)
    y_data = torch.tensor(y, dtype=torch.long)
    
    return X_text_data, X_image_data, y_data

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Multimodal Concatenation Model
class Multimodal_Concat_Model(nn.Module):
    def __init__(self, text_size, image_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()
        combined_size = text_size + image_size
        self.network = nn.Sequential(
            nn.Linear(combined_size, fc1_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc2_hidden, output_size),
        )
    
    def forward(self, x_text, x_image):
        # Concatenate text and image features
        combined = torch.cat((x_text, x_image), dim=1)
        return self.network(combined)

# Get predictions
def getPerformanceOfLoader(model, dataloader, id_list):
    model.eval()
    predictions, true_labels = [], []
    
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_text, b_image, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_text, b_image)
        
        logits = outputs.max(1, keepdim=True)[1]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.extend(logits)
        true_labels.extend(label_ids)
    
    pred = [i[0] for i in predictions]
    df = pd.DataFrame()
    df['Ids'] = id_list
    df['true'] = true_labels
    df['target'] = pred
    return df

# Training Function
def trainModel(model, train_dataloader, validation_dataloader, test_dataloader,
               val_ids, test_ids):
    model.to(device)
    
    bestValMF1 = 0
    besttest_df = None
    bestEpochs = -1
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch_i in range(EPOCHS):
        print(f'\n======== Epoch {epoch_i + 1} / {EPOCHS} ========')
        print('Training...')
        
        t0 = time.time()
        total_loss = 0
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            b_text = batch[0].to(device)
            b_image = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad()
            outputs = model(b_text, b_image)
            
            loss = F.cross_entropy(outputs, b_labels,
                                  weight=torch.FloatTensor([0.374, 0.626]).to(device))
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {format_time(time.time() - t0)}")
        
        # Validation
        print("\nRunning Validation...")
        t0 = time.time()
        
        val_df = getPerformanceOfLoader(model, validation_dataloader, val_ids)
        origValValue = list(val_df['true'])
        preValValue = list(val_df['target'])
        
        valMf1Score = evalMetric(origValValue, preValValue)['mF1Score']
        tempValAcc = evalMetric(origValValue, preValValue)['accuracy']
        
        if valMf1Score > bestValMF1:
            bestEpochs = epoch_i
            bestValMF1 = valMf1Score
            besttest_df = getPerformanceOfLoader(model, test_dataloader, test_ids)
        
        print(f"  Accuracy: {tempValAcc:.2f}")
        print(f"  Macro F1: {valMf1Score:.2f}")
        print(f"  Validation took: {format_time(time.time() - t0)}")
    
    print(f"\nBest epoch: {bestEpochs}")
    print("Training complete!")
    return besttest_df

# Load fold data
print("\n" + "="*60)
print("LOADING FOLD DATA")
print("="*60)

if not os.path.exists(FOLD_FILE):
    print(f"ERROR: {FOLD_FILE} not found!")
    print("Please run script 03 first to create fold splits.")
    exit(1)

with open(FOLD_FILE, 'rb') as fp:
    allDataAnnotation = pickle.load(fp)

print(f"Loaded fold data for {len(allDataAnnotation)} folds\n")

# Feature file mapping
modelNameMapping = {
    "banglaBERT": os.path.join(FEATURES_FOLDER, 'banglaBERTEmbedding.p'),
    "mBERT": os.path.join(FEATURES_FOLDER, 'mBERTEmbedding_bn_memes.p'),
    "MuRIL": os.path.join(FEATURES_FOLDER, 'MuRILEmbedding_bn_memes.p'),
    "XLM-R": os.path.join(FEATURES_FOLDER, 'xlmBERTEmbedding_bn_memes.p'),
    "ResNet152": os.path.join(FEATURES_FOLDER, 'resnet152_features.p'),
    "ViT": os.path.join(FEATURES_FOLDER, 'vit_features.p'),
    "VGG16": os.path.join(FEATURES_FOLDER, 'vgg16_features.p'),
    "VAN": os.path.join(FEATURES_FOLDER, 'van_features.p'),
    "CLIP": os.path.join(FEATURES_FOLDER, 'clip_features.p'),
}

metricType = ['accuracy', 'mF1Score', 'f1Score', 'auc', 'precision', 'recall']
allFolds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

# Define combinations to test
# Best performing models from unimodal: VGG16 (image), text models
text_models = ["MuRIL", "XLM-R"]
image_models = ["VGG16", "ResNet152", "ViT"]

print("\n" + "="*60)
print("MULTIMODAL TRAINING - CONCATENATION APPROACH")
print("="*60)
print(f"Text models: {text_models}")
print(f"Image models: {image_models}")
print(f"Total combinations: {len(text_models) * len(image_models)}\n")

outputFp = open("multimodal_concat_results.txt", 'w')

for text_mod in text_models:
    for image_mod in image_models:
        modelName = f"{text_mod}_{image_mod}_concat"
        
        print(f"\n{'='*60}")
        print(f"Training: {modelName}")
        print(f"{'='*60}")
        
        # Check if feature files exist
        if text_mod not in modelNameMapping or not os.path.exists(modelNameMapping[text_mod]):
            print(f"WARNING: Text features '{text_mod}' not found. Skipping...")
            continue
        
        if image_mod not in modelNameMapping or not os.path.exists(modelNameMapping[image_mod]):
            print(f"WARNING: Image features '{image_mod}' not found. Skipping...")
            continue
        
        # Load features
        with open(modelNameMapping[text_mod], 'rb') as fp:
            inputTextFeatures = pickle.load(fp)
        
        with open(modelNameMapping[image_mod], 'rb') as fp:
            inputImageFeatures = pickle.load(fp)
        
        # Get feature dimensions
        sample_key = list(inputTextFeatures.keys())[0]
        
        # Handle CLIP special case
        if text_mod == "CLIP":
            input_size_text = len(inputTextFeatures[sample_key]['text'])
        else:
            input_size_text = len(inputTextFeatures[sample_key])
        
        if image_mod == "CLIP":
            input_size_image = len(inputImageFeatures[sample_key]['image'])
        else:
            input_size_image = len(inputImageFeatures[sample_key])
        
        print(f"Text feature size: {input_size_text}")
        print(f"Image feature size: {input_size_image}")
        print(f"Combined size: {input_size_text + input_size_image}")
        
        fc1_hidden, fc2_hidden = 256, 256
        finalOutputAccrossFold = {}
        
        for fold in allFolds:
            print(f"\n--- {fold} ---")
            
            train_list, train_label = allDataAnnotation[fold]['train']
            val_list, val_label = allDataAnnotation[fold]['val']
            test_list, test_label = allDataAnnotation[fold]['test']
            
            X_train_text, X_train_image, y_train = getFeaturesandLabel(
                train_list, train_label, inputTextFeatures, inputImageFeatures)
            X_val_text, X_val_image, y_val = getFeaturesandLabel(
                val_list, val_label, inputTextFeatures, inputImageFeatures)
            X_test_text, X_test_image, y_test = getFeaturesandLabel(
                test_list, test_label, inputTextFeatures, inputImageFeatures)
            
            train_data = TensorDataset(X_train_text, X_train_image, y_train)
            val_data = TensorDataset(X_val_text, X_val_image, y_val)
            test_data = TensorDataset(X_test_text, X_test_image, y_test)
            
            train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data),
                                         batch_size=BATCH_SIZE)
            val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data),
                                        batch_size=BATCH_SIZE)
            test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                                         batch_size=BATCH_SIZE)
            
            model = Multimodal_Concat_Model(input_size_text, input_size_image,
                                           fc1_hidden, fc2_hidden, 2).to(device)
            
            besttest_df = trainModel(model, train_dataloader, val_dataloader,
                                    test_dataloader, val_list, test_list)
            
            finalOutputAccrossFold[fold] = besttest_df
        
        # Save fold-wise results
        with open(f"{OUTPUT_FOLDER}/{modelName}_foldwise.p", 'wb') as fp:
            pickle.dump(finalOutputAccrossFold, fp)
        
        # Calculate average metrics
        allValueDict = {}
        for fold in allFolds:
            evalObject = evalMetric(finalOutputAccrossFold[fold]['true'],
                                   finalOutputAccrossFold[fold]['target'])
            for metType in metricType:
                if metType not in allValueDict:
                    allValueDict[metType] = []
                allValueDict[metType].append(evalObject[metType])
        
        print(f"\n{modelName} - Final Results:")
        outputFp.write(f"\n{modelName}\n")
        outputFp.write("="*40 + "\n")
        
        for metric in allValueDict:
            mean_val = np.mean(allValueDict[metric])
            std_val = np.std(allValueDict[metric])
            result_str = f"{metric}: Mean={mean_val:.4f}, STD={std_val:.4f}"
            print(f"  {result_str}")
            outputFp.write(result_str + "\n")
        
        outputFp.write("\n")

outputFp.close()

print("\n" + "="*60)
print("MULTIMODAL CONCATENATION COMPLETE!")
print("="*60)
print(f"Fold-wise details saved in: {OUTPUT_FOLDER}/")
print("="*60)