#!/usr/bin/env python
# coding: utf-8

"""
Unimodal Meme Classification with K-Fold Cross-Validation
Trains classification models on individual features (text-only or image-only)
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# CONFIGURATION
DATASET_CSV = "/content/drive/MyDrive/Project_7th_sem/BanglaAbuseMeme/Dataset/BanglaAbuseMeme/BanglaAbuseMeme_annotation_with_captions.csv"
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

# Set random seeds for reproducibility
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
def getFeaturesandLabel(X, y, text_features):
    X_text_data = []
    for i in X:
        if i in text_features:
            X_text_data.append(text_features[i])
        else:
            print(f"Warning: {i} not found in features")
    X_text_data = torch.tensor(X_text_data, dtype=torch.float32)
    y_data = torch.tensor(y, dtype=torch.long)
    return X_text_data, y_data

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Simple Neural Network Model
class Uni_Model(nn.Module):
    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc2_hidden, output_size),
        )
    
    def forward(self, xb):
        return self.network(xb)

# Get predictions for a dataloader
def getPerformanceOfLoader(model, dataloader, id_list):
    model.eval()
    predictions, true_labels = [], []
    
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_features, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_features)
        
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
        # print(f'\n======== Epoch {epoch_i + 1} / {EPOCHS} ========')
        # print('Training...')
        
        t0 = time.time()
        total_loss = 0
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            b_features = batch[0].to(device)
            b_labels = batch[1].to(device)
            
            model.zero_grad()
            outputs = model(b_features)
            
            # Class weights for imbalanced data
            loss = F.cross_entropy(outputs, b_labels, 
                                  weight=torch.FloatTensor([0.374, 0.626]).to(device))
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {format_time(time.time() - t0)}")
        
        # Validation
        # print("\nRunning Validation...")
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
        
        # print(f"  Accuracy: {tempValAcc:.2f}")
        # print(f"  Macro F1: {valMf1Score:.2f}")
        # print(f"  Validation took: {format_time(time.time() - t0)}")
    
    print(f"\nBest epoch: {bestEpochs}")
    print("Training complete!")
    return besttest_df

# Load Dataset and Create Folds
print("\n" + "="*60)
print("LOADING DATASET AND CREATING FOLDS")
print("="*60)

allData = pd.read_csv(DATASET_CSV)
print(f"Loaded {len(allData)} samples\n")

# Check if label column exists
label_columns = ['abuse', 'abusive', 'label', 'target']
label_col = None
for col in label_columns:
    if col in allData.columns:
        label_col = col
        break

if label_col is None:
    print("ERROR: No label column found!")
    print(f"Available columns: {allData.columns.tolist()}")
    print("\nPlease specify which column contains the labels (Abusive/Non-abusive)")
    exit(1)

print(f"Using '{label_col}' as label column")

# Convert labels to binary (0/1)
if allData[label_col].dtype == 'object':
    # If labels are text (e.g., "Abusive", "Non-abusive")
    unique_labels = allData[label_col].unique()
    print(f"Unique labels: {unique_labels}")
    
    # Map to binary
    label_mapping = {}
    if 'abusive' in str(unique_labels[0]).lower():
        label_mapping = {unique_labels[0]: 1, unique_labels[1]: 0}
    else:
        label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
    
    allData['binary_label'] = allData[label_col].map(label_mapping)
else:
    allData['binary_label'] = allData[label_col]

print(f"Label distribution:")
print(allData['binary_label'].value_counts())

# Create K-Fold splits
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=2021)
allDataAnnotation = {}

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(allData['Ids'], allData['binary_label'])):
    fold_name = f'fold{fold_idx + 1}'
    
    # Further split train into train and validation
    train_val_data = allData.iloc[train_val_idx]
    train_idx, val_idx = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=2021).split(
        train_val_data['Ids'], train_val_data['binary_label']))[0]
    
    train_data = train_val_data.iloc[train_idx]
    val_data = train_val_data.iloc[val_idx]
    test_data = allData.iloc[test_idx]
    
    allDataAnnotation[fold_name] = {
        'train': (list(train_data['Ids']), list(train_data['binary_label'])),
        'val': (list(val_data['Ids']), list(val_data['binary_label'])),
        'test': (list(test_data['Ids']), list(test_data['binary_label']))
    }
    
    print(f"{fold_name}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# Save fold information
with open('FoldWiseDetail.p', 'wb') as fp:
    pickle.dump(allDataAnnotation, fp)
print("\n✓ Fold splits saved to 'FoldWiseDetail.p'")

# Feature Files Mapping
modelNameMapping = {
    "banglaBERT": os.path.join(FEATURES_FOLDER, 'banglaBERTEmbedding.p'),
    "mBERT": os.path.join(FEATURES_FOLDER, 'mBERTEmbedding_bn_memes.p'),
    "MuRIL": os.path.join(FEATURES_FOLDER, 'MuRILEmbedding_bn_memes.p'),
    "XLM-R": os.path.join(FEATURES_FOLDER, 'xlmBERTEmbedding_bn_memes.p'),
    "ResNet152": os.path.join(FEATURES_FOLDER, 'resnet152_features.p'),
    "ViT": os.path.join(FEATURES_FOLDER, 'vit_features.p'),
    "VGG16": os.path.join(FEATURES_FOLDER, 'vgg16_features.p'),
    "VAN": os.path.join(FEATURES_FOLDER, 'van_features.p'),
}

metricType = ['accuracy', 'mF1Score', 'f1Score', 'auc', 'precision', 'recall']
allFolds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']

# Main Training Loop
outputFp = open("unimodal_results.txt", 'w')

print("\n" + "="*60)
print("STARTING UNIMODAL TRAINING")
print("="*60)

for modelName in modelNameMapping:
    print(f"\n{'='*60}")
    print(f"Training with: {modelName}")
    print(f"{'='*60}")
    
    feature_path = modelNameMapping[modelName]
    
    if not os.path.exists(feature_path):
        print(f"WARNING: Feature file not found: {feature_path}")
        print("Skipping this model...\n")
        continue
    
    with open(feature_path, 'rb') as fp:
        inputDataFeatures = pickle.load(fp)
    
    # Get feature dimension
    sample_key = list(inputDataFeatures.keys())[0]
    input_size = len(inputDataFeatures[sample_key])
    print(f"Feature dimension: {input_size}")
    
    fc1_hidden, fc2_hidden = 256, 256
    finalOutputAccrossFold = {}
    
    for fold in allFolds:
        print(f"\n--- {fold} ---")
        
        train_list, train_label = allDataAnnotation[fold]['train']
        val_list, val_label = allDataAnnotation[fold]['val']
        test_list, test_label = allDataAnnotation[fold]['test']
        
        X_train, y_train = getFeaturesandLabel(train_list, train_label, inputDataFeatures)
        X_val, y_val = getFeaturesandLabel(val_list, val_label, inputDataFeatures)
        X_test, y_test = getFeaturesandLabel(test_list, test_label, inputDataFeatures)
        
        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(X_val, y_val)
        test_data = TensorDataset(X_test, y_test)
        
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), 
                                     batch_size=BATCH_SIZE)
        val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), 
                                    batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), 
                                     batch_size=BATCH_SIZE)
        
        model = Uni_Model(input_size, fc1_hidden, fc2_hidden, 2).to(device)
        
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
print("UNIMODAL CLASSIFICATION COMPLETE!")
print("="*60)
print(f"Results saved to: unimodal_results.txt")
print(f"Fold-wise details saved in: {OUTPUT_FOLDER}/")