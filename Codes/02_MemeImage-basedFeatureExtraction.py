#!/usr/bin/env python
# coding: utf-8

"""
Meme Image Feature Extraction Script
Extracts visual features from meme images using multiple pre-trained models:
1. ResNet-152
2. Vision Transformer (ViT)
3. VGG16
4. CLIP (both image and text features)
5. Visual Attention Network (VAN)
"""

import pandas as pd
import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn

# CONFIGURATION
DATASET_CSV = "/content/drive/MyDrive/Project_7th_sem/BanglaAbuseMeme/Dataset/BanglaAbuseMeme/BanglaAbuseMeme_annotation_with_captions.csv"  # CSV with caption column
IMAGE_FOLDER = "/content/drive/MyDrive/Project_7th_sem/BanglaAbuseMeme/Dataset/BanglaAbuseMeme/Images"  # Folder containing meme images
OUTPUT_FOLDER = "AllFeatures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print("MEME IMAGE FEATURE EXTRACTION")
print("="*60)
print(f"Using device: {DEVICE}")
print(f"Image folder: {IMAGE_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}\n")

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load dataset
print("Loading dataset...")
allData = pd.read_csv(DATASET_CSV)
print(f"Loaded {len(allData)} images\n")

# HELPER FUNCTION: Load and preprocess image
def get_image(image_path):
    """Load image and convert to RGB"""
    try:
        image = Image.open(image_path).convert('RGB')
        # Handle grayscale images
        if len(np.array(image).shape) == 2:
            image = np.array(image)
            image = np.stack([image, image, image], axis=2)
            image = Image.fromarray(image)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

# TEXT PREPROCESSING FUNCTIONS (for CLIP)
import re
import emoji

puncts = [">", "+", ":", ";", "*", "'", "_", "●", "■", "•", "-", ".", "''", "``", 
          "'", "|", "​", "!", ",", "@", "?", "\u200d", "#", "(", ")", "|", "%", 
          "।", "=", "``", "&", "[", "]", "/", "”", "'", "'", "'", '0', '1', 
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


# =============================================================================
# 1. EXTRACT RESNET-152 FEATURES
# =============================================================================
print("\n" + "="*60)
print("EXTRACTING RESNET-152 FEATURES")
print("="*60)

try:
    # Load ResNet-152
    resnet152 = models.resnet152(pretrained=True)
    resnet152 = torch.nn.Sequential(*list(resnet152.children())[:-1])
    resnet152.to(DEVICE)
    resnet152.eval()
    
    # Transform for ResNet
    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    resnet_features = {}
    
    for idx, row in tqdm(allData.iterrows(), total=len(allData)):
        image_id = row['Ids']
        image_path = os.path.join(IMAGE_FOLDER, image_id)
        
        image = get_image(image_path)
        if image is None:
            continue
        
        try:
            image_tensor = resnet_transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                features = resnet152(image_tensor)
            
            features = torch.flatten(features).cpu().numpy()
            resnet_features[image_id] = features
            
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue
    
    # Save ResNet features
    output_path = os.path.join(OUTPUT_FOLDER, "resnet152_features.p")
    with open(output_path, 'wb') as fp:
        pickle.dump(resnet_features, fp)
    
    print(f"✓ Saved {len(resnet_features)} ResNet-152 features")
    
    del resnet152
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"Error with ResNet-152: {str(e)}")


# =============================================================================
# 2. EXTRACT VISION TRANSFORMER (ViT) FEATURES
# =============================================================================
print("\n" + "="*60)
print("EXTRACTING VISION TRANSFORMER (ViT) FEATURES")
print("="*60)

try:
    from transformers import ViTFeatureExtractor, ViTModel
    
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_model.to(DEVICE)
    vit_model.eval()
    
    vit_features = {}
    
    for idx, row in tqdm(allData.iterrows(), total=len(allData)):
        image_id = row['Ids']
        image_path = os.path.join(IMAGE_FOLDER, image_id)
        
        image = get_image(image_path)
        if image is None:
            continue
        
        try:
            inputs = feature_extractor(image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = vit_model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                vit_features[image_id] = last_hidden_states[0][0].cpu().numpy()
                
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue
    
    # Save ViT features
    output_path = os.path.join(OUTPUT_FOLDER, "vit_features.p")
    with open(output_path, 'wb') as fp:
        pickle.dump(vit_features, fp)
    
    print(f"✓ Saved {len(vit_features)} ViT features")
    
    del vit_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"Error with ViT: {str(e)}")


# =============================================================================
# 3. EXTRACT VGG16 FEATURES
# =============================================================================
print("\n" + "="*60)
print("EXTRACTING VGG16 FEATURES")
print("="*60)

try:
    class FeatureExtractor(nn.Module):
        def __init__(self, model):
            super(FeatureExtractor, self).__init__()
            self.features = nn.Sequential(*list(model.features))
            self.pooling = model.avgpool
            self.flatten = nn.Flatten()
            self.fc = model.classifier[0]
        
        def forward(self, x):
            out = self.features(x)
            out = self.pooling(out)
            out = self.flatten(out)
            out = self.fc(out)
            return out
    
    vgg_model = models.vgg16(pretrained=True)
    vgg_extractor = FeatureExtractor(vgg_model)
    vgg_extractor.to(DEVICE)
    vgg_extractor.eval()
    
    vgg_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    vgg16_features = {}
    
    for idx, row in tqdm(allData.iterrows(), total=len(allData)):
        image_id = row['Ids']
        image_path = os.path.join(IMAGE_FOLDER, image_id)
        
        image = get_image(image_path)
        if image is None:
            continue
        
        try:
            image_tensor = vgg_transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = vgg_extractor(image_tensor)
            
            vgg16_features[image_id] = output[0].cpu().numpy()
            
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue
    
    # Save VGG16 features
    output_path = os.path.join(OUTPUT_FOLDER, "vgg16_features.p")
    with open(output_path, 'wb') as fp:
        pickle.dump(vgg16_features, fp)
    
    print(f"✓ Saved {len(vgg16_features)} VGG16 features")
    
    del vgg_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"Error with VGG16: {str(e)}")


# =============================================================================
# 4. EXTRACT CLIP FEATURES (Image + Text)
# =============================================================================
print("\n" + "="*60)
print("EXTRACTING CLIP FEATURES (Image + Text)")
print("="*60)

try:
    import clip
    
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    clip_features = {}
    
    for idx, row in tqdm(allData.iterrows(), total=len(allData)):
        image_id = row['Ids']
        image_path = os.path.join(IMAGE_FOLDER, image_id)
        
        image = get_image(image_path)
        if image is None:
            continue
        
        try:
            # Process image
            image_input = clip_preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Process text
            caption = str(row['caption']) if pd.notna(row['caption']) else ""
            text_input = clip.tokenize(preprocess_sent(caption), truncate=True).to(DEVICE)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features = clip_model.encode_text(text_input)
                
                clip_features[image_id] = {
                    'text': text_features[0].cpu().numpy(),
                    'image': image_features[0].cpu().numpy()
                }
                
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue
    
    # Save CLIP features
    output_path = os.path.join(OUTPUT_FOLDER, "clip_features.p")
    with open(output_path, 'wb') as fp:
        pickle.dump(clip_features, fp)
    
    print(f"✓ Saved {len(clip_features)} CLIP features")
    
    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"Error with CLIP: {str(e)}")
    print("You may need to install CLIP: pip install git+https://github.com/openai/CLIP.git")


# =============================================================================
# 5. EXTRACT VISUAL ATTENTION NETWORK (VAN) FEATURES
# =============================================================================
print("\n" + "="*60)
print("EXTRACTING VISUAL ATTENTION NETWORK (VAN) FEATURES")
print("="*60)

try:
    from transformers import AutoImageProcessor, VanModel
    
    van_processor = AutoImageProcessor.from_pretrained("Visual-Attention-Network/van-base")
    van_model = VanModel.from_pretrained("Visual-Attention-Network/van-base")
    van_model.to(DEVICE)
    van_model.eval()
    
    van_features = {}
    
    for idx, row in tqdm(allData.iterrows(), total=len(allData)):
        image_id = row['Ids']
        image_path = os.path.join(IMAGE_FOLDER, image_id)
        
        image = get_image(image_path)
        if image is None:
            continue
        
        try:
            inputs = van_processor(image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = van_model(**inputs)
                van_features[image_id] = outputs.pooler_output[0].cpu().numpy()
                
        except Exception as e:
            print(f"Error processing {image_id}: {str(e)}")
            continue
    
    # Save VAN features
    output_path = os.path.join(OUTPUT_FOLDER, "van_features.p")
    with open(output_path, 'wb') as fp:
        pickle.dump(van_features, fp)
    
    print(f"✓ Saved {len(van_features)} VAN features")
    
    del van_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
except Exception as e:
    print(f"Error with VAN: {str(e)}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("IMAGE FEATURE EXTRACTION COMPLETE!")
print("="*60)
print(f"\nAll features saved in '{OUTPUT_FOLDER}/' directory")
print("\n" + "="*60)