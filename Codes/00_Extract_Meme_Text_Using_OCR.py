#!/usr/bin/env python
# coding: utf-8

"""
Step 0: Extract Text from Meme Images using EasyOCR
This script must be run BEFORE the text feature extraction
"""

import pandas as pd
import easyocr
import os
from tqdm import tqdm

# CONFIGURATION
CSV_PATH = "/content/drive/MyDrive/Project_7th_sem/BanglaAbuseMeme/Dataset/BanglaAbuseMeme/BanglaAbuseMeme_annotation.csv"
IMAGE_FOLDER = "/content/drive/MyDrive/Project_7th_sem/BanglaAbuseMeme/Dataset/BanglaAbuseMeme/Images"  # Change this to your actual image folder path
OUTPUT_CSV = "/content/drive/MyDrive/Project_7th_sem/BanglaAbuseMeme/Dataset/BanglaAbuseMeme/BanglaAbuseMeme_annotation_with_captions.csv"

print("="*60)
print("MEME TEXT EXTRACTION USING EasyOCR")
print("="*60)

# Load the dataset
print(f"\nLoading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Initialize EasyOCR reader for Bengali
print("Initializing EasyOCR reader for Bengali...")
print("This may take a few minutes on first run (downloading models)...")
reader = easyocr.Reader(['bn', 'en'], gpu=True)  # Bengali and English
print("EasyOCR reader initialized!\n")

# Check if image folder exists
if not os.path.exists(IMAGE_FOLDER):
    print(f"ERROR: Image folder '{IMAGE_FOLDER}' not found!")
    print("Please update IMAGE_FOLDER path in the script.")
    exit(1)

# Extract text from each image
captions = []
errors = []

print(f"Extracting text from {len(df)} memes...")
print(f"Looking for images in: {IMAGE_FOLDER}\n")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    image_id = row['Ids']
    image_path = os.path.join(IMAGE_FOLDER, image_id)
    
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            captions.append("")
            errors.append(f"Image not found: {image_path}")
            continue
        
        # Extract text using OCR
        result = reader.readtext(image_path)
        
        # Combine all detected text
        extracted_text = " ".join([text[1] for text in result])
        
        # If no text found, use empty string
        if not extracted_text.strip():
            extracted_text = ""
        
        captions.append(extracted_text)
        
    except Exception as e:
        captions.append("")
        errors.append(f"Error processing {image_id}: {str(e)}")

# Add captions to dataframe
df['caption'] = captions

# Show statistics
print("\n" + "="*60)
print("EXTRACTION STATISTICS")
print("="*60)
print(f"Total images processed: {len(df)}")
print(f"Images with text detected: {sum(1 for c in captions if c.strip())}")
print(f"Images with no text: {sum(1 for c in captions if not c.strip())}")
print(f"Errors encountered: {len(errors)}")

if errors:
    print("\nErrors (first 10):")
    for error in errors[:10]:
        print(f"  - {error}")

# Show sample extracted captions
print("\n" + "="*60)
print("SAMPLE EXTRACTED CAPTIONS")
print("="*60)
sample_df = df[df['caption'].str.len() > 0].head(5)
for idx, row in sample_df.iterrows():
    print(f"\nImage: {row['Ids']}")
    print(f"Caption: {row['caption'][:100]}...")  # First 100 chars

# Save the updated CSV
print(f"\n{'='*60}")
print(f"Saving updated dataset to: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Saved successfully!")
print("="*60)
