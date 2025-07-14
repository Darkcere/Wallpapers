import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import shutil

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Folders
input_folder = "A"
output_folder = "sorted_wallpapers"
os.makedirs(output_folder, exist_ok=True)

# Category mapping based on keywords
CATEGORY_KEYWORDS = {
    "Nature": ["mountain", "forest", "tree", "river", "nature", "lake", "sunset", "flower", "grass", "desert"],
    "City": ["city", "building", "street", "architecture", "skyscraper", "urban", "bridge"],
    "Space": ["galaxy", "space", "stars", "planet", "moon", "nebula"],
    "Abstract": ["abstract", "pattern", "geometric", "shape", "texture"],
    "Anime": ["anime", "character", "manga", "girl", "boy", "cartoon"],
}

def get_category(caption):
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in caption for keyword in keywords):
            return category
    return "Other"

def classify_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True).lower()
        return caption
    except Exception as e:
        print(f"[Error] {img_path}: {e}")
        return "unknown"

def copy_to_folder(src_path, category):
    dest_dir = os.path.join(output_folder, category)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dest_path)

# Walk through all images
print("Scanning wallpapers...")
for root, _, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, filename)
            print(f"Processing: {img_path}")
            caption = classify_image(img_path)
            category = get_category(caption)
            print(f"→ {caption} → {category}")
            copy_to_folder(img_path, category)

print("\n✅ Done! Check the 'sorted_wallpapers' folder.")

