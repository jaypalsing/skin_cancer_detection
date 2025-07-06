import os
import cv2
import pandas as pd
from tqdm import tqdm

def check_images(df, image_dir):
    """
    Checks if all images are readable and reports missing or corrupt files.
    """
    missing = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        img_id = row["isic_id"]
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            missing.append(img_path)
            continue
        img = cv2.imread(img_path)
        if img is None:
            missing.append(img_path)
    return missing

def resize_and_save(df, image_dir, output_dir, size=(224, 224)):
    """
    Resizes images and saves them to a new directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Resizing images"):
        img_id = row["isic_id"]
        input_path = os.path.join(image_dir, f"{img_id}.jpg")
        output_path = os.path.join(output_dir, f"{img_id}.jpg")
        img = cv2.imread(input_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, size)
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, resized)

if __name__ == "__main__":
    # ===============================
    # Resolve base directory
    # ===============================
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # ===============================
    # Config
    # ===============================
    DATA_CSV = os.path.join(BASE_DIR, "data", "train.csv")
    IMAGE_DIR = os.path.join(BASE_DIR, "data", "train_images")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "preprocessed_images")
    RESIZE_IMAGES = False  # Change to True if you want resized copies

    # ===============================
    # Load metadata
    # ===============================
    df = pd.read_csv(DATA_CSV)

    # ===============================
    # Check all images
    # ===============================
    missing = check_images(df, IMAGE_DIR)

    if missing:
        print("⚠️ Missing or unreadable images:")
        for m in missing:
            print(m)
    else:
        print("✅ All images are present and readable.")

    # ===============================
    # Optional resizing
    # ===============================
    if RESIZE_IMAGES:
        resize_and_save(df, IMAGE_DIR, OUTPUT_DIR)
        print(f"✅ Images resized and saved to {OUTPUT_DIR}")
