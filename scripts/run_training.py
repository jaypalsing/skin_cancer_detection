import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===============================
# Resolve paths and imports
# ===============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.dataset import ISICDataset
from src.model import get_model
from src.train import train_model

# ===============================
# Transforms
# ===============================
def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    # -------------------------------
    # Configurations
    # -------------------------------
    MODEL_NAME = "densenet121"           # Options: 'resnet50', 'efficientnet_b0', 'densenet121'
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    CLASS_WEIGHTS = (0.25, 0.75)         # (benign, melanoma)
    NUM_WORKERS = 2

    DATA_CSV = os.path.join(BASE_DIR, "data", "train.csv")
    IMAGE_DIR = os.path.join(BASE_DIR, "data", "train_images")
    SAVE_PATH = os.path.join(BASE_DIR, "outputs", "checkpoints", f"{MODEL_NAME}_final.pth")

    # -------------------------------
    # Load metadata
    # -------------------------------
    df = pd.read_csv(DATA_CSV)

    # -------------------------------
    # Transforms
    # -------------------------------
    transforms = get_transforms()

    # -------------------------------
    # Dataset and DataLoader
    # -------------------------------
    dataset = ISICDataset(
        dataframe=df,
        image_dir=IMAGE_DIR,
        transform=transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # -------------------------------
    # Device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # -------------------------------
    # Model
    # -------------------------------
    model = get_model(model_name=MODEL_NAME).to(device)
    print(f"✅ Model architecture: {MODEL_NAME}")

    # -------------------------------
    # Train
    # -------------------------------
    train_model(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        save_path=SAVE_PATH,
        class_weights=CLASS_WEIGHTS
    )
