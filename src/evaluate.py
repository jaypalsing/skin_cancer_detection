import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    recall_score
)
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Automatically get the project root (one folder up from 'src')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define paths
DATA_CSV = os.path.join(BASE_DIR, "data", "train.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "train_images")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "outputs", "checkpoints", "resnet50_final.pth")

# Imports from src
from src.dataset import ISICDataset
from src.model import get_model

def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

def evaluate_model(model, dataloader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    roc_auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds, pos_label=1)
    specificity = recall_score(all_labels, all_preds, pos_label=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"ROC AUC       : {roc_auc:.4f}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Sensitivity   : {sensitivity:.4f}")
    print(f"Specificity   : {specificity:.4f}")
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    # Load metadata
    df = pd.read_csv(DATA_CSV)

    # Optionally split into validation subset
    df_eval = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

    # Transforms
    transforms = get_transforms()

    # Dataset and DataLoader
    dataset = ISICDataset(
        dataframe=df_eval,
        image_dir=IMAGE_DIR,
        transform=transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model().to(device)

    # Load trained weights
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded model weights from {CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # Evaluate
    evaluate_model(model, dataloader, device)
