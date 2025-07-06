import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===============================
# Add project root to sys.path
# ===============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Imports
from src.dataset import ISICDataset
from src.model import get_model
from src.utils import (
    calculate_metrics,
    plot_roc_curve,
    plot_confusion_matrix
)

# ===============================
# Transforms
# ===============================
def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
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
    MODEL_NAME = "resnet50"
    SAMPLE_FRACTION = 0.2   # Change to 1.0 to evaluate all data
    BATCH_SIZE = 16
    NUM_WORKERS = 2

    DATA_CSV = os.path.join(BASE_DIR, "data", "train.csv")
    IMAGE_DIR = os.path.join(BASE_DIR, "data", "train_images")
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "outputs", "checkpoints", f"{MODEL_NAME}_final.pth")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------
    # Load metadata
    # -------------------------------
    df = pd.read_csv(DATA_CSV)
    df_eval = df.sample(frac=SAMPLE_FRACTION, random_state=42).reset_index(drop=True)
    print(f"✅ Loaded {len(df_eval)} samples for evaluation.")

    # -------------------------------
    # Dataset and DataLoader
    # -------------------------------
    transforms = get_transforms()

    dataset = ISICDataset(
        dataframe=df_eval,
        image_dir=IMAGE_DIR,
        transform=transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"✅ Loaded model weights from {CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")

    model.eval()

    # -------------------------------
    # Inference
    # -------------------------------
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

    # -------------------------------
    # Metrics
    # -------------------------------
    metrics = calculate_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_prob=all_probs
    )

    # -------------------------------
    # Save predictions CSV
    # -------------------------------
    pred_df = pd.DataFrame({
        "y_true": all_labels,
        "y_pred": all_preds,
        "prob_melanoma": all_probs
    })
    csv_path = os.path.join(OUTPUT_DIR, "predictions_report.csv")
    pred_df.to_csv(csv_path, index=False)
    print(f"✅ Saved predictions CSV: {csv_path}")

    # -------------------------------
    # Save ROC and Confusion Matrix
    # -------------------------------
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve_report.png")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_report.png")

    plot_roc_curve(
        y_true=all_labels,
        y_prob=all_probs,
        save_path=roc_path
    )
    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        save_path=cm_path
    )

    print("✅ Reports generated and saved in 'outputs/'.")
