import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ===============================
# Resolve paths
# ===============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.model import get_model

# ===============================
# Custom Dataset for Unlabeled Images
# ===============================
class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, os.path.basename(img_path)

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
    # Config
    MODEL_NAME = "resnet50"
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "outputs", "checkpoints", f"{MODEL_NAME}_final.pth")
    IMAGE_DIR = os.path.join(BASE_DIR, "data", "unseen_images\ISIC_2020_Test_Input")  # <<-- Change this to your folder
    OUTPUT_CSV = os.path.join(BASE_DIR, "outputs", "inference_predictions.csv")
    BATCH_SIZE = 16
    NUM_WORKERS = 2

    # Find all images
    image_paths = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR) if fname.lower().endswith(".jpg")]
    print(f"✅ Found {len(image_paths)} images to predict.")

    # Transforms
    transforms = get_transforms()

    # Dataset and DataLoader
    dataset = InferenceDataset(image_paths, transform=transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # Model
    model = get_model(model_name=MODEL_NAME).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"✅ Loaded model weights from {CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")

    model.eval()

    # Inference
    results = []
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for fname, prob, pred in zip(filenames, probs[:,1].cpu().numpy(), preds.cpu().numpy()):
                results.append({
                    "image_name": fname,
                    "prob_melanoma": prob,
                    "predicted_label": pred
                })

    # Save CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Predictions saved to {OUTPUT_CSV}")
