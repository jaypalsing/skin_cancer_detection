import os
import cv2
import torch
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    """
    Custom Dataset for ISIC 2020 skin lesion images.
    Loads images and associated labels.
    """
    def __init__(self, dataframe, image_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with columns:
                - 'isic_id': image file names without extension
                - 'target': 0 (benign) or 1 (melanoma) OR
                - 'diagnosis': string labels (benign, melanoma)
            image_dir (str): Path to directory with images
            transform: Albumentations transforms (optional)
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image name
        if "isic_id" in self.df.columns:
            image_id = self.df.loc[idx, "isic_id"]
        else:
            raise KeyError("❌ No 'isic_id' column found in dataframe.")

        # Get label
        if "target" in self.df.columns:
            label = self.df.loc[idx, "target"]
        elif "diagnosis" in self.df.columns:
            diagnosis = str(self.df.loc[idx, "diagnosis"]).lower()
            label = 1 if diagnosis == "melanoma" else 0
        else:
            raise KeyError("❌ No 'target' or 'diagnosis' column found in dataframe.")

        # Full image path
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        # Load image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms if any
        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(label).long()
