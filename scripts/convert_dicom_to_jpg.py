import os
import pydicom
import cv2
import numpy as np

# Path to your DICOM images
INPUT_DIR = r"D:\code\skin_cancer_detection\data\unseen_images\ISIC_2020_Test_DICOM_corrected\test"
OUTPUT_DIR = r"D:\code\skin_cancer_detection\data\\ISIC_2020_Test_DICOM_corrected\test1\unseen_images_jpg"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(INPUT_DIR):
    if fname.lower().endswith(".dcm"):
        dcm_path = os.path.join(INPUT_DIR, fname)
        ds = pydicom.dcmread(dcm_path)
        img_array = ds.pixel_array

        # Normalize to 0-255
        img_norm = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_norm.astype(np.uint8)

        # Save as .jpg
        out_path = os.path.join(OUTPUT_DIR, fname.replace(".dcm", ".jpg"))
        cv2.imwrite(out_path, img_uint8)

print("✅ All DICOM files converted to JPG.")
