import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def plot_class_distribution(counts, labels, save_path=None):
    """
    Plot a bar chart of class distribution.
    """
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=counts)
    plt.title("Class Distribution")
    plt.ylabel("Number of Images")
    plt.xlabel("Class")
    for i, v in enumerate(counts):
        plt.text(i, v + 10, str(v), ha="center")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Saved class distribution plot to {save_path}")
    plt.show()

def show_random_samples(df, image_dir, n=6, label_column="target"):
    """
    Display random image samples with labels.
    """
    sample_df = df.sample(n)
    plt.figure(figsize=(15, 5))
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img_id = row["image_name"]
        label = row[label_column]
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not load image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n, idx + 1)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ===============================
    # Resolve paths robustly
    # ===============================
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATA_CSV = os.path.join(BASE_DIR, "data", "train.csv")
    IMAGE_DIR = os.path.join(BASE_DIR, "data", "train_images")
    SAVE_PLOT = os.path.join(BASE_DIR, "outputs", "class_distribution.png")

    # ===============================
    # Load metadata
    # ===============================
    df = pd.read_csv(DATA_CSV)
    print(f"✅ Loaded {len(df)} entries from {DATA_CSV}")

    # ===============================
    # Detect label column
    # ===============================
    if "target" in df.columns:
        label_column = "target"
        num_melanoma = df["target"].sum()
        num_benign = len(df) - num_melanoma
        counts = [num_benign, num_melanoma]
        labels = ["Benign (0)", "Melanoma (1)"]
    elif "diagnosis" in df.columns:
        label_column = "diagnosis"
        num_melanoma = (df["diagnosis"].str.lower() == "melanoma").sum()
        num_benign = len(df) - num_melanoma
        counts = [num_benign, num_melanoma]
        labels = ["Benign", "Melanoma"]
    else:
        raise ValueError("❌ Could not find 'target' or 'diagnosis' column in CSV.")

    # ===============================
    # Basic statistics
    # ===============================
    print(f"Benign cases   : {num_benign}")
    print(f"Melanoma cases : {num_melanoma}")
    print(f"Melanoma ratio : {num_melanoma / len(df):.4f}")

    # ===============================
    # Plot class distribution
    # ===============================
    os.makedirs(os.path.dirname(SAVE_PLOT), exist_ok=True)
    plot_class_distribution(counts, labels, save_path=SAVE_PLOT)

    # ===============================
    # Show random samples
    # ===============================
    show_random_samples(df, IMAGE_DIR, n=6, label_column=label_column)

    print("✅ Data exploration complete.")
