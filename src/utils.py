import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import numpy as np

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Compute and print standard metrics.
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Benign", "Melanoma"])

    print("=== Classification Report ===")
    print(report)
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report
    }

def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plot ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.4f}")
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Melanoma"],
        yticklabels=["Benign", "Melanoma"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
