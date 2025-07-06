import os
import torch
from torch import nn, optim
from tqdm import tqdm

def train_model(
    model,
    dataloader,
    device,
    epochs=5,
    lr=1e-4,
    save_path="outputs/checkpoints/model_final.pth",
    class_weights=(0.25, 0.75)
):
    """
    Trains a PyTorch model.

    Args:
        model: nn.Module
        dataloader: DataLoader
        device: CUDA or CPU
        epochs: number of epochs
        lr: learning rate
        save_path: where to save final weights
        class_weights: tuple with weights for classes (benign, melanoma)
    """
    # Weighted CrossEntropyLoss
    weight_tensor = torch.tensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler (fixed: removed 'verbose')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"Batch Loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        acc = correct / total
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")

        # Scheduler step
        scheduler.step(avg_loss)

    # Save final model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")
