import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import MNISTCNN
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
DEVICE = torch.device("cpu")
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "mnist_cnn.pth"


def get_dataloaders():
    """
    Create and return MNIST train, validation, and test dataloaders.
    Splits training data into train (90%) and validation (10%).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # (Mean, Std Deviation)
    ])
    
    full_train_set = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Split train set into train (90%) and validation (10%)
    train_size = int(0.9 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])
    
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False),
    )


def train_model():
    # Initialize model, loss, optimizer
    model = MNISTCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load data
    train_loader, val_loader = get_dataloaders()
    
    best_val_loss = float("inf")
    iteration_losses = []  # Training loss per iteration
    epoch_val_losses = []  # Validation loss per epoch

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            iteration_losses.append(loss.item())  # Store training loss for each iteration

        # Compute average training loss
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)  # Compute validation loss
                val_loss += loss.item() * images.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_losses.append(epoch_val_loss)  # Store validation loss per epoch

        print(f"Training Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

        # Save best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved new best model with validation loss {best_val_loss:.4f}")

    # Plot Training Loss vs Iteration and Validation Loss vs Epoch
    plt.plot(range(len(iteration_losses)), iteration_losses, label="Training Loss", alpha=0.7)
    plt.plot(
        [i * len(train_loader) for i in range(1, len(epoch_val_losses) + 1)],
        epoch_val_losses,
        label="Validation Loss",
        marker="o",
        linestyle="dashed",
        color="red",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_model()
    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")