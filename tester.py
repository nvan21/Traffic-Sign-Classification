from CNN import CNN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def test_cnn_model(model: CNN, test_loader: DataLoader):
    # Put the model in evaluation mode
    model.eval()

    # Initialize counters
    correct = 0
    total = 0

    # Disable gradient calculation for testing
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            # Move data to the appropriate device (GPU or CPU)
            images, labels = images.to(model.device), labels.to(model.device)

            # Forward pass to get predictions
            logits = model(images)
            outputs = nn.Softmax(dim=1)(logits)

            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs, 1)

            # Update total and correct counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy on the test dataset: {accuracy:.2f}%")
