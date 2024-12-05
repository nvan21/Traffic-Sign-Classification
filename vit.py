import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models
from tqdm import tqdm

from model import BaseModel


class ViT(BaseModel):
    def __init__(self):
        super().__init__()

        self.lr = 0.001
        self.num_epochs = 30
        self.num_classes = 43

        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_init(self):
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = models.vit_b_16(weights=weights)

        # Replace the final fully connected layer
        self.model.heads[0] = nn.Linear(
            self.model.heads[0].in_features, self.num_classes
        )

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze only the classification head
        for param in self.model.heads[0].parameters():
            param.requires_grad = True

        self.model.to(self.device)

        # Specify fully connected layer parameters to optimize
        params = [param for param in self.model.heads[0].parameters()]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def train(self, train_loader: DataLoader, validate_loader: DataLoader):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in tqdm(train_loader, desc="Training"):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward and backward propagation
                outputs = self.forward(images)
                loss = self.loss_fn(outputs, labels)
                self.backward(loss=loss)

                # Update training statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_accuracy = 100.0 * train_correct / train_total
            print(
                f"Training - Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%"
            )

    def eval(self, test_loader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.forward(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Testing accuracy: {correct / total * 100:.2f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
