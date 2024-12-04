import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models
from tqdm import tqdm

from model import BaseModel


class ResNet(BaseModel):
    def __init__(self):
        super().__init__()

        self.lr = 0.001
        self.num_epochs = 30

        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_init(self):
        self.model = models.resnet50(pretrained=True)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer
        num_classes = 15
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.model.to(self.device)

        # Specify fully connected layer parameters to optimize
        params = [param for param in self.model.fc.parameters()]
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
