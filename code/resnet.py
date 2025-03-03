import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models
from tqdm import tqdm
import numpy as np
import os

from model import BaseModel


class ResNet(BaseModel):
    def __init__(self):
        super().__init__()

        self.res_lr = 0.0001
        self.fc_lr = 0.001
        self.num_epochs = 5
        self.num_classes = 43

        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_init(self, logger):
        self.model = models.resnet50(pretrained=True)

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        # Unfreeze the last two ResNet blocks
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Move model to the right device
        self.model.to(self.device)

        # Specify layers to optimize
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.model.layer3.parameters(),
                    "lr": self.res_lr,
                },  # Fine-tuned layers
                {
                    "params": self.model.layer4.parameters(),
                    "lr": self.res_lr,
                },  # Fine-tuned layers
                {"params": self.model.fc.parameters(), "lr": self.fc_lr},  # New head
            ]
        )

        self.logger = logger

    def train(self, train_loader: DataLoader, validate_loader: DataLoader):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
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
            train_loss = train_loss / len(train_loader)
            print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

            with torch.no_grad():
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                for images, labels in tqdm(validate_loader, desc=f"Validate {epoch}"):
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forward propagation and loss calculation
                    outputs = self.forward(images)
                    loss = self.loss_fn(outputs, labels)

                    # Update validation statistics
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

                val_accuracy = 100.0 * val_correct / val_total
                val_loss = val_loss / len(validate_loader)
                print(
                    f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%"
                )

            self.logger.log_train(
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
            )

    def eval(self, test_loader: DataLoader):
        self.model.eval()
        y_true = []
        y_pred = []

        correct = 0
        total = 0
        total_inference_time = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass with inference time calculation
                self.logger.start_timer()
                outputs = self.forward(images)
                _, preds = torch.max(outputs, 1)
                total_inference_time += self.logger.stop_timer()

                # Calculate the correct predictions
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                y_true.append(labels.cpu().numpy())
                y_pred.append(preds.cpu().numpy())

        print(f"Testing accuracy: {correct / total * 100:.2f}")

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        inference_time = total_inference_time / len(test_loader)

        self.logger.log_test(
            y_true=y_true, y_pred=y_pred, inference_time=inference_time
        )

    def save(self, save_path: str):
        self.logger.save(save_path=save_path)
        torch.save(self.model.state_dict(), os.path.join(save_path, "weights.pth"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
