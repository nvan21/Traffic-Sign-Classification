import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models
from tqdm import tqdm
import numpy as np
import os

from model import BaseModel


class ViT(BaseModel):
    def __init__(self):
        super().__init__()

        self.lr = 0.001
        self.num_epochs = 10
        self.num_classes = 15

        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_init(self, logger):
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

        # Create logger
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
