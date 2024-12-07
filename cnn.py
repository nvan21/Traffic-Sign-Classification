import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

from model import BaseModel


class CNN(BaseModel, nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = [
            (3, 16, 3, 1, 1),
            (16, 32, 3, 1, 1),
            (32, 64, 3, 1, 1),
            (64, 128, 3, 1, 1),
        ]
        self.fc_layers = [(128 * 14 * 14, 1024), (1024, 256), (256, 15)]
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.num_epochs = 10

        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model_init(self, logger):
        layers = []

        # Add convolutional layers
        for in_dim, out_dim, kernel_size, stride, padding in self.conv_layers:
            layers.append(
                nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(self.dropout_rate))
            layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(nn.Flatten())

        # Add fully connected layers
        for i, (in_dim, out_dim) in enumerate(self.fc_layers):
            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            if i < len(self.fc_layers) - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout_rate))

        # Create sequential model
        self.model = nn.Sequential(*layers)
        # self.model.apply(self._init_weights)
        self.model.to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )

        self.logger = logger

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

            if m.bias is not None:
                nn.init.zeros_(m.bias)
