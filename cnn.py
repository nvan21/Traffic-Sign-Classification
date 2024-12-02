import torch
import torch.nn as nn
from tqdm import tqdm

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

    def model_init(self):
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
        print(self.model)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        validate_loader: torch.utils.data.DataLoader,
    ):
        self.model.to(self.device)

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

            # Validation phase
            self.model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():  # Disable gradient computation for validation
                for inputs, targets in tqdm(validate_loader, desc="Validation"):
                    # Move data to the device
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # Forward pass
                    outputs = self.model(inputs)

                    # Compute loss
                    loss = self.loss_fn(outputs, targets)

                    # Update validation statistics
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_accuracy = 100.0 * val_correct / val_total
            print(
                f"Validation - Loss: {val_loss / len(validate_loader):.4f}, Accuracy: {val_accuracy:.2f}%\n"
            )

    def eval(self, test_loader: torch.utils.data.DataLoader):
        pass
