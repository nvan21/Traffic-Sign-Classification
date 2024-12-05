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
        # self.model.apply(self._init_weights)
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

    def eval(self, test_loader: torch.utils.data.DataLoader):
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

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

            if m.bias is not None:
                nn.init.zeros_(m.bias)
