from CNN import CNN

import torch
from torch.utils.data import DataLoader
from typing import Union
from tqdm import tqdm
import os


class CNNTrainer:
    def __init__(
        self,
        model: CNN,
        train_loader: DataLoader,
        test_loader: DataLoader,
        save_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_dir = save_dir

    def train(
        self, num_epochs: int, criterion: torch.nn.CrossEntropyLoss
    ) -> tuple[list, list, list]:
        # Metrics to store
        best_test_accuracy = 0.0
        test_accuracies = []
        test_losses = []
        train_losses = []

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0

            for images, labels in tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
            ):
                images, labels = images.to(self.model.device), labels.to(
                    self.model.device
                )

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                self.model.backward(loss=loss)

                # Add loss to running loss
                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.model.device), labels.to(
                        self.model.device
                    )
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Compute average losses and accuracy
            avg_train_loss = train_loss / len(self.train_loader)
            avg_test_loss = test_loss / len(self.test_loader)
            test_accuracy = 100 * correct / total

            # Save average losses and accuracy
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)

            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, "best_model.pt"),
                )

            # Print the results
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Training Loss: {avg_train_loss:.4f}, "
                f"Testing Loss: {avg_test_loss:.4f}, "
                f"Testing Accuracy: {test_accuracy:.2f}%"
            )

        torch.save(
            {
                "training_losses": train_losses,
                "testing_losses": test_losses,
                "testing_accuracies": test_accuracies,
            },
            os.path.join(self.save_dir, "training_losses.pt"),
        )
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, "last_model.pt"),
        )
        return train_losses, test_losses, test_accuracies
