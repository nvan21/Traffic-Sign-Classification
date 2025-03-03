from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch
import numpy as np
import pickle
import os

from model import BaseModel


class KNN(BaseModel):
    def __init__(self):
        super().__init__()

    def model_init(self, logger):
        # Hyperparameters
        param_grid = {
            "n_neighbors": [3, 5, 7, 9],  # Test different values of k
            "weights": ["uniform", "distance"],  # Uniform or distance-based weighting
            "metric": ["euclidean", "manhattan"],  # Distance metrics
        }

        self.grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid=param_grid,
            refit=True,
            cv=5,
            scoring="accuracy",
            verbose=3,
        )
        self.model = KNeighborsClassifier(
            metric="manhattan", n_neighbors=5, weights="distance"
        )

        self.logger = logger

    def train(self, train_loader: DataLoader, validate_loader: DataLoader):
        X_train, y_train = self._loader_to_numpy(loader=train_loader)
        X_validate, y_validate = self._loader_to_numpy(loader=validate_loader)

        X = np.vstack((X_train, X_validate))
        y = np.hstack((y_train, y_validate))

        # Find the best model hyperparameters
        # self.grid_search.fit(X=X, y=y)

        # Select the best model
        # self.model = self.grid_search.best_estimator_
        # print(self.grid_search.best_estimator_)

        # Train the best model
        self.model.fit(X=X, y=y)

        # Print out training accuracy
        y_train_pred = self.model.predict(X=X)
        train_accuracy = accuracy_score(y, y_train_pred)
        print(f"Train accuracy with best model: {train_accuracy * 100}%")

    def eval(self, test_loader: DataLoader):
        # Predict on the test set
        X_test, y_test = self._loader_to_numpy(loader=test_loader)
        self.logger.start_timer()
        y_test_pred = self.model.predict(X=X_test)
        inference_time = self.logger.stop_timer() / len(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test accuracy with best model: {test_accuracy*100}%")

        self.logger.log_test(
            y_true=y_test, y_pred=y_test_pred, inference_time=inference_time
        )

    def save(self, save_path):
        self.logger.save(save_path=save_path)
        model_save_path = os.path.join(save_path, "model.pkl")
        with open(model_save_path, "wb") as f:
            pickle.dump(self.model, f)

    def _loader_to_numpy(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        x_list, y_list = [], []

        for batch in loader:
            features, labels = batch
            x_list.append(features)
            y_list.append(labels)

        # Combine into single tensors and then convert to NumPy
        x = torch.cat(x_list).numpy()
        y = torch.cat(y_list).numpy()

        return x, y
