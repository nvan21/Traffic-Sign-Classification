from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import time
import pickle
import os


class Logger:
    def __init__(self):
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "test_accuracy": None,
            "inference_time": None,
            "confusion_matrix": None,
            "f1_score": None,
        }

        # Stopwatch variable
        self._start_time = None

    def log_train(
        self,
        train_loss: float,
        val_loss: float,
        train_accuracy: float,
        val_accuracy: float,
    ):
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["train_accuracy"].append(train_accuracy)
        self.metrics["val_accuracy"].append(val_accuracy)

    def log_test(self, y_true: np.ndarray, y_pred: np.ndarray, inference_time: float):
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise ValueError("y_true and y_pred must be NumPy arrays.")

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")

        # Calculate metrics
        self.metrics["test_accuracy"] = np.mean(y_true == y_pred)
        self.metrics["inference_time"] = inference_time
        self.metrics["confusion_matrix"] = confusion_matrix(
            y_true=y_true, y_pred=y_pred
        )
        self.metrics["f1_score"] = f1_score(
            y_true=y_true, y_pred=y_pred, average="weighted"
        )

    def save(self, save_path: str):
        file = os.path.join(save_path, "log.pkl")
        with open(file, "wb") as f:
            pickle.dump(self.metrics, f)

    def start_timer(self):
        self._start_time = time.time()

    def stop_timer(self):
        if self._start_time is None:
            raise RuntimeError(
                "Timer was not started. Call start_timer() before stop_timer()."
            )

        elapsed = time.time() - self._start_time
        self._start_time = None
        return elapsed
