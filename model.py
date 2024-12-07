from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from logger import Logger


class BaseModel(ABC):
    @abstractmethod
    def model_init(self, logger: Logger):
        """
        This method is responsible for taking in model hyperparameters and instantiating it.
        """
        pass

    @abstractmethod
    def train(self, train_loader: DataLoader, validate_loader: DataLoader):
        """
        This method is responsible for running the training loop for the model.
        """
        pass

    @abstractmethod
    def eval(self, test_loader: DataLoader):
        """
        This method is responsible for running the model evaluation. It should save
        the training metrics as a pickle file.
        """
        pass

    @abstractmethod
    def save(self, save_path: str):
        """
        This method is responsible for saving the model and pickling the training/testing metrics.
        """
        pass
