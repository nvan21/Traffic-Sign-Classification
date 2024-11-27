from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseModel(ABC):
    @abstractmethod
    def model_init(self, params):
        """
        This method is responsible for taking in model hyperparameters and instantiating it.
        """
        pass

    @abstractmethod
    def train(self, train_loader: DataLoader, validate_loader: DataLoader):
        """
        This method is responsible for running the training loop for the model
        """
        pass

    @abstractmethod
    def eval(self):
        """
        This method is responsible for running the model evaluation. It should save
        the training metrics as a pickle file.
        """
        pass
