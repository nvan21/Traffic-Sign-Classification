import os
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import ImageDataset
import hyperparameters
from model import BaseModel
from cnn import CNN

cnn_params = hyperparameters.CNNHyperparameters()
resnet_params = hyperparameters.ResNetHyperparameters()
vit_params = hyperparameters.ViTHyperparameters()
svm_params = hyperparameters.SVMHyperparameters()
knn_params = hyperparameters.KNNHyperparameters()

DATA_PATH = "./data/car"
BASE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
EXPERIMENTS = {
    "base_cnn": {
        "data_path": DATA_PATH,
        "batch_size": 32,
        "transform": BASE_TRANSFORM,
        "model": CNN(),
        "params": cnn_params,
    },
}


class Experiment:
    def __init__(self, id: str, data_path: str, batch_size: int):
        #! Still needs save protocols
        self.id = id
        self.data_path = data_path
        self.batch_size = batch_size

    def preprocessing(self, transform: transforms):
        train_path = os.path.join(self.data_path, "train")
        validate_path = os.path.join(self.data_path, "valid")
        test_path = os.path.join(self.data_path, "test")

        train_dataset = ImageDataset(root_dir=train_path, transform=transform)
        validate_dataset = ImageDataset(root_dir=validate_path, transform=transform)
        test_dataset = ImageDataset(root_dir=test_path, transform=transform)

        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.validate_loader = DataLoader(
            dataset=validate_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def create_model(self, model: BaseModel):
        self.model = model
        self.model.model_init()
