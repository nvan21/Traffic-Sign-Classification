import os
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dataset import ImageDataset
import hyperparameters
from model import BaseModel
from cnn import CNN
from svm import SVM

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
    # "base_cnn": {
    #     "data_path": DATA_PATH,
    #     "batch_size": 32,
    #     "transform": BASE_TRANSFORM,
    #     "model": CNN(),
    #     "params": cnn_params,
    # },
    "base_svm": {
        "data_path": DATA_PATH,
        "batch_size": 32,
        "transform": BASE_TRANSFORM,
        "do_pca": True,
        "n_components": 50,
        "model": SVM(),
    },
}


class Experiment:
    def __init__(self, id: str, data_path: str, batch_size: int):
        #! Still needs save protocols
        self.id = id
        self.data_path = data_path
        self.batch_size = batch_size

    def preprocessing(
        self, transform: transforms, do_pca: bool = False, n_components: int = 50
    ):
        train_path = os.path.join(self.data_path, "train")
        validate_path = os.path.join(self.data_path, "valid")
        test_path = os.path.join(self.data_path, "test")

        train_dataset = ImageDataset(root_dir=train_path, transform=transform)
        validate_dataset = ImageDataset(root_dir=validate_path, transform=transform)
        test_dataset = ImageDataset(root_dir=test_path, transform=transform)

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        self.validate_loader = DataLoader(
            dataset=validate_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        if do_pca:
            # Collect all images from the training set for PCA
            train_images = []
            labels = []
            for images, label_batch in self.train_loader:
                labels.append(label_batch)
                train_images.append(
                    images.view(images.size(0), -1)
                )  # Flatten each image
            train_matrix = torch.cat(train_images).numpy()

            # Standardize the training data. This scaler will be used to standardize the validation and testing datasets
            scaler = StandardScaler()
            train_standardized = scaler.fit_transform(train_matrix)

            # Perform PCA on the training data
            pca = PCA(n_components=n_components)
            train_pca = pca.fit_transform(train_standardized)
            train_pca = torch.tensor(train_pca)
            train_labels = torch.cat(labels)

            # Perform PCA on the validation and testing data using the training scaler
            validate_pca, validate_labels = self._transform_to_pca(
                loader=self.validate_loader, scaler=scaler, pca=pca
            )
            test_pca, test_labels = self._transform_to_pca(
                loader=self.test_loader, scaler=scaler, pca=pca
            )

            # Update dataloaders with PCA-transformed versions
            self.train_loader = DataLoader(
                dataset=TensorDataset(train_pca, train_labels),
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self._collate_fn,
            )
            self.validate_loader = DataLoader(
                dataset=TensorDataset(validate_pca, validate_labels),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
            )
            self.test_loader = DataLoader(
                dataset=TensorDataset(test_pca, test_labels),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
            )

    def create_model(self, model: BaseModel):
        self.model = model
        self.model.model_init()

    def _transform_to_pca(self, loader: DataLoader, scaler: StandardScaler, pca: PCA):
        # Collect all images from the training set for PCA
        pca_images = []
        labels = []
        for images, label_batch in loader:
            labels.append(label_batch)
            pca_images.append(images.view(images.size(0), -1))  # Flatten each image
        matrix = torch.cat(pca_images).numpy()
        standardized = scaler.transform(matrix)
        pca_transformed = pca.transform(standardized)

        return torch.tensor(pca_transformed), torch.cat(labels)

    def _collate_fn(self, batch):
        # Filter out images that don't have labels
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        images, labels = zip(*batch)
        return torch.stack(images), torch.tensor(labels)
