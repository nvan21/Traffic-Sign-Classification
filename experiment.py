import pathlib
import re
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
import torch
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
from tqdm import tqdm

from dataset import ImageDataset
from model import BaseModel
from logger import Logger
from cnn import CNN
from svm import SVM
from knn import KNN
from resnet import ResNet
from vit import ViT

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


DATA_PATHS = ["./Data/Train", "./Data/train_augment"]
SAVE_PATH = "./runs"
BASE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
RESNET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
EXPERIMENTS = {
    # "base_cnn": {
    #     "data_paths": DATA_PATHS,
    #     "save_path": SAVE_PATH,
    #     "batch_size": 64,
    #     "transform": BASE_TRANSFORM,
    #     "model": CNN(),
    # },
    # "base_svm": {
    #     "data_paths": DATA_PATHS,
    #     "save_path": SAVE_PATH,
    #     "batch_size": 64,
    #     "transform": BASE_TRANSFORM,
    #     "do_pca": True,
    #     "n_components": 150,
    #     "model": SVM(),
    # },
    # "base_knn": {
    #     "data_paths": DATA_PATHS,
    #     "save_path": SAVE_PATH,
    #     "batch_size": 64,
    #     "transform": BASE_TRANSFORM,
    #     "do_pca": True,
    #     "n_components": 150,
    #     "model": KNN(),
    # },
    "base_resnet": {
        "data_paths": DATA_PATHS,
        "save_path": SAVE_PATH,
        "batch_size": 64,
        "transform": RESNET_TRANSFORM,
        "model": ResNet(),
    },
    "base_vit": {
        "data_paths": DATA_PATHS,
        "save_path": SAVE_PATH,
        "batch_size": 64,
        "transform": RESNET_TRANSFORM,
        "model": ViT(),
    },
}


class Experiment:
    def __init__(self, id: str, data_paths: str, batch_size: int):
        self.id = id
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.logger = Logger()

    def preprocessing(
        self, transform: transforms, do_pca: bool = False, n_components: int = 50
    ):
        # Make ImageFolder datasets for each augmented image path
        image_datasets = []
        for path in self.data_paths:
            image_datasets.append(datasets.ImageFolder(path, transform=transform))

        # Make a combined training dataset and then split into training and validation
        combined_dataset = ConcatDataset(image_datasets)
        train_size = int(0.8 * len(combined_dataset))
        validate_size = len(combined_dataset) - train_size
        train_dataset, validate_dataset = random_split(
            combined_dataset, [train_size, validate_size]
        )
        test_dataset = ImageDataset(csv_path="./Data/Test.csv", transform=transform)

        # Create dataloaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.validate_loader = DataLoader(
            dataset=validate_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        if do_pca:
            # 1. Load PCA datasets
            train_data = torch.load(f"./Data/pca/{n_components}/X_train.pt")
            train_labels = torch.load(f"./Data/pca/{n_components}/y_train.pt")
            validate_data = torch.load(f"./Data/pca/{n_components}/X_validate.pt")
            validate_labels = torch.load(f"./Data/pca/{n_components}/y_validate.pt")
            test_data = torch.load(f"./Data/pca/{n_components}/X_test.pt")
            test_labels = torch.load(f"./Data/pca/{n_components}/y_test.pt")

            # 2. Create Tensor Datasets
            train_dataset = TensorDataset(train_data, train_labels)
            validate_dataset = TensorDataset(validate_data, validate_labels)
            test_dataset = TensorDataset(test_data, test_labels)

            # 3. Update dataloaders with PCA-transformed versions
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
            self.validate_loader = DataLoader(
                dataset=validate_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )

    def create_model(self, model: BaseModel):
        self.model = model
        self.model.model_init(logger=self.logger)

    def _transform_to_pca(
        self, loader: DataLoader, scaler: StandardScaler, ipca: IncrementalPCA
    ):
        # Collect all images from the training set for PCA
        transformed_list = []
        label_list = []
        for images, labels in tqdm(loader, "Fitting PCA for dataset"):
            batch_np = images.view(images.size(0), -1).numpy()
            batch_scaled = scaler.transform(batch_np)
            batch_pca = ipca.transform(batch_scaled)
            transformed_list.append(batch_pca)
            label_list.append(labels.numpy())

        # Concatenate all batches
        X_transformed = np.concatenate(transformed_list, axis=0)
        y = np.concatenate(label_list, axis=0)

        # Convert to torch tensors
        X_transformed = torch.from_numpy(X_transformed).float()
        y = torch.from_numpy(y).long()

        return X_transformed, y

    def _load_images_in_batches(paths, batch_size: int = 500):
        """Generator that yields batches of flattened images"""
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            batch_data = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
