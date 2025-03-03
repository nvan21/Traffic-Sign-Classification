import os
import argparse
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
import torch
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import csv


# Creates the custom image dataset to only get the class ID from the label file
class ImageDataset(Dataset):
    def __init__(self, csv_path: str, transform=transforms):
        self.transform = transform
        self.data = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Each row is a dict:
                # {
                #   'Width': str_value,
                #   'Height': str_value,
                #   'Roi.X1': str_value,
                #   'Roi.Y1': str_value,
                #   'Roi.X2': str_value,
                #   'Roi.Y2': str_value,
                #   'ClassId': str_value,
                #   'Path': str_value
                # }
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]

        # Extract the image path and label
        img_path = os.path.join("./data", row["Path"])
        class_id = int(row["ClassId"])  # convert label to int if necessary

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Apply any transforms
        if self.transform:
            image = self.transform(image)

        # Return image and its class label
        return image, class_id


def transform_to_pca(loader: DataLoader, scaler: StandardScaler, ipca: IncrementalPCA):
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
    X_transformed_tensor = torch.from_numpy(X_transformed).float()
    y_tensor = torch.from_numpy(y).long()

    return X_transformed_tensor, y_tensor, X_transformed, y


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA dimensionality reduction for image datasets"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=150,
        help="Number of PCA components (default: 150)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for data loading (default: 512)",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Set up data transformation
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    data_paths = ["./data/Train", "./data/Train_Augmented"]
    batch_size = args.batch_size

    # Make ImageFolder datasets for each augmented image path
    image_datasets = []
    for path in data_paths:
        image_datasets.append(datasets.ImageFolder(path, transform=transform))

    # Make a combined training dataset and then split into training and validation
    combined_dataset = ConcatDataset(image_datasets)
    train_size = int(0.8 * len(combined_dataset))
    validate_size = len(combined_dataset) - train_size
    train_dataset, validate_dataset = random_split(
        combined_dataset, [train_size, validate_size]
    )
    test_dataset = ImageDataset(csv_path="./data/Test.csv", transform=transform)

    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Get number of PCA components from command line args
    n_components = args.n_components

    # 1. Incrementally fit the scaler on the entire training set
    scaler = StandardScaler()
    for images, _ in tqdm(train_loader, desc="Getting PCA scaler"):
        batch_np = images.view(images.size(0), -1).numpy()
        scaler.partial_fit(batch_np)

    # 2. Incrementally fit PCA to avoid memory issues
    ipca = IncrementalPCA(n_components=n_components)
    for images, _ in tqdm(train_loader, desc="Fitting PCA"):
        batch_np = images.view(images.size(0), -1).numpy()
        batch_scaled = scaler.transform(batch_np)
        ipca.partial_fit(batch_scaled)

    # 3. Convert each dataset to the PCA version
    train_pca, train_labels, train_pca_np, train_labels_np = transform_to_pca(
        loader=train_loader, scaler=scaler, ipca=ipca
    )
    validate_pca, validate_labels, validate_pca_np, validate_labels_np = (
        transform_to_pca(loader=validate_loader, scaler=scaler, ipca=ipca)
    )
    test_pca, test_labels, test_pca_np, test_labels_np = transform_to_pca(
        loader=test_loader, scaler=scaler, ipca=ipca
    )

    # 4. Save the transformed datasets
    path = os.path.join("./data/PCA}", f"{n_components}")
    os.makedirs(path, exist_ok=True)

    torch.save(train_pca, os.path.join(path, "X_train.pt"))
    torch.save(train_labels, os.path.join(path, "y_train.pt"))
    torch.save(validate_pca, os.path.join(path, "X_validate.pt"))
    torch.save(validate_labels, os.path.join(path, "y_validate.pt"))
    torch.save(test_pca, os.path.join(path, "X_test.pt"))
    torch.save(test_labels, os.path.join(path, "y_test.pt"))


if __name__ == "__main__":
    main()
