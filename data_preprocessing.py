from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, self.image_files[idx])

        image = Image.open(image_path)
        label = int(image_name[:3])
        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    root_dir: str,
    transforms: transforms.Compose,
    batch_size: int,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    # Create dataset paths
    train_path = os.path.join(root_dir, "traffic_Data/DATA")
    test_path = os.path.join(root_dir, "traffic_Data/TEST")

    # Create the training and test datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=transforms)
    test_dataset = TestDataset(root_dir=test_path, transform=transforms)

    # Create the training and testing dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=shuffle
    )

    return train_loader, test_loader
