import os
from torch.utils.data import DataLoader, ConcatDataset
import glob
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

original_paths = sorted(
    glob.glob("/work/flemingc/nvan21/projects/COMS_Project_573/Data/Train")
)
occlusion_paths = sorted(
    glob.glob("/work/flemingc/nvan21/projects/COMS_Project_573/Data/train_augment")
)
all_paths = original_paths + occlusion_paths


class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


original_dataset = datasets.ImageFolder(
    "/work/flemingc/nvan21/projects/COMS_Project_573/Data/Train"
)
occlusion_dataset = datasets.ImageFolder(
    "/work/flemingc/nvan21/projects/COMS_Project_573/Data/train_augment"
)
combined_dataset = ConcatDataset([original_dataset, occlusion_dataset])

dataloader = DataLoader(combined_dataset, batch_size=64, shuffle=True, num_workers=4)
