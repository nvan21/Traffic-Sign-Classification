import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=transforms):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform

        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        # Load label
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        with open(label_path, "r") as f:
            # Read the class label
            try:
                class_id = int(f.readline().split()[0])
            except:
                print(f"Skipping file: {img_path}")
                return None

        # Apply transformations
        img = self.transform(img)

        return img, class_id
