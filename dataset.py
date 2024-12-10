import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
        img_path = os.path.join(
            "/work/flemingc/nvan21/projects/COMS_573_Project/Data", row["Path"]
        )
        class_id = int(row["ClassId"])  # convert label to int if necessary

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Apply any transforms
        if self.transform:
            image = self.transform(image)

        # Return image and its class label
        return image, class_id
