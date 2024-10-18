from CNN import create_model
from data_preprocessing import create_dataloaders
from trainer import CNNTrainer
from plotter import plot_training_metrics

import torch
from torchvision import transforms
import os

# Dataset hyperparameters
ROOT_DIR = "/work/flemingc/nvan21/projects/COMS_573_Project"
BASE_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
EXPERIMENTS = {
    "Base": BASE_TRANSFORMS,
    "Variable Brightness": transforms.Compose(
        [*BASE_TRANSFORMS.transforms, transforms.ColorJitter(brightness=0.2)]
    ),
    "Variable Contrast": transforms.Compose(
        [*BASE_TRANSFORMS.transforms, transforms.ColorJitter(contrast=0.2)]
    ),
    "Variable Saturation": transforms.Compose(
        [*BASE_TRANSFORMS.transforms, transforms.ColorJitter(saturation=0.2)]
    ),
    "Variable Hue": transforms.Compose(
        [*BASE_TRANSFORMS.transforms, transforms.ColorJitter(hue=0.2)]
    ),
}
EXPERIMENT_NAMES = [
    "Base",
    "Variable Brightness",
    "Variable Contrast",
    "Variable Saturation",
    "Variable Hue",
]
BATCH_SIZE = 32
SHUFFLE = True
TRAIN_SPLIT = 0.8

# CNN hyperparameters
NUM_CLASSES = 58
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = torch.nn.CrossEntropyLoss()
SAVE_DIR = "/work/flemingc/nvan21/projects/COMS_573_Project/model_data"
NUM_EPOCHS = 10


if __name__ == "__main__":
    # for name, transform in EXPERIMENTS.items():
    #     train_loader, test_loader = create_dataloaders(
    #         root_dir=ROOT_DIR,
    #         transforms=transform,
    #         batch_size=BATCH_SIZE,
    #         shuffle=SHUFFLE,
    #     )

    #     cnn = create_model(num_classes=NUM_CLASSES, device=DEVICE)
    #     save_dir = os.path.join(SAVE_DIR, name)
    #     os.makedirs(save_dir, exist_ok=True)
    #     trainer = CNNTrainer(
    #         model=cnn,
    #         train_loader=train_loader,
    #         test_loader=test_loader,
    #         save_dir=save_dir,
    #     )

    #     train_losses, val_losses, val_accuracies = trainer.train(
    #         num_epochs=NUM_EPOCHS, criterion=CRITERION
    #     )

    plot_training_metrics(main_dir="model_data", subdirs=list(EXPERIMENTS.keys()))
