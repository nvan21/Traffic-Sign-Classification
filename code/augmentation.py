#!/usr/bin/env python3
"""
Image Augmentation Script for Traffic Sign Dataset

This script:
1. Downloads the German Traffic Sign Recognition Benchmark dataset from Kaggle
2. Extracts the dataset to a Data folder
3. Adds black squares to images to augment the dataset

The augmentation simulates occlusions to improve model robustness.
"""

import os
import random
import argparse
import subprocess
import zipfile
import sys
from PIL import Image, ImageDraw


def install_kaggle_if_needed():
    """Install kaggle package if not already installed."""
    try:
        import kaggle

        print("Kaggle package already installed.")
    except ImportError:
        print("Installing Kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("Kaggle package installed successfully.")


def download_kaggle_dataset(data_dir):
    """
    Download the German Traffic Sign dataset from Kaggle.

    Parameters:
    - data_dir: Directory where the dataset will be saved
    """
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    dataset = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
    zip_file = os.path.join(data_dir, "gtsrb-german-traffic-sign.zip")

    # Check if dataset already downloaded
    if os.path.exists(zip_file):
        print(f"Dataset zip file already exists at {zip_file}")
        return zip_file

    print(f"Downloading dataset {dataset} to {data_dir}...")

    # Check for KAGGLE_CONFIG_DIR or KAGGLE_KEY environment variables
    if not os.environ.get("KAGGLE_CONFIG_DIR") and not os.environ.get("KAGGLE_KEY"):
        print("\nWARNING: Kaggle API credentials not found!")
        print("To use the Kaggle API, you need to set up your API credentials:")
        print("1. Go to kaggle.com → Account → 'Create New API Token'")
        print("2. Save the kaggle.json file to ~/.kaggle/ or set KAGGLE_CONFIG_DIR")
        print("3. Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n")

    try:
        subprocess.check_call(
            ["kaggle", "datasets", "download", dataset, "-p", data_dir]
        )
        print(f"Dataset downloaded successfully to {zip_file}")
        return zip_file
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)


def extract_dataset(zip_file, extract_dir):
    """
    Extract the downloaded dataset zip file.

    Parameters:
    - zip_file: Path to the zip file
    - extract_dir: Directory to extract the zip file to
    """
    print(f"Extracting {zip_file} to {extract_dir}...")

    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Dataset extracted successfully.")
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        sys.exit(1)


def add_black_squares(image, num_squares=7, square_size_range=(15, 70)):
    """
    Adds black squares to the image to occlude parts of it.

    Parameters:
    - image: PIL Image object
    - num_squares: Number of squares to add to the image
    - square_size_range: Tuple (min_size, max_size) for random square sizes

    Returns:
    - Augmented PIL Image with black squares
    """
    width, height = image.size
    size_ref = min(width, height)
    small_square = int(size_ref / 12)
    big_square = int(size_ref / 6)

    # Ensure minimum size is at least 1
    if small_square == 0:
        small_square += 1
    if small_square == big_square:
        big_square += 1

    square_size_range = (small_square, big_square)
    draw = ImageDraw.Draw(image)

    for _ in range(num_squares):
        # Randomly determine the size of the square
        square_size = random.randint(square_size_range[0], square_size_range[1])

        # Randomly determine the top-left corner of the square
        top_left_x = random.randint(0, width - square_size)
        top_left_y = random.randint(0, height - square_size)

        # Define the bottom-right corner of the square
        bottom_right_x = top_left_x + square_size
        bottom_right_y = top_left_y + square_size

        # Draw a black square
        draw.rectangle(
            [top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill="black"
        )

    return image


def augment_images(
    input_folder, output_folder, num_squares=7, square_size_range=(15, 70)
):
    """
    Augments all images in the input folder by adding black squares and saves them to the output folder.

    Loops through subfolders in input_folder (representing class directories) and augments images in each class.

    Parameters:
    - input_folder: Path to the folder containing images to augment (with subfolders for each class)
    - output_folder: Path to the folder where augmented images will be saved
    - num_squares: Number of black squares to add to each image
    - square_size_range: Range for the size of the squares
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Loop through all subfolders (each representing a class)
    print(f"Processing images from {input_folder}...")

    class_folders = [
        f
        for f in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, f))
    ]
    total_classes = len(class_folders)
    total_images = 0
    processed_images = 0

    print(f"Found {total_classes} class folders")

    for i, class_folder in enumerate(class_folders, 1):
        class_path = os.path.join(input_folder, class_folder)

        # Create an output folder for the class if it doesn't exist
        class_output_folder = os.path.join(output_folder, class_folder)
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)

        # Count images in this class
        image_files = [
            f
            for f in os.listdir(class_path)
            if f.lower().endswith(("png", "jpg", "jpeg", "bmp", "tiff"))
        ]
        class_images = len(image_files)
        total_images += class_images

        print(
            f"Processing class {i}/{total_classes}: {class_folder} ({class_images} images)"
        )

        # Loop through all images in the class folder
        for j, img_name in enumerate(image_files, 1):
            img_path = os.path.join(class_path, img_name)

            try:
                # Open the image
                image = Image.open(img_path)

                # Apply black squares augmentation
                augmented_image = add_black_squares(
                    image.copy(), num_squares, square_size_range
                )

                # Save the augmented image to the appropriate class folder in the output folder
                augmented_img_path = os.path.join(class_output_folder, img_name)
                augmented_image.save(augmented_img_path)

                processed_images += 1
                if j % 100 == 0:
                    print(f"  Processed {j}/{class_images} images in current class")

            except Exception as e:
                print(f"  Error processing {img_path}: {e}")

    print(
        f"Augmentation complete! Processed {processed_images}/{total_images} images successfully."
    )


def main():
    """Main function to parse arguments and run the augmentation."""
    parser = argparse.ArgumentParser(
        description="Download, extract, and augment the German Traffic Sign dataset."
    )
    parser.add_argument(
        "--data-dir",
        default="Data",
        help="Directory to store the dataset (default: Data)",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip downloading the dataset"
    )
    parser.add_argument(
        "--input", help="Input folder containing images (default: Data/train)"
    )
    parser.add_argument(
        "--output",
        help="Output folder for augmented images (default: Data/train_augment)",
    )
    parser.add_argument(
        "--num-squares",
        "-n",
        type=int,
        default=7,
        help="Number of black squares to add to each image",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=15,
        help="Minimum size of squares (will be adjusted based on image size)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=70,
        help="Maximum size of squares (will be adjusted based on image size)",
    )

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # Download dataset if not skipped
    if not args.skip_download:
        install_kaggle_if_needed()
        zip_file = download_kaggle_dataset(args.data_dir)
        extract_dataset(zip_file, args.data_dir)

    # Set default input and output paths if not provided
    input_folder = args.input if args.input else os.path.join(args.data_dir, "train")
    output_folder = (
        args.output if args.output else os.path.join(args.data_dir, "train_augment")
    )

    # Verify input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        print(
            "If you just downloaded the dataset, make sure the train folder path is correct."
        )
        print("Use --input to specify the correct path to the training images.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Starting image augmentation with the following settings:")
    print(f"- Number of black squares per image: {args.num_squares}")
    print(
        f"- Square size range: {args.min_size} to {args.max_size} (adjusted based on image size)"
    )
    print(f"- Input folder: {input_folder}")
    print(f"- Output folder: {output_folder}")
    print("=" * 50 + "\n")

    augment_images(
        input_folder,
        output_folder,
        num_squares=args.num_squares,
        square_size_range=(args.min_size, args.max_size),
    )

    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()
