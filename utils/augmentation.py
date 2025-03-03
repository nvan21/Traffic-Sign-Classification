#!/usr/bin/env python3
"""
Image Augmentation Script for Traffic Sign Dataset

This script adds black squares to images to augment the dataset.
The augmentation simulates occlusions to improve model robustness.
"""

import os
import random
import argparse
import sys
from PIL import Image, ImageDraw


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

    # Use the provided range, but adjust if necessary for small images
    min_size = min(square_size_range[0], int(size_ref / 12))
    max_size = min(square_size_range[1], int(size_ref / 6))

    # Ensure minimum size is at least 1
    min_size = max(1, min_size)
    max_size = max(min_size + 1, max_size)

    adjusted_range = (min_size, max_size)
    draw = ImageDraw.Draw(image)

    for _ in range(num_squares):
        # Randomly determine the size of the square
        square_size = random.randint(adjusted_range[0], adjusted_range[1])

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

                # Convert to RGB if not already (ensures black fill works)
                if image.mode != "RGB":
                    image = image.convert("RGB")

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
        description="Augment traffic sign images by adding black square occlusions."
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory to store the dataset (default: data)",
    )
    parser.add_argument(
        "--input",
        default="./data/Train",
        help="Input folder containing images (default: data/train)",
    )
    parser.add_argument(
        "--output",
        default="./data/Train_Augmented",
        help="Output folder for augmented images (default: data/train_augment)",
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

    # Verify input folder exists
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist.")
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
    print(f"- Input folder: {args.input}")
    print(f"- Output folder: {args.output}")
    print("=" * 50 + "\n")

    augment_images(
        args.input,
        args.output,
        num_squares=args.num_squares,
        square_size_range=(args.min_size, args.max_size),
    )

    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()
