import torch
import os
import matplotlib.pyplot as plt


def plot_training_metrics(main_dir, subdirs, save_dir="plots"):
    """
    Plots and saves training losses, testing losses, and testing accuracies for each subdirectory.

    Args:
    - main_dir (str): Path to the main directory containing subdirectories with the 'training_losses.pt' files.
    - subdirs (list of str): List of subdirectory names where the 'training_losses.pt' files are stored.
    - save_dir (str): Directory where the plots will be saved (default: 'plots').

    Returns:
    - None
    """

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize dictionaries to store the data
    training_losses = {}
    testing_losses = {}
    testing_accuracies = {}

    # Loop through each subdirectory and load the training_losses.pt file
    for subdir in subdirs:
        path = os.path.join(main_dir, subdir, "training_losses.pt")
        data = torch.load(path)

        # Store only the first 10 epochs in the respective dictionaries
        training_losses[subdir] = data["training_losses"][:10]
        testing_losses[subdir] = data["testing_losses"][:10]
        testing_accuracies[subdir] = data["testing_accuracies"][:10]

    # Helper function for plotting
    def plot_data(data, title, ylabel, filename):
        plt.figure(figsize=(10, 6))

        # Plot the first 10 epochs for each subdirectory
        for subdir in subdirs:
            plt.plot(
                range(1, 11), data[subdir], label=subdir
            )  # Set x values to range from 1 to 10

        plt.xlim(1, 10)  # Set x-axis limits from 1 to 10

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)

        # Save the plot as a PNG file
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()  # Close the plot after saving

    # Plot and save training losses
    plot_data(
        training_losses,
        "Training Losses for Different Augmented Datasets",
        "Training Loss",
        "training_losses_plot.png",
    )

    # Plot and save testing losses
    plot_data(
        testing_losses,
        "Testing Losses for Different Augmented Datasets",
        "Testing Loss",
        "testing_losses_plot.png",
    )

    # Plot and save testing accuracies
    plot_data(
        testing_accuracies,
        "Testing Accuracies for Different Augmented Datasets",
        "Testing Accuracy",
        "testing_accuracies_plot.png",
    )
