#!/usr/bin/env python3
"""
Model Metrics Plotter

This script loads model metrics from pickle files and creates 
visualizations for comparison between different models.
"""

import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics():
    """
    Load model metrics and create visualizations
    """
    # Define the directory containing the models
    base_dir = "runs"
    models = ["base_svm", "base_knn", "base_cnn", "base_resnet", "base_vit"]

    # Define the folder to save images
    save_dir = "images"
    os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn't exist

    # Initialize storage for metrics
    metrics = {}

    # Iterate through models and load data
    for model in models:
        log_path = os.path.join(base_dir, model, "log.pkl")
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                metrics[model] = pickle.load(f)

    # Ensure all models have been processed
    if not metrics:
        print("No log.pkl files found in the specified directories.")
        return

    # Extract metrics for bar graphs and tables
    inference_times = []
    testing_accuracies = []
    f1_scores = []
    confusion_matrices = []
    simplified_metrics = []
    model_names = []

    # Process metrics for each model
    for model, data in metrics.items():
        model_names.append(model.replace("base_", "").upper())
        inference_times.append(data.get("inference_time", 0))
        testing_accuracies.append(data.get("test_accuracy", 0))
        f1_scores.append(data.get("f1_score", 0))

    # Display simplified confusion matrix metrics
    df_simplified = pd.DataFrame(simplified_metrics)
    print("\n=== Simplified Confusion Matrix Metrics ===")
    print(df_simplified)

    # Print testing metrics
    print("\n=== Model Testing Metrics ===")
    df_metrics = pd.DataFrame(
        {
            "Model": model_names,
            "F1 Score": [f"{score:.4f}" for score in f1_scores],
            "Testing Accuracy": [f"{acc:.4f}" for acc in testing_accuracies],
            "Inference Time (ms)": [f"{time * 1000:.4f}" for time in inference_times],
        }
    )
    print(df_metrics)

    # Plot training accuracy
    plot_models = ["base_cnn", "base_resnet", "base_vit"]
    for model in plot_models:
        if model in metrics:
            data = metrics[model]
            epochs = range(
                1, len(data.get("train_accuracy", [])) + 1
            )  # Epoch range from 1 to n_epochs

            # Plot training and validation accuracy
            plt.figure(figsize=(8, 5))
            plt.plot(
                epochs,
                data.get("train_accuracy", []),
                label="Training Accuracy",
                marker="o",
                linestyle="-",
            )
            plt.plot(
                epochs,
                data.get("val_accuracy", []),
                label="Validation Accuracy",
                marker="o",
                linestyle="--",
            )
            plt.title(
                f"Training vs Validation Accuracy: {model.replace('base_', '').upper()}",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel("Accuracy", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(visible=True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    save_dir, f"{model.replace('base_', '').lower()}_accuracy.png"
                ),
                dpi=300,
                format="png",
            )
            plt.savefig(
                os.path.join(
                    save_dir, f"{model.replace('base_', '').lower()}_accuracy.svg"
                ),
                format="svg",
            )
            plt.close()

            # Plot training and validation loss
            plt.figure(figsize=(8, 5))
            plt.plot(
                epochs,
                data.get("train_loss", []),
                label="Training Loss",
                marker="s",
                linestyle="-",
            )
            plt.plot(
                epochs,
                data.get("val_loss", []),
                label="Validation Loss",
                marker="s",
                linestyle="--",
            )
            plt.title(
                f"Training vs Validation Loss: {model.replace('base_', '').upper()}",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(visible=True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    save_dir, f"{model.replace('base_', '').lower()}_loss.png"
                ),
                dpi=300,
                format="png",
            )
            plt.savefig(
                os.path.join(
                    save_dir, f"{model.replace('base_', '').lower()}_loss.svg"
                ),
                format="svg",
            )
            plt.close()

    # Bar graph for inference time
    inference_times = [t * 1000 for t in inference_times]  # Convert to ms
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, inference_times, color="royalblue", edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.title("Inference Time (ms)", fontsize=14, fontweight="bold")
    plt.xlabel("Models", fontsize=12)
    plt.ylabel("Time (ms)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "inference_time.png"), dpi=300, format="png")
    plt.savefig(os.path.join(save_dir, "inference_time.svg"), format="svg")
    plt.close()

    # Bar graph for testing accuracy
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        model_names, testing_accuracies, color="forestgreen", edgecolor="black"
    )
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.title("Testing Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Models", fontsize=12)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)  # Ensure accuracy scale is 0-1
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "testing_accuracy.png"), dpi=300, format="png")
    plt.savefig(os.path.join(save_dir, "testing_accuracy.svg"), format="svg")
    plt.close()


def main():
    """
    Main function to execute the script
    """
    print("Starting to plot model metrics...")
    plot_metrics()

    print("\nScript execution completed.")


if __name__ == "__main__":
    main()
