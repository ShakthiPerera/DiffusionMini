from scipy.stats import kurtosis
import torch
import matplotlib.pyplot as plt
import numpy as np


def kurtosis_(data):
    kurtosis_ = kurtosis(data, fisher=True)
    return kurtosis_


def isotropy(data):
    iso = torch.trace((data.T @ data)) / data.size(dim=0)
    return iso


def kurtosis_plot(metric_values, metric_name, ylim, save_name):
    plt.figure(figsize=(8, 6))

    # Define the window size for the moving average
    window_size = 10

    # Compute the moving average
    smoothed_metric = np.convolve(
        metric_values, np.ones(window_size) / window_size, mode="same"
    )
    smoothed_metric[:window_size] = metric_values[:window_size]
    plt.scatter(range(1000), smoothed_metric, label=f"{metric_name}", s=1)
    plt.axhline(y=0, color="red", linestyle="-")  # Add a red line along y = 0

    # Label the region above the x-axis as "Super Gaussian" and below the x-axis as "Sub Gaussian"
    plt.text(
        900,
        1.5,
        "Super Gaussian",
        horizontalalignment="center",
        color="black",
        fontsize=12,
    )
    plt.text(
        900,
        -1.5,
        "Sub Gaussian",
        horizontalalignment="center",
        color="black",
        fontsize=12,
    )

    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel(f"{metric_name} Values")
    plt.title(f"{metric_name} Plot ")
    if not ylim is None:
        plt.ylim(*ylim)
    plt.grid(True)
    plt.savefig(f"{path}/{save_name}.png")
