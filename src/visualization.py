"""
Visualization utilities for the QSVM project.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)


FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


def save_confusion_matrix(
    y_true,
    y_pred,
    title,
    filename,
):
    """
    Save a confusion matrix plot.
    """

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    ax.set_title(title)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
    )

    plt.close()

def plot_dataset(
    X,
    y,
    filename="dataset.png",
):
    """
    Plot the PCA-reduced dataset.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="k",
        alpha=0.8,
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Breast Cancer Dataset (PCA Projection)")

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Class"
    )

    ax.add_artist(legend)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
    )

    plt.close()
def plot_kernel_matrix(
    kernel_matrix,
    filename="kernel_matrix.png",
):
    """
    Plot the quantum kernel matrix.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    image = ax.imshow(
        kernel_matrix,
        cmap="viridis",
    )

    ax.set_title("Quantum Kernel Matrix")

    plt.colorbar(
        image,
        ax=ax,
        label="Kernel Value",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
    )

    plt.close()
def plot_accuracy(
    classical_accuracy,
    quantum_accuracy,
    filename="accuracy_comparison.png",
):
    """
    Compare classical and quantum accuracies.
    """

    fig, ax = plt.subplots(figsize=(6, 5))

    models = [
        "Classical\nRBF SVM",
        "Quantum\nSVM",
    ]

    scores = [
        classical_accuracy,
        quantum_accuracy,
    ]

    bars = ax.bar(
        models,
        scores,
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel("Accuracy")

    ax.set_title("Classification Accuracy")

    for bar, score in zip(bars, scores):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.01,
            f"{score:.3f}",
            ha="center",
        )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
    )

    plt.close()


def plot_feature_map(
    n_qubits: int = 4,
    reps: int = 2,
    filename: str = "feature_map.png",
):
    """
    Generate and save the ZZFeatureMap circuit used by the QSVM.
    """

    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
    )

    fig = feature_map.decompose().draw(
        output="mpl",
        fold=-1,
        idle_wires=False,
    )

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    fig.clf()