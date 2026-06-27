from src.dataset import load_dataset
from src.classical import ClassicalSVM
from src.quantum import QuantumKernelSVM
from src.visualization import (
    plot_accuracy,
    plot_dataset,
    plot_kernel_matrix,
    save_confusion_matrix,
    plot_feature_map
)

def main():
    print("Loading dataset...")

    X_train, X_test, y_train, y_test = load_dataset()

    print("Training classical SVM...")
    classical = ClassicalSVM()
    classical.fit(X_train, y_train)

    print("Training quantum SVM...")
    quantum = QuantumKernelSVM()
    quantum.fit(X_train, y_train)

    classical_acc = classical.accuracy(X_test, y_test)
    quantum_acc = quantum.accuracy(X_test, y_test)

    print("=" * 40)
    print("Results")
    print("=" * 40)
    print(f"Classical SVM Accuracy : {classical_acc:.4f}")
    print(f"Quantum SVM Accuracy   : {quantum_acc:.4f}")
    
    # Predictions
    classical_pred = classical.predict(X_test)
    quantum_pred = quantum.predict(X_test)

    # Confusion matrices
    save_confusion_matrix(
        y_test,
        classical_pred,
        "Classical SVM",
        "confusion_classical.png",
    )

    save_confusion_matrix(
        y_test,
        quantum_pred,
        "Quantum SVM",
        "confusion_quantum.png",
    )

    # PCA projection
    plot_dataset(
        X_train,
        y_train,
    )

    # Kernel heatmap
    plot_kernel_matrix(
        quantum.kernel_matrix(X_train)
    )

    # Accuracy comparison
    plot_accuracy(
        classical_acc,
        quantum_acc,
    )
    plot_feature_map()


if __name__ == "__main__":
    main()