"""
Dataset utilities for the QSVM project.

Loads the Breast Cancer Wisconsin dataset,
performs preprocessing, and returns train/test splits.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_dataset(
    test_size: float = 0.2,
    n_components: int = 4,
    random_state: int = 42,
):
    """
    Load and preprocess the Breast Cancer dataset.

    Parameters
    ----------
    test_size : float
        Fraction of samples used for testing.

    n_components : int
        Number of PCA components (equals number of qubits).

    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    dataset = load_breast_cancer()

    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=n_components)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))

    X_train = angle_scaler.fit_transform(X_train)
    X_test = angle_scaler.transform(X_test)

    return X_train, X_test, y_train, y_test