"""
Quantum Support Vector Machine using a Fidelity Quantum Kernel.
"""

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute


class QuantumKernelSVM:

    def __init__(
        self,
        n_qubits: int = 4,
        reps: int = 2,
    ):

        feature_map = ZZFeatureMap(
            feature_dimension=n_qubits,
            entanglement="linear",
        )

        sampler = StatevectorSampler()

        fidelity = ComputeUncompute(sampler)

        self.kernel = FidelityQuantumKernel(
            fidelity=fidelity,
            feature_map=feature_map,
        )

        self.model = SVC(kernel="precomputed")

        self.X_train = None

    def fit(self, X_train, y_train):

        self.X_train = X_train

        train_kernel = self.kernel.evaluate(x_vec=X_train)

        self.model.fit(train_kernel, y_train)

    def predict(self, X_test):

        test_kernel = self.kernel.evaluate(
            x_vec=X_test,
            y_vec=self.X_train,
        )

        return self.model.predict(test_kernel)

    def accuracy(self, X_test, y_test):

        predictions = self.predict(X_test)

        return accuracy_score(y_test, predictions)

    def kernel_matrix(self, X):

        return self.kernel.evaluate(x_vec=X)