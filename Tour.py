import numpy as np

try:
    import cupy as cp
    gpu_available = True
except ImportError:
    cp = np
    gpu_available = False


class Tour:
    def __init__(self, dim: int, data: np.ndarray, labels: np.ndarray = None, labels_dict: dict = None, rotation_speed: float = 0.001) -> None:
        self.dim = dim
        self.data = cp.array(data) if gpu_available else np.array(data)
        self.running = True
        self.rotation_speed = rotation_speed
        self.rotation_matrix = None
        self.target_basis = None
        self.current_basis = None
        self.interpolation_step = 0

        if labels_dict:
            self.labels_dict = labels_dict
        else:
            self.labels_dict = dict()
            self.labels_dict[0] = "No Cluster"

        if labels is not None:
            self.labels = cp.array(labels) if gpu_available else np.array(labels)
        else:
            self.labels = cp.zeros(data.shape[0], dtype=int) if gpu_available else np.zeros(data.shape[0], dtype=int)
        if self.dim == 2:
            z = 0
            z_column = cp.full((self.data.shape[0], 1), z) if gpu_available else np.full((self.data.shape[0], 1), z)
            self.data = cp.hstack((self.data, z_column)) if gpu_available else np.hstack((self.data, z_column))

        self._initialize_tour()
        self.update_projection()

        self.normalized_data = self._normalize_data(self.data)

    def run(self):
        self.running = True

    def stop(self):
        self.running = False

    def getData(self):
        return cp.asnumpy(self.data) if gpu_available else self.data

    def getNormalizedData(self):
        return cp.asnumpy(self.normalized_data) if gpu_available else self.normalized_data

    def getLabels(self):
        return cp.asnumpy(self.labels) if gpu_available else self.labels

    def addCluster(self, i, cluster_number):
        if self.labels[i] == 0:
            self.labels[i] = cluster_number
            return True
        return False

    def getClusterLabel(self, cluster_number) -> str:
        if self.labels_dict.get(cluster_number, 0):
            return self.labels_dict[cluster_number]

    def getLabelsDict(self):
        return self.labels_dict

    def addClusterLabel(self, cluster_number: int, label: str):
        self.labels_dict[cluster_number] = label

    def deleteCluster(self, cluster_number):
        for i in range(len(self.labels)):
            if self.labels[i] == cluster_number:
                self.labels[i] = 0

    @staticmethod
    def _normalize_data(data):
        min_val = cp.min(data, axis=0) if gpu_available else np.min(data, axis=0)
        max_val = cp.max(data, axis=0) if gpu_available else np.max(data, axis=0)
        range_val = max_val - min_val
        normalized_data = 2 * (data - min_val) / range_val - 1
        return normalized_data

    def _initialize_tour(self):
        self.current_basis = self._generate_random_basis()
        self.target_basis = self._generate_random_basis()
        self.interpolation_step = 0

    def _generate_random_basis(self):
        basis = cp.random.randn(self.data.shape[1], 3) if gpu_available else np.random.randn(self.data.shape[1], 3)
        basis, _ = cp.linalg.qr(basis) if gpu_available else np.linalg.qr(basis)
        return basis

    def _interpolate_basis(self):
        if self.interpolation_step >= 1:
            self.current_basis = self.target_basis
            self.target_basis = self._generate_random_basis()
            self.interpolation_step = 0

        self.interpolation_step += self.rotation_speed
        t = self.interpolation_step

        interpolated_basis = (1 - t) * self.current_basis + t * self.target_basis

        interpolated_basis, _ = cp.linalg.qr(interpolated_basis) if gpu_available else np.linalg.qr(interpolated_basis)

        return interpolated_basis

    def update_projection(self):
        interpolated_basis = self._interpolate_basis()
        projected_data = cp.dot(self.data, interpolated_basis) if gpu_available else np.dot(self.data, interpolated_basis)
        self.normalized_data = self._normalize_data(projected_data)