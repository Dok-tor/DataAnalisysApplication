import numpy as np


def matrixToString(matrix: np.ndarray) -> str:
    if not len(matrix):
        return ""

    # Проверка на одномерный массив
    if matrix.ndim == 1:
        return '\t' + '\t'.join(map(str, matrix)) + '\n'

    n = len(matrix)
    m = len(matrix[0])
    string = ""
    for i in range(n):
        string += '\t'
        for j in range(m):
            string += str(matrix[i][j])
            if not j == m - 1:
                string += '\t'
        string += "\n"
    return string


class DataSaver:
    def __init__(self, points: np.ndarray, labels: np.ndarray) -> None:
        self.points = points
        self.labels = labels

    def saveStatistics(self, path_to_stat_file: str) -> bool:
        with open(path_to_stat_file, 'w') as f:
            if not len(self.points):
                return False

            dim = len(self.points[0])
            n = len(self.points)
            f.write(f'points: {n}\tdim: {dim}\n')

            cluster_labels, cluster_means, cluster_covariances, cluster_correlations = self.calculateStatistics()

            for i, label in enumerate(cluster_labels):
                f.write(f"For Cluster {label}\n")
                f.write('\n')

                f.write(matrixToString(cluster_means[i]))
                f.write('\n')

                f.write("Covariance matrix\n")
                f.write(matrixToString(cluster_covariances[i]))
                f.write('\n')

                f.write("Correlation matrix\n")
                f.write(matrixToString(cluster_correlations[i]))
                f.write('\n')
        return True

    def saveLabels(self, path_to_label_file: str) -> bool:
        if not path_to_label_file or not len(self.points) or not len(self.labels):
            return False

        dim = len(self.points[0])
        n = len(self.points)

        with open(path_to_label_file, 'w') as f:
            f.write(f'{n} {dim}\n')

            labels_reshaped = self.labels[:, np.newaxis]
            points_and_labels = np.hstack((self.points, labels_reshaped))
            f.write(f'{matrixToString(np.round(points_and_labels, 6))}\n')
        return True

    def calculateStatistics(self):
        unique_labels = np.sort(np.unique(self.labels))
        cluster_labels = []
        cluster_means = []
        cluster_covariances = []
        cluster_correlations = []

        for label in unique_labels:
            cluster_labels.append(label)
            cluster_points = self.points[self.labels == label]
            mean = np.mean(cluster_points, axis=0)
            covariance = np.cov(cluster_points, rowvar=False)
            correlation = np.corrcoef(cluster_points, rowvar=False)

            cluster_means.append(mean)
            cluster_covariances.append(covariance)
            cluster_correlations.append(correlation)

        cluster_means = np.round(np.array(cluster_means), 6)
        cluster_covariances = np.round(np.array(cluster_covariances), 6)
        cluster_correlations = np.round(np.array(cluster_correlations), 6)

        return cluster_labels, cluster_means, cluster_covariances, cluster_correlations
