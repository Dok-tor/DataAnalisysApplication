import numpy as np


def matrixToString(matrix: np.ndarray, labels_dict: dict = None) -> str:
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
            if j == m - 1:
                if labels_dict:
                    # Подразумевается, что если словарь есть, значит последнее число в строке это метка класса
                    # Значит сделаем её целочисленной
                    string += str(int(matrix[i][j]))
                else:
                    string += str(matrix[i][j])  # Попытка сделать функцию универсальной, чтобы она печатала как обычные матрицы так и специальные
            else:
                string += str(matrix[i][j])
                string += '\t\t'
        if labels_dict:
            string += f"\t\"{labels_dict[matrix[i][-1]]}\""
        string += "\n"
    return string


class DataSaver:
    def __init__(self, points: np.ndarray, labels: np.ndarray, original_dim: int, labels_dict: dict = None) -> None:
        self.points = points
        self.labels = labels
        self.dim = original_dim
        if self.dim == 2:
            # Если оригинальная размерность была 2 значит мы добавили ещё один столбец, он мешает вычислению статистики
            # следовательно его нужно убрать
            self.points = points[:, :-1]
        else:
            self.points = points
        self.labels_dict = labels_dict

    def saveStatistics(self, path_to_stat_file: str) -> bool:
        with open(path_to_stat_file, 'w') as f:
            if not len(self.points):
                return False

            dim = len(self.points[0])
            n = len(self.points)
            f.write(f'points: {n}\tdim: {dim}\n')
            f.write('\n')

            cluster_labels, cluster_means, cluster_covariances, cluster_correlations = self.calculateStatistics()

            for i, label in enumerate(cluster_labels):
                f.write(f"For Cluster number: {label}, name: {self.labels_dict.get(label, "")}\n")
                f.write('\n')

                f.write(matrixToString(cluster_means[i]))
                f.write('\n')

                f.write("\tCovariance matrix\n")
                f.write(matrixToString(cluster_covariances[i]))
                f.write('\n')

                f.write("\tCorrelation matrix\n")
                f.write(matrixToString(cluster_correlations[i]))
                f.write('\n')
        return True

    def saveLabels(self, path_to_label_file: str) -> bool:
        if not path_to_label_file or not len(self.points) or not len(self.labels):
            return False

        dim = len(self.points[0])
        n = len(self.points)

        with open(path_to_label_file, 'w') as f:
            f.write(f'{n} {dim} {len(np.unique(self.labels))}\n')

            labels_reshaped = self.labels[:, np.newaxis]
            points_and_labels = np.hstack((self.points, labels_reshaped))
            f.write(f'{matrixToString(np.round(points_and_labels, 6), self.labels_dict)}\n')
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
