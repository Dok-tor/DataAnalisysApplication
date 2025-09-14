# import numpy as np
# import scipy.linalg
#
#
# try:
#     import cupy as cp
#     gpu_available = True
# except ImportError:
#     cp = np
#     gpu_available = False
#
#
# class Tour:
#     def __init__(self, outer_instance: object, dim: int, data: np.ndarray, labels: np.ndarray = None, labels_dict: dict = None, rotation_speed: float = 700) -> None:
#         self.outer_instance = outer_instance
#         self.dim = dim
#         self.data = cp.array(data) if gpu_available else np.array(data)
#         self.running = True
#         self.rotation_speed = rotation_speed
#         self.rotation_matrix = None
#         self.target_basis = None
#         self.current_basis = None
#         self.interpolation_step = 0
#         self.angle = 0
#
#         if labels_dict:
#             self.labels_dict = labels_dict
#         else:
#             self.labels_dict = dict()
#             self.labels_dict[0] = "No Cluster"
#
#         if labels is not None:
#             self.labels = cp.array(labels) if gpu_available else np.array(labels)
#         else:
#             self.labels = cp.zeros(data.shape[0], dtype=int) if gpu_available else np.zeros(data.shape[0], dtype=int)
#         if self.dim == 2:
#             z = 0
#             z_column = cp.full((self.data.shape[0], 1), z) if gpu_available else np.full((self.data.shape[0], 1), z)
#             self.data = cp.hstack((self.data, z_column)) if gpu_available else np.hstack((self.data, z_column))
#
#         self.normalized_data = self._normalize_data(self.data)
#         self.current_normalized_data = self._normalize_data(self.data)
#
#         self._initialize_tour()
#         self.update_projection()
#
#
#     def run(self):
#         self.running = True
#
#     def stop(self):
#         self.running = False
#
#     def getData(self):
#         return cp.asnumpy(self.data) if gpu_available else self.data
#
#     def getNormalizedData(self):
#         return cp.asnumpy(self.normalized_data) if gpu_available else self.normalized_data
#
#     def getCurrentNormalizedData(self):
#         return cp.asnumpy(self.current_normalized_data) if gpu_available else self.current_normalized_data
#
#     def getLabels(self):
#         return cp.asnumpy(self.labels) if gpu_available else self.labels
#
#     def addCluster(self, i, cluster_number):
#         if self.labels[i] == 0:
#             self.labels[i] = cluster_number
#             return True
#         return False
#
#     def getClusterLabel(self, cluster_number) -> str:
#         if self.labels_dict.get(cluster_number, 0):
#             return self.labels_dict[cluster_number]
#
#     def getLabelsDict(self):
#         return self.labels_dict
#
#     def addClusterLabel(self, cluster_number: int, label: str):
#         self.labels_dict[cluster_number] = label
#
#     def deleteCluster(self, cluster_number):
#         for i in range(len(self.labels)):
#             if self.labels[i] == cluster_number:
#                 self.labels[i] = 0
#
#     @staticmethod
#     def _normalize_data(data):
#         min_coords = np.min(data, axis=0)
#         max_coords = np.max(data, axis=0)
#
#         center = (max_coords + min_coords) / 2
#         range_max = np.max(max_coords - min_coords)
#
#         normalized_points_array = (data - center) / (range_max / 2)
#
#         return normalized_points_array
#
#     # @staticmethod
#     # def _normalize_data(data):
#     #     min_val = cp.min(data, axis=0) if gpu_available else np.min(data, axis=0)
#     #     max_val = cp.max(data, axis=0) if gpu_available else np.max(data, axis=0)
#     #     range_val = max_val - min_val
#     #     normalized_data = 2 * (data - min_val) / range_val - 1
#     #     return normalized_data
#
#     def _initialize_tour(self):
#         self.current_basis = self._generate_random_basis()
#         self.target_basis = self._generate_random_basis()
#         self.interpolation_step = 0
#         self.angle = self._calculate_angle()
#
#     def _generate_random_basis(self):
#         basis = cp.random.randn(self.data.shape[1], 3) if gpu_available else np.random.randn(self.data.shape[1], 3)
#         basis, _ = cp.linalg.qr(basis) if gpu_available else np.linalg.qr(basis)
#         return basis
#
#     # def _generate_random_basis(self):
#     #     basis = cp.random.randn(self.data.shape[1], 3) if gpu_available else np.random.rand(self.data.shape[1], 3)
#     #     basis -= 0.5  # Centering around zero
#     #     basis, _ = cp.linalg.qr(basis) if gpu_available else np.linalg.qr(basis)
#     #     return basis
#
#     # def _calculate_angle(self):
#     #     vector = self.normalized_data[0]
#     #     current_direction = np.dot(vector, self.current_basis)
#     #     next_direction = np.dot(vector, self.target_basis)
#     #     u = current_direction
#     #     v = next_direction
#     #     if np.dot(u, v) == 0:
#     #         return np.pi / 2
#     #
#     #     c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)  # -> cosine of the angle
#     #     angle = np.arccos(np.clip(c, -1, 1))
#     #
#     #     return angle
#
#     def _calculate_angle(self):
#         return np.linalg.norm(self.current_basis - self.target_basis, ord='fro')
#
#     @staticmethod
#     def logarithm_map_interpolation(R1, R2, t):
#         """
#         Выполняет интерполяцию Logarithm Map между двумя матрицами вращения.
#
#         :param R1: Первая матрица вращения размером N x N.
#         :param R2: Вторая матрица вращения размером N x N.
#         :param t: Шаг интерполяции, число от 0 до 1.
#         :return: Промежуточная матрица вращения размером N x N.
#         """
#         # Шаг 1: Найти относительное вращение между R1 и R2
#         R_relative = np.dot(np.linalg.inv(R1), R2)
#
#         # Шаг 2: Взять логарифм от относительного вращения
#         log_R_relative = scipy.linalg.logm(R_relative)
#
#         # Шаг 3: Умножить логарифм на шаг интерполяции t
#         t_log_R_relative = t * log_R_relative
#
#         # Шаг 4: Экспоненцировать результат
#         R_interpolated = np.dot(R1, scipy.linalg.expm(t_log_R_relative))
#
#         return R_interpolated
#
#     def _interpolate_basis(self):
#         if self.interpolation_step > 1:
#             self.current_basis = self.target_basis
#             self.target_basis = self._generate_random_basis()
#             self.interpolation_step = 0
#             self.angle = self._calculate_angle()
#             # print("_____________________________________________________________________________")
#
#         # Calculate step size based on the angle and desired rotation speed
#         # adaptive_speed = 1.0 / (self.rotation_speed * self.angle + 1e-9)  # Adding a small constant to avoid division by zero
#         # a = 0.1
#         var = -0.01
#         a = 0.0169
#         b = 0.6
#         c = 0.003
#         t = 1
#         # d = 0.3337
#         adaptive_speed = a / (t * self.angle + b) - c
#         # adaptive_speed = 0.0018
#         # adaptive_speed = a * np.log(b * self.angle + 1) + c
#         self.outer_instance.setIndicateString(f'speed: {adaptive_speed:.4f}\t angle: {self.angle:.4f}')
#
#         self.interpolation_step += adaptive_speed
#         t = self.interpolation_step
#
#         # interpolated_basis = (1 - t) * self.current_basis + t * self.target_basis
#         # interpolated_basis, _ = cp.linalg.qr(interpolated_basis) if gpu_available else np.linalg.qr(interpolated_basis)
#         # print(interpolated_basis)
#         interpolated_basis = self.logarithm_map_interpolation(self.current_basis, self.target_basis, t)
#
#         return interpolated_basis
#
#     def update_projection(self):
#         interpolated_basis = self._interpolate_basis()
#         projected_data = cp.dot(self.normalized_data, interpolated_basis) if gpu_available else np.dot(self.normalized_data, interpolated_basis)
#
#         self.current_normalized_data = projected_data


import numpy as np
from typing import TYPE_CHECKING
import scipy.linalg

if TYPE_CHECKING:
    from main import MainWindow

# Предварительно скомпилированная самописная библиотека
# Если в редакторе отображается с ошибкой, то так и должно быть, он может её не находить
import my_custom_interpolation as interp  # type: ignore


class Tour:
    def __init__(self, outer_instance: 'MainWindow', dim: int, data: np.ndarray, labels: np.ndarray = None, labels_dict: dict = None, rotation_speed: float = 0.0) -> None:
        self.outer_instance = outer_instance
        self.dim = dim
        self.data = np.array(data)
        self.running = True
        self.rotation_speed = rotation_speed
        self.rotation_matrix = None
        self.target_basis = None
        self.current_basis = None
        self.interpolation_step = 0
        self.angle = 0
        self.adaptive_speed = 0.0018

        self._last_interpolated_basis = None  # <— добавим

        self.border_dim = 50  # Экспериментально найденное значение

        if labels_dict:
            self.labels_dict = labels_dict
        else:
            self.labels_dict = dict()
            self.labels_dict[0] = "No Cluster"

        if labels is not None:
            self.labels = np.array(labels)
        else:
            self.labels = np.zeros(data.shape[0], dtype=int)
        if self.dim == 2:
            z = 0
            z_column = np.full((self.data.shape[0], 1), z)
            self.data = np.hstack((self.data, z_column))

        self.normalized_data = self._normalize_data(self.data)
        self.current_normalized_data = self._normalize_data(self.data)

        self._initialize_tour()
        self.update_projection()

    def setRotationSpeed(self, speed: float) -> None:
        self.rotation_speed = speed

    def run(self):
        self.running = True

    def stop(self):
        self.running = False

    def getData(self):
        return self.data

    def getDim(self):
        return self.dim

    def getNormalizedData(self):
        return self.normalized_data

    def getCurrentNormalizedData(self):
        return self.current_normalized_data

    def getLabels(self):
        return self.labels

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
        min_coords = np.min(data, axis=0)
        max_coords = np.max(data, axis=0)

        center = (max_coords + min_coords) / 2
        range_max = np.max(max_coords - min_coords)

        normalized_points_array = (data - center) / (range_max / 2)

        return normalized_points_array

    # @staticmethod
    # def _normalize_data(data):
    #     min_val = cp.min(data, axis=0) if gpu_available else np.min(data, axis=0)
    #     max_val = cp.max(data, axis=0) if gpu_available else np.max(data, axis=0)
    #     range_val = max_val - min_val
    #     normalized_data = 2 * (data - min_val) / range_val - 1
    #     return normalized_data

    def _initialize_tour(self):
        self.current_basis = self._generate_random_basis()
        self.target_basis = self._generate_random_basis()
        self.interpolation_step = 0
        # self.angle = interp.calculate_angle(self.current_basis, self.target_basis)
        self.angle = self._calculate_angle()

    def _generate_random_basis(self):
        if self.dim <= self.border_dim:
            basis = np.random.randn(self.data.shape[1], self.data.shape[1])
        else:
            basis = np.random.rand(self.data.shape[1], 3)
        basis -= 0.5  # Centering around zero
        basis, _ = np.linalg.qr(basis)
        return basis

    # def _generate_random_basis_for_low_dim(self):
    #     """ <= 50"""
    #     basis = np.random.randn(self.data.shape[1], self.data.shape[1])
    #     basis, _ = np.linalg.qr(basis)
    #     return basis
    #
    # def _generate_random_basis_for_high_dim(self):
    #     """ >= 50"""
    #     basis = np.random.rand(self.data.shape[1], 3)
    #     basis -= 0.5  # Centering around zero
    #     basis, _ = np.linalg.qr(basis)
    #     return basis

    # def _calculate_angle(self):
    #     vector = self.normalized_data[0]
    #     current_direction = np.dot(vector, self.current_basis)
    #     next_direction = np.dot(vector, self.target_basis)
    #     u = current_direction
    #     v = next_direction
    #     if np.dot(u, v) == 0:
    #         return np.pi / 2
    #
    #     c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)  # -> cosine of the angle
    #     angle = np.arccos(np.clip(c, -1, 1))
    #
    #     return angle

    def _calculate_angle(self):
        angle = np.linalg.norm(self.current_basis - self.target_basis, ord='fro')
        if angle > np.pi:
            angle = np.pi
        return angle

    # @staticmethod
    # def logarithm_map_interpolation(R1, R2, t):
    #     """
    #     Выполняет интерполяцию Logarithm Map между двумя матрицами вращения.
    #
    #     :param R1: Первая матрица вращения размером N x N.
    #     :param R2: Вторая матрица вращения размером N x N.
    #     :param t: Шаг интерполяции, число от 0 до 1.
    #     :return: Промежуточная матрица вращения размером N x N.
    #     """
    #     # Шаг 1: Найти относительное вращение между R1 и R2
    #     R_relative = np.dot(np.linalg.inv(R1), R2)
    #
    #     # Шаг 2: Взять логарифм от относительного вращения
    #     log_R_relative = scipy.linalg.logm(R_relative)
    #
    #     # Шаг 3: Умножить логарифм на шаг интерполяции t
    #     t_log_R_relative = t * log_R_relative
    #
    #     # Шаг 4: Экспоненцировать результат
    #     R_interpolated = np.dot(R1, scipy.linalg.expm(t_log_R_relative))
    #
    #     return R_interpolated

    def calculateAdaptiveSpeed(self):
        a = 0.0169
        b = 0.6
        c = 0.003
        t = 1

        self.adaptive_speed = a / (t * self.angle + b) - c + self.rotation_speed  # поднимаем или опускаем кривую
        if self.adaptive_speed < 0:
            self.adaptive_speed = 0.0001

    def _interpolate_basis(self):
        if self.interpolation_step > 1:
            self.current_basis = self.target_basis
            self.target_basis = self._generate_random_basis()
            self.interpolation_step = 0
            # self.angle = interp.calculate_angle(self.current_basis, self.target_basis)
            self.angle = self._calculate_angle()

        self.calculateAdaptiveSpeed()

        # Обращение к родительскому классу

        self.outer_instance.setIndicateString(f'speed: {self.adaptive_speed:.4f}\t angle: {self.angle:.4f}')

        self.interpolation_step += self.adaptive_speed
        t = self.interpolation_step

        if self.dim <= self.border_dim:
            interpolated_basis = interp.logarithm_map_interpolation(self.current_basis, self.target_basis, t)
        else:
            # Размерность слишком высокая для таких сложных вычислений, выбираем упрощённый вариант
            interpolated_basis = (1 - t) * self.current_basis + t * self.target_basis
            interpolated_basis, _ = np.linalg.qr(interpolated_basis)

        return interpolated_basis

    def update_projection(self):
        interpolated_basis = self._interpolate_basis()
        self._last_interpolated_basis = interpolated_basis  # <— запомним

        if self.dim <= 2:
            self.current_normalized_data = self.normalized_data
            return
        elif self.dim <= self.border_dim:
            projected_data = interp.rotate_to_3d(self.normalized_data, interpolated_basis, self.data.shape[1])
        else:
            projected_data = np.dot(self.normalized_data, interpolated_basis)
        self.current_normalized_data = projected_data
        # if self.dim <= 2:
        #     self.current_normalized_data = self.normalized_data


    def getProjectedAxes3D(self) -> np.ndarray:
        """
        Возвращает массив shape: (self.dim, 3), где каждая строка — это образ e_i
        (единичной оси исходного d-мерного пространства) в текущем 3D-представлении Grand Tour.
        Важно: возвращаем ровно self.dim осей (без добавленного искусственного Z при dim==2).
        """
        d = self.dim
        Dcols = self.data.shape[1]  # число столбцов данных (учитывает добавленный Z при dim==2)
        if d <= 0:
            return np.zeros((0, 3), dtype=float)

        # Базис e_i исходного пространства: берём I_d и при необходимости дорисовываем нулевой Z-столбец,
        # чтобы соответствовать числу столбцов self.data (которое может быть 3 при dim==2).
        # Формируем "данные" из d строк-образцов: X_axes shape (d, Dcols)
        X_axes = np.zeros((d, Dcols), dtype=float)
        X_axes[:, :d] = np.eye(d, dtype=float)  # первые d столбцов — единичные e_i

        B = self._last_interpolated_basis
        if d <= 2:
            # При dim<=2 данные уже рисуются как есть (с добавленным Z=0), так что образ e_i — это e_i в XY-плоскости
            # (третья компонента — 0). Возвращаем первые d осей как 3D-векторы (X,Y,0).
            axes3d = np.zeros((d, 3), dtype=float)
            axes3d[:, :d] = np.eye(d, dtype=float)
            return axes3d

        if B is None:
            # на всякий случай — если ещё не успели посчитать проекцию
            # вернём нули, чтобы вызывающая сторона могла просто ничего не рисовать
            return np.zeros((d, 3), dtype=float)

        if d <= self.border_dim:
            # Используем тот же путь, что и для данных
            # (через вашу быструю rotate_to_3d), прогоняем все e_i за один вызов
            axes3d_full = interp.rotate_to_3d(X_axes, B, Dcols)  # shape: (d, 3)
            return axes3d_full
        else:
            # Высокая размерность: данные проецируются как X @ B, где B shape (Dcols, 3)
            # Для e_i: I_d @ B => берём первые d строк B (т.к. X_axes имеет единицы в первых d столбцах)
            # Если Dcols==d, это просто B.
            return (X_axes @ B)  # shape: (d, 3)