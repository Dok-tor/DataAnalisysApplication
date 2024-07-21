import numpy as np
from OpenGL.GL import *

class Ellipse:
    def __init__(self, start_point, sensitivity=1.0):
        self.start_point = start_point
        self.end_point = start_point  # Конечная точка для определения размеров эллипса
        self.sensitivity = sensitivity
        self.width = None
        self.height = None
        self.center_x = None
        self.center_y = None
        self.finished = False
        self.angle = 0  # Добавленный атрибут для угла поворота
        self.ellipse_rotation_matrix = None

    def updateEndPoint(self, end_point, scale):
        dx = (end_point[0] - self.start_point[0]) * self.sensitivity
        dy = (end_point[1] - self.start_point[1]) * self.sensitivity
        self.end_point = (self.start_point[0] + dx,
                          self.start_point[1] + dy)

    def calculateParams(self):
        self.width = abs(self.end_point[0] - self.start_point[0])
        self.height = abs(self.end_point[1] - self.start_point[1])
        self.center_x = (self.start_point[0] + self.end_point[0]) / 2
        self.center_y = (self.start_point[1] + self.end_point[1]) / 2

    def calculateAngle(self):
        ellipse_angle_rad = -np.radians(self.angle)

        # Матрица поворота для преобразования точек в систему координат эллипса
        cos_angle = np.cos(ellipse_angle_rad)
        sin_angle = np.sin(ellipse_angle_rad)
        self.ellipse_rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

    def increaseHeight(self):
        # self.height += 0.02
        self.height *= 1.05

    def decreaseHeight(self):
        # self.height -= 0.02
        self.height /= 1.05

    def increaseWight(self):
        self.width *= 1.05

    def decreaseWight(self):
        self.width /= 1.05

    def turnUp(self):
        self.center_y += 0.01

    def turnDown(self):
        self.center_y -= 0.01

    def turnLeft(self):
        self.center_x -= 0.01

    def turnRight(self):
        self.center_x += 0.01

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.calculateParams()
        self.calculateAngle()

    def rotate(self, angle_delta):
        self.angle -= angle_delta
        self.calculateAngle()

    def move(self, x, y):
        if self.finished:
            self.center_x += x
            self.center_y += y

    def isPointInside(self, x, y):
        # ellipse_angle_rad = -np.radians(self.angle)
        # cos_angle = np.cos(ellipse_angle_rad)
        # sin_angle = np.sin(ellipse_angle_rad)
        #
        # # Матрица поворота для преобразования точек в систему координат эллипса
        # cos_angle = np.cos(ellipse_angle_rad)
        # sin_angle = np.sin(ellipse_angle_rad)
        # ellipse_rotation_matrix = np.array([
        #     [cos_angle, -sin_angle],
        #     [sin_angle, cos_angle]
        # ])

        translated_point = np.array([x - self.center_x, y - self.center_y])
        transformed_point = np.dot(self.ellipse_rotation_matrix, translated_point)

        return (transformed_point[0] / (self.width / 2)) ** 2 + (transformed_point[1] / (self.height / 2)) ** 2 <= 1

    def draw(self):
        glColor3f(0.0, 1.0, 1.0)
        glBegin(GL_LINE_LOOP)
        if not self.finished:
            self.calculateParams()
        for i in range(360):
            theta = np.radians(i)
            x = self.center_x + (self.width / 2) * np.cos(theta) * np.cos(np.radians(self.angle)) - (
                        self.height / 2) * np.sin(theta) * np.sin(np.radians(self.angle))
            y = self.center_y + (self.width / 2) * np.cos(theta) * np.sin(np.radians(self.angle)) + (
                        self.height / 2) * np.sin(theta) * np.cos(np.radians(self.angle))
            glVertex2f(x, y)
        glEnd()