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
        self.finished = True

    def rotate(self, angle_delta):
        self.angle -= angle_delta

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