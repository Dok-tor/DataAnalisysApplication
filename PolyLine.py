from OpenGL.GL import glColor3f, glBegin, GL_LINE_LOOP, glVertex, glEnd, GL_LINE_STRIP
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid


class Line:
    def __init__(self, start_point):
        self.points = []
        self.points.append(start_point)
        self.finished = False
        self.polygon = None

    def addPoint(self, point):
        self.points.append(point)

    def isFinished(self):
        return self.finished

    def finish(self):
        self.finished = True
        self.makePolygon()

    def draw(self):
        glColor3f(0.0, 1.0, 1.0)
        if self.finished:
            # Замкнутая
            glBegin(GL_LINE_LOOP)
        else:
            # Разомкнутая
            glBegin(GL_LINE_STRIP)

        for point in self.points:
            glVertex(point[0], point[1])
        glEnd()

    def move(self, x, y):
        if self.finished:
            self.points[:, 0] += x
            self.points[:, 1] += y

    def makePolygon(self):
        # polygon_points = np.array(self.points)
        self.points = np.array(self.points)
        # Если полигон это точка или линия (защита от ощибок в вычислениях)
        if self.points.shape[0] < 3:
            return False

        # Создание полигона из точек
        polygon = Polygon(self.points)

        # Создание точки

        # Если полигон невалиден (самопересечения) то исправляем его логику
        if not polygon.is_valid:
            polygon1 = make_valid(polygon)  # Хорошо обрабатывает внешние самопересечения
            polygon2 = polygon.buffer(0)  # Хорошо обрабатывает внутренние самопересечения
            polygon = unary_union([polygon1, polygon2])

        self.polygon = polygon

    def isPointInside(self, point):
        x, y = point

        p = Point(x, y)

        return self.polygon.contains(p)
