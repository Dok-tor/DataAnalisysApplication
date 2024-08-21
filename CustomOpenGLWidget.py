import numpy as np
from OpenGL.GL import *
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import QOpenGLWidget
from scipy.spatial.transform import Rotation as R
from PyQt5.QtCore import QTimer, Qt
from enum import Enum


from PolyLine import Line
from Tour import Tour
from Ellipse import Ellipse

cluster_colors = [
    ('Red', (1.0, 0.0, 0.0)),
    ('Green', (0.0, 1.0, 0.0)),
    ('Blue', (0.0, 0.0, 1.0)),
    ('Yellow', (1.0, 1.0, 0.0)),
    ('Cyan', (0.0, 1.0, 1.0)),
    ('Magenta', (1.0, 0.0, 1.0)),
    ('Orange', (1.0, 0.647, 0.0)),
    ('Purple', (0.502, 0.0, 0.502)),
    ('Lime', (0.0, 1.0, 0.0)),
    ('Pink', (1.0, 0.753, 0.796)),
    ('Teal', (0.0, 0.502, 0.502)),
    ('Lavender', (0.902, 0.902, 0.980)),
    ('Brown', (0.647, 0.165, 0.165)),
    ('Maroon', (0.502, 0.0, 0.0)),
    ('Olive', (0.502, 0.502, 0.0)),
    ('Navy', (0.0, 0.0, 0.502)),
    ('Gold', (1.0, 0.843, 0.0)),
    ('Coral', (1.0, 0.498, 0.314)),
    ('Turquoise', (0.251, 0.878, 0.816)),
    ('SlateGray', (0.439, 0.502, 0.565)),
]


class State(Enum):
    MOVE = 0
    MANIPULATE = 1
    DRAW = 2


class SplineType(Enum):
    ELLIPSE = 0
    POLYLINE = 1


class VisualizeDataWidget(QOpenGLWidget):
    def __init__(self, outer_instance):
        super().__init__()

        self.aspect = self.width() / self.height()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateScene)
        self.timer.start(16)  # обновление каждые 16 мс (~60 FPS)
        self.tour = None
        self.size = 1
        self.scale = 0.7
        self.rotation = R.from_quat([0, 0, 0, 1])  # Начальное вращение (единичный кватернион)
        self.last_mouse_position = None
        self.mode = State.MOVE  # Режим по умолчанию - перемещение
        self.ellipse = None
        self.line = None
        self.splineType = SplineType.ELLIPSE
        self.is_drawing_moving_now = False

        self.outer_instance = outer_instance

    def setMode(self, mode):
        self.mode = mode

    def updateScene(self):
        if self.tour and self.mode == State.MOVE:
            self.tour.update_projection()
        self.update()

    def setTour(self, tour: Tour) -> None:
        """Здесь tour передаётся по ссылке, то есть после присвоения его можно менять снаружи и он будет
         меняться внутри"""
        self.tour = tour

    def setDefaultState(self):
        self.aspect = self.width() / self.height()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateScene)
        self.timer.start(16)  # обновление каждые 16 мс (~60 FPS)
        self.tour = None
        self.size = 1
        self.scale = 0.7
        self.rotation = R.from_quat([0, 0, 0, 1])  # Начальное вращение (единичный кватернион)
        self.last_mouse_position = None
        self.mode = State.MOVE  # Режим по умолчанию - перемещение
        self.ellipse = None
        self.line = None
        self.splineType = SplineType.ELLIPSE
        self.is_drawing_moving_now = False

    def setSize(self, size) -> None:
        self.size = size

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        self.resetView()
        self.updateProjection()

    def resizeGL(self, w, h):
        self.updateProjection()

    # Функция отрисовки всего на виджете
    def paintGL(self):
        if self.tour:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Установка камеры
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # отрисовка осей
            self.drawAxes()

            # отрисовка точек
            self.drawPoints()

        # отрисовка эллипса (только в draw)
        if self.ellipse:
            self.ellipse.draw()

        # отрисовка замкнутой линии (только в draw)
        if self.line:
            self.line.draw()

    def drawPoints(self):

        glPointSize(self.size)

        # сохраняем текущую матрицу состояний
        glPushMatrix()

        # масштабирование (приближение/отдалениек)
        glScalef(self.scale, self.scale, self.scale)

        # поворот точек
        rotation_matrix = self.rotation.as_matrix()

        glBegin(GL_POINTS)

        normalized_data = self.tour.getCurrentNormalizedData()
        rotated_points = np.dot(rotation_matrix, normalized_data.T).T
        for i, rotated_point in enumerate(rotated_points):
            # rotated_point = np.dot(rotation_matrix, point)
            if self.tour.labels[i] == 0:
                glColor3f(255, 255, 255)
            else:
                glColor3f(*cluster_colors[self.tour.labels[i] % len(cluster_colors) - 1][1])
            glVertex3f(*rotated_point)
        glEnd()
        glPopMatrix()

    def resetView(self):
        self.rotation = R.from_quat([0, 0, 0, 1])  # Сброс вращения на единичный кватернион

    def updateProjection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        w = self.width()
        h = self.height()
        glViewport(0, 0, w, h)

        self.aspect = w / h
        start_scale = 1

        if self.aspect >= 1.0:
            glOrtho(-self.aspect * start_scale, self.aspect * start_scale, -start_scale, start_scale,
                    -10 * start_scale, 10 * start_scale)
        else:
            glOrtho(-start_scale, start_scale, -start_scale / self.aspect, start_scale / self.aspect,
                    -10 * start_scale, 10 * start_scale)
        self.update()

    def zoomIn(self):
        self.scale /= 1.1  # Уменьшаем радиус для увеличения масштаба
        self.update()

    def zoomOut(self):
        self.scale *= 1.1  # Увеличиваем радиус для уменьшения масштаба
        self.update()

    def mousePressEvent(self, event):
        # Если состояние движения
        if self.mode == State.MOVE:
            if event.button() == Qt.LeftButton:
                if self.tour:
                    # TODO остановка нужно оттестировать взаимодействие с другими частями программы
                    self.tour.stop()
                    self.mode = State.MANIPULATE

        # Если состояние манипулирования (вращения)
        if self.mode == State.MANIPULATE:
            if event.button() == Qt.LeftButton:
                self.last_mouse_position = event.pos()
        # if event.button() == Qt.LeftButton and self.mode == State.MANIPULATE:

        # Если состояние рисование объектов
        elif self.mode == State.DRAW:
            if event.button() == Qt.LeftButton:

                pos = event.pos()
                ogl_x, ogl_y = self.getCurrentCoords(pos)

                if self.splineType == SplineType.ELLIPSE:
                    # Начинаем рисовать эллипс
                    self.ellipse = Ellipse((ogl_x, ogl_y), 1)

                elif self.splineType == SplineType.POLYLINE:
                    # Начинаем рисовать линию
                    self.line = Line((ogl_x, ogl_y))
            elif event.button() == Qt.RightButton:
                pos = event.pos()
                ogl_x, ogl_y = self.getCurrentCoords(pos)
                # Начинаем рисовать эллипс
                self.last_mouse_position = event.pos()
                if self.splineType == SplineType.ELLIPSE and self.ellipse:
                    if self.ellipse.isPointInside(ogl_x, ogl_y):
                        self.is_drawing_moving_now = True
                elif self.splineType == SplineType.POLYLINE and self.line:
                    if self.line.isPointInside((ogl_x, ogl_y)):
                        self.is_drawing_moving_now = True

    def mouseMoveEvent(self, event):
        if self.mode == State.MANIPULATE:

            if self.last_mouse_position is not None:
                dx = event.x() - self.last_mouse_position.x()
                dy = event.y() - self.last_mouse_position.y()

                angle_x = np.radians(dy * 0.5)
                angle_y = np.radians(dx * 0.5)

                rotation_x = R.from_rotvec(angle_x * np.array([1, 0, 0]))
                rotation_y = R.from_rotvec(angle_y * np.array([0, 1, 0]))

                self.rotation = rotation_y * rotation_x * self.rotation

                self.update()

                self.last_mouse_position = event.pos()

        # В состоянии рисования
        elif self.mode == State.DRAW:
            if event.buttons() & Qt.LeftButton:
                pos = event.pos()

                ogl_x, ogl_y = self.getCurrentCoords(pos)

                if self.splineType == SplineType.ELLIPSE:
                    if self.ellipse:
                        # if
                        self.ellipse.updateEndPoint((ogl_x, ogl_y), self.scale)

                elif self.splineType == SplineType.POLYLINE:
                    if self.line:
                        self.line.addPoint((ogl_x, ogl_y))

                self.update()

            if event.buttons() & Qt.RightButton:
                pos = event.pos()

                ogl_x, ogl_y = self.getCurrentCoords(pos)
                old_ogl_x, old_ogl_y = self.getCurrentCoords(self.last_mouse_position)
                dx = ogl_x - old_ogl_x
                dy = ogl_y - old_ogl_y
                if self.splineType == SplineType.ELLIPSE and self.ellipse:
                    if self.is_drawing_moving_now:
                        # print(ogl_x, ogl_y, self.ellipse.center_x, self.ellipse.center_y)
                        self.ellipse.move(dx, dy)

                elif self.splineType == SplineType.POLYLINE and self.line:
                    if self.is_drawing_moving_now:
                        self.line.move(dx, dy)
                self.last_mouse_position = pos
                self.update()

    def getCurrentCoords(self, pos):
        x, y = pos.x(), pos.y()

        ogl_x = (2.0 * x) / self.width() - 1.0
        ogl_y = 1.0 - (2.0 * y) / self.height()

        # Соблюдение условий нормализации в соответствии с размерами окна и масштабом проецирования
        if self.aspect >= 1.0:
            ogl_x = ogl_x * self.aspect
        else:
            ogl_y = ogl_y / self.aspect

        return ogl_x, ogl_y

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_position = None
        if self.ellipse:
            self.ellipse.finish()
        if self.line:
            self.line.finish()
        if self.is_drawing_moving_now:
            self.is_drawing_moving_now = False

    def wheelEvent(self, event: QWheelEvent):
        if self.mode == State.MOVE or self.mode == State.MANIPULATE:
            if event.angleDelta().y() > 0:
                self.zoomIn()
            else:
                self.zoomOut()
        elif self.mode == State.DRAW and self.ellipse:
            # от себя -- эллипс вращается вправо, на себя -- влево
            angle_delta = event.angleDelta().y() / 120  # каждый шаг колеса мыши обычно равен 120 единицам
            self.ellipse.rotate(angle_delta)  # изменение угла поворота
        event.accept()

    # def drawAxes(self):
    #     axis_length = 0.3
    #
    #     glBegin(GL_LINES)
    #     glColor3f(1.0, 0.0, 0.0)
    #     glVertex3f(0, 0, 0)
    #     glVertex3f(axis_length, 0, 0)
    #
    #     glColor3f(0.0, 1.0, 0.0)
    #     glVertex3f(0, 0, 0)
    #     glVertex3f(0, axis_length, 0)
    #
    #     glColor3f(0.0, 0.0, 1.0)
    #     glVertex3f(0, 0, 0)
    #     glVertex3f(0, 0, axis_length)
    #     glEnd()

    def setSplineTypeEllipse(self):
        self.splineType = SplineType.ELLIPSE

    def setSplineTypePolyline(self):
        self.splineType = SplineType.POLYLINE

    def drawAxes(self):
        axis_length = 0.1
        rotation_matrix = self.rotation.as_matrix()

        axes = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]

        glBegin(GL_LINES)
        for i, color in enumerate([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]):
            glColor3f(*color)
            start_point = np.array([0.0, 0.0, 0.0])
            end_point = axis_length * np.array(axes[i])
            rotated_start_point = np.dot(rotation_matrix, start_point)
            rotated_end_point = np.dot(rotation_matrix, end_point)
            glVertex3f(*rotated_start_point)
            glVertex3f(*rotated_end_point)
        glEnd()

    def highlightPointsInPolygon(self):
        if self.line:
            highlighted_points = []

            normalized_data = self.tour.getCurrentNormalizedData()
            rotation_matrix = self.rotation.as_matrix()

            rotated_points = np.dot(rotation_matrix, normalized_data.T).T
            scaled_points = rotated_points[:, :2] * self.scale

            for i, scaled_point in enumerate(scaled_points):
                if self.line.isPointInside(scaled_point[:2]):
                    highlighted_points.append(i)

            return highlighted_points

    def highlightPointsInEllipse(self):
        if self.ellipse:
            highlighted_points = []

            rotation_matrix = self.rotation.as_matrix()

            rotated_points = np.dot(rotation_matrix, self.tour.getCurrentNormalizedData().T).T
            scaled_points = rotated_points[:, :2] * self.scale

            for i, scaled_point in enumerate(scaled_points):

                scaled_x, scaled_y = scaled_point[:2]
                if self.ellipse.isPointInside(scaled_x, scaled_y):
                    highlighted_points.append(i)

            return highlighted_points
        return None

    def addCluster(self, cluster_number=1):
        count_points_in_cluster = 0
        points_in_cluster = []
        if self.splineType == SplineType.ELLIPSE:
            points_in_cluster = self.highlightPointsInEllipse()
        elif self.splineType == SplineType.POLYLINE:
            points_in_cluster = self.highlightPointsInPolygon()

        for i in points_in_cluster:
            result = self.tour.addCluster(i, cluster_number)
            if result:
                count_points_in_cluster += 1
        self.update()
        return count_points_in_cluster
