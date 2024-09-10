import os
import sys

import numpy as np
from PyQt5.QtCore import Qt, QUrl, QEvent
from PyQt5.QtGui import QIcon, QDesktopServices
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QPushButton, \
    QHBoxLayout, QTabWidget, QAction, QSpinBox, QBoxLayout, QFrame, QGridLayout, QTableWidget, \
    QMenu, QTableWidgetItem, QMessageBox, QRadioButton, QStyledItemDelegate, QLineEdit, QSlider

from CustomOpenGLWidget import VisualizeDataWidget, State
from DataLoader import DataLoader
from DataSaver import DataSaver
from Tour import Tour

# Эти 2 строчки нужны для корректного отображения интерфейса на экранах с высоким dpi на функциональность программы
# не влияет
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # enable highdpi scaling
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # use highdpi icons


class ReadOnlyDelegate(QStyledItemDelegate):
    """Класс-делегат, который запрещает редактирование. Используется для запрещения редактирования столбцов таблицы
     Clusters"""

    def createEditor(self, parent, option, index):
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Задаём название окна
        self.setWindowTitle("Data_analysis")
        self.setWindowIcon(QIcon("images/window_icon_2_128.png"))

        # Создаем главное меню
        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('File')

        # Добавляем пункты меню
        self.openAction = QAction('Open Data')
        self.saveAction = QAction('Save Data')
        self.openDir = QAction('Open Directory')
        self.closeAction = QAction('Close Data')

        self.fileMenu.addAction(self.openAction)
        self.fileMenu.addAction(self.saveAction)
        self.fileMenu.addAction(self.openDir)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.closeAction)

        # Создаем центральный виджет
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        # Создаем горизонтальный layout для верхних кнопок и спинбокса
        self.topLayout = QHBoxLayout()
        self.topLayout.setDirection(QBoxLayout.LeftToRight)
        self.topLayout.setContentsMargins(0, 0, 0, 0)  # Уменьшаем отступы
        self.topLayout.setSpacing(2)
        self.topLayout.setAlignment(Qt.AlignLeft)

        # Создаём верхние кнопки
        self.openButton = QPushButton(' Open')
        self.openButton.setFixedSize(75, 30)
        self.openButton.setIcon(QIcon('images/open_file_new.png'))

        self.saveButton = QPushButton(' Save')
        self.saveButton.setFixedSize(75, 30)
        self.saveButton.setIcon(QIcon('images/save_new.png'))

        self.startButton = QPushButton(' Start')
        self.startButton.setFixedSize(75, 30)
        self.startButton.setIcon(QIcon('images/play_new.png'))

        self.stopButton = QPushButton(' Stop')
        self.stopButton.setFixedSize(75, 30)
        self.stopButton.setIcon(QIcon('images/pause_new.png'))

        self.spin_box_dim = QSpinBox()
        self.spin_box_dim.setRange(0, 100000)
        self.spin_box_dim.setFixedSize(75, 28)
        self.spin_box_dim.setStyleSheet("QSpinBox { border-radius: 5px; }")
        self.spin_box_dim.lineEdit().setReadOnly(True)
        # self.spin_box_dim.setEnabled(False)

        # Добавляем созданные кнопки и поле для ввода на горизонтальный layout
        self.topLayout.addWidget(self.openButton)
        self.topLayout.addWidget(self.saveButton)
        self.topLayout.addWidget(self.startButton)
        self.topLayout.addWidget(self.stopButton)
        self.topLayout.addWidget(self.spin_box_dim)

        # self.topLayout.removeWidget()

        # Создаем основной вертикальный layout
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.topLayout)
        self.mainLayout.setContentsMargins(3, 0, 3, 3)
        self.mainLayout.setSpacing(0)

        # Создаем горизонтальный layout для OpenGL виджета и табов
        self.contentLayout = QHBoxLayout()

        # Создаем OpenGL виджет
        self.opengl_widget = VisualizeDataWidget()
        self.opengl_widget.setMinimumSize(529, 444)
        self.contentLayout.addWidget(self.opengl_widget)
        self.contentLayout.setContentsMargins(0, 0, 0, 0)  # Уменьшаем отступы
        self.contentLayout.setSpacing(3)

        # Создаём вертикальную разделительную черту между openGL виджетом и вкладками
        self.splitLine = QFrame()
        self.splitLine.setFrameShape(QFrame.VLine)
        self.splitLine.setFrameShadow(QFrame.Sunken)
        self.contentLayout.addWidget(self.splitLine)

        # Создаем многостраничный виджет (Tab Widget)
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(298)

        # Первая вкладка
        self.tab1 = QWidget()
        self.tab1Layout = QGridLayout()
        self.draw_or_move_button = QPushButton("Add Cluster")
        self.draw_or_move_button.setFixedHeight(27)
        self.add_cluster_button = QPushButton("Add cluster")
        self.add_cluster_button.setFixedHeight(27)
        self.reset_drawing_button = QPushButton("Reset drawing")
        self.reset_drawing_button.setFixedHeight(27)

        # Флажки переключения режима
        self.ellipse_radio_button = QRadioButton("Ellipse")
        self.line_radio_button = QRadioButton("Line")

        # Кнопки управления размерами эллипса
        self.increase_ellipse_height = QPushButton("+ Height")
        self.increase_ellipse_height.setFixedHeight(27)
        self.decrease_ellipse_height = QPushButton("- Height")
        self.decrease_ellipse_height.setFixedHeight(27)
        self.increase_ellipse_wight = QPushButton("+ Wight")
        self.increase_ellipse_wight.setFixedHeight(27)
        self.decrease_ellipse_wight = QPushButton("- Wight")
        self.decrease_ellipse_wight.setFixedHeight(27)

        # Кнопки перемещения эллипса
        self.turn_ellipse_up = QPushButton("Up")
        self.turn_ellipse_up.setFixedHeight(27)
        self.turn_ellipse_down = QPushButton("Down")
        self.turn_ellipse_down.setFixedHeight(27)
        self.turn_ellipse_right = QPushButton("Right")
        self.turn_ellipse_right.setFixedHeight(27)
        self.turn_ellipse_left = QPushButton("Left")
        self.turn_ellipse_left.setFixedHeight(27)

        # Создаём виджет для задания номера кластера
        self.spin_box_cluster_number = QSpinBox()
        self.spin_box_cluster_number.setRange(1, 99)
        self.spin_box_cluster_number.setFixedHeight(27)
        self.spin_box_cluster_number.setStyleSheet("QSpinBox { border-radius: 5px; }")

        # Создаём пустой виджет, который нужен для корректного отображения остальных виджетов, он по сути невидим
        self.empty_widget1 = QWidget()

        # Добавляем кнопки в сетку
        self.tab1Layout.addWidget(self.draw_or_move_button, 0, 1)
        self.tab1Layout.addWidget(self.add_cluster_button, 1, 2)
        self.tab1Layout.addWidget(self.reset_drawing_button, 1, 0)
        self.tab1Layout.addWidget(self.spin_box_cluster_number, 1, 1)

        # Создаём layout для флажков
        self.splineTypeLayout = QVBoxLayout()
        self.splineTypeLayout.addWidget(self.ellipse_radio_button)
        self.splineTypeLayout.addWidget(self.line_radio_button)


        # Создаём виджет как обёртку для этого layout
        self.splineTypeContainer = QWidget()
        self.splineTypeContainer.setLayout(self.splineTypeLayout)

        # Помещаем виджет с флажками на общую сетку
        self.tab1Layout.addWidget(self.splineTypeContainer, 2, 1)

        self.tab1Layout.addWidget(self.increase_ellipse_height, 4, 1)
        self.tab1Layout.addWidget(self.decrease_ellipse_height, 5, 1)
        self.tab1Layout.addWidget(self.increase_ellipse_wight, 5, 2)
        self.tab1Layout.addWidget(self.decrease_ellipse_wight, 5, 0)

        self.tab1Layout.addWidget(self.turn_ellipse_up, 7, 1)
        self.tab1Layout.addWidget(self.turn_ellipse_down, 8, 1)
        self.tab1Layout.addWidget(self.turn_ellipse_right, 8, 2)
        self.tab1Layout.addWidget(self.turn_ellipse_left, 8, 0)

        # Вставляем пустой виджет для того, чтобы появились пустые строки
        self.tab1Layout.addWidget(self.empty_widget1, 3, 0)
        self.tab1Layout.addWidget(self.empty_widget1, 6, 0)
        self.tab1Layout.addWidget(self.empty_widget1, 9, 0)

        # Добавляем расположение на 1 вкладку и размещаем её на вкладочном виджете
        self.tab1.setLayout(self.tab1Layout)
        self.tabs.addTab(self.tab1, "Add Cluster")

        # Вторая вкладка
        self.tab2 = QWidget()
        self.tab2Layout = QVBoxLayout()

        self.tabs.addTab(self.tab2, "Clusters")

        # Создаем таблицу
        self.cluster_table = QTableWidget()
        self.cluster_table.verticalHeader().setVisible(False)
        self.cluster_table.setColumnCount(3)
        self.cluster_table.setHorizontalHeaderLabels(["Cluster №", "Name", "Points"])
        self.cluster_table.setContextMenuPolicy(Qt.CustomContextMenu)

        self.cluster_table.setColumnWidth(0, 80)
        self.cluster_table.setColumnWidth(1, 80)
        self.cluster_table.setColumnWidth(2, 100)

        # Установка делегатов для столбцов "Cluster №" и "Points"
        readonly_delegate = ReadOnlyDelegate()
        self.cluster_table.setItemDelegateForColumn(0, readonly_delegate)
        self.cluster_table.setItemDelegateForColumn(2, readonly_delegate)

        # Добавляем расположение на 2 вкладку и размещаем её на вкладочном виджете

        self.tab2Layout.addWidget(self.cluster_table)
        self.tab2.setLayout(self.tab2Layout)

        # Третья вкладка
        self.tab3 = QWidget()
        self.tab3Layout = QVBoxLayout()

        self.tabs.addTab(self.tab3, "Parameters")

        # Строка скорости и угла
        self.indicateString = QLineEdit()
        self.indicateString.setText("")
        self.indicateString.setReadOnly(True)

        # Создаем ползунок скорости
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setValue(115)  # Устанавливаем начальное значение ползунка
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)

        # Добавляем на layout третьей вкладки
        self.tab3Layout.addWidget(self.indicateString)
        self.tab3Layout.addWidget(self.slider)

        # Добавляем layout третьей вкладки на вкладку
        self.tab3.setLayout(self.tab3Layout)

        # Добавляем на общее расположение виджет с вкладками
        self.contentLayout.addWidget(self.tabs)
        self.mainLayout.addLayout(self.contentLayout)

        # Устанавливаем основной layout на центральный виджет
        self.centralWidget.setLayout(self.mainLayout)



        ### Обработка событий

        # Открытие файла
        self.openButton.clicked.connect(self.openFile)
        self.openAction.triggered.connect(self.openFile)

        # Закрытие файла
        self.closeAction.triggered.connect(self.closeFile)

        # Открытие директории с файлом
        self.openDir.triggered.connect(self.showCurrentDirectoryInFileManager)

        # Сохранение файла
        self.saveAction.triggered.connect(self.savePointsAndStatistics)
        self.saveButton.clicked.connect(self.savePointsAndStatistics)

        # Стоп
        self.stopButton.clicked.connect(self.stop)

        # Старт
        self.startButton.clicked.connect(self.start)

        # Рисование перемещение
        self.draw_or_move_button.clicked.connect(self.changeMode)

        # Добавление кластера
        self.add_cluster_button.clicked.connect(self.addCluster)

        # Сброс нарисованных фигур
        self.reset_drawing_button.clicked.connect(self.resetDrawing)

        self.line_radio_button.clicked.connect(self.setSplineTypeLine)
        self.ellipse_radio_button.clicked.connect(self.setSplineTypeEllipse)

        self.increase_ellipse_height.clicked.connect(self.increaseEllipseHeight)
        self.decrease_ellipse_height.clicked.connect(self.decreaseEllipseHeight)

        self.increase_ellipse_wight.clicked.connect(self.increaseEllipseWight)
        self.decrease_ellipse_wight.clicked.connect(self.decreaseEllipseWight)

        self.turn_ellipse_up.clicked.connect(self.turnEllipseUp)
        self.turn_ellipse_down.clicked.connect(self.turnEllipseDown)
        self.turn_ellipse_right.clicked.connect(self.turnEllipseRight)
        self.turn_ellipse_left.clicked.connect(self.turnEllipseLeft)

        self.cluster_table.customContextMenuRequested.connect(self.showContextMenu)
        self.cluster_table.itemChanged.connect(self.onClusterNameChanged)

        self.slider.valueChanged.connect(self.sliderValueChange)

        # Так как начальное состояние move деактивируем кнопки связанные с выделением кластеров
        self.add_cluster_button.setEnabled(False)
        self.reset_drawing_button.setEnabled(False)
        self.spin_box_cluster_number.setEnabled(False)
        self.decrease_ellipse_height.setEnabled(False)
        self.increase_ellipse_height.setEnabled(False)
        self.increase_ellipse_wight.setEnabled(False)
        self.decrease_ellipse_wight.setEnabled(False)
        self.turn_ellipse_up.setEnabled(False)
        self.turn_ellipse_down.setEnabled(False)
        self.turn_ellipse_right.setEnabled(False)
        self.turn_ellipse_left.setEnabled(False)
        self.ellipse_radio_button.setEnabled(False)
        self.line_radio_button.setEnabled(False)

        self.tour = None
        self.file_name = None
        self.dim = None

    def sliderValueChange(self, value):
        if self.tour:
            float_value = -0.0026 + (value / 1000) * (0.02 - (-0.0026))
            self.tour.setRotationSpeed(float_value)
            # print(float_value)

    def setIndicateString(self, string):
        self.indicateString.setText(string)

    def on_resize(self, event):
        """Устанавливаем новые размеры и шрифты для виджетов в зависимости от размера окна
         Не очень нужная функция но я её пока здесь оставлю (возможно работает для мониторов с высоким разрешением
          и dpi для корректного функционирования программы не нужно"""
        new_width = self.width()
        # new_height = self.height()

        self.label.setStyleSheet(f"font-size: {new_width // 30}px;")
        self.button.setStyleSheet(f"font-size: {new_width // 40}px;")

        # Не забываем вызвать оригинальный обработчик
        QMainWindow.resizeEvent(self, event)

    def _getFileName(self):
        file_name = QFileDialog().getOpenFileName(parent=self,
                                                  caption="Select a file",
                                                  directory=os.path.join(os.getcwd(), "Data"),
                                                  filter="All Files (*);;Cluster files (*.data)",
                                                  initialFilter="Cluster files (*.data)",
                                                  options=QFileDialog.Options())
        file_name = file_name[0]

        return file_name

    def openFile(self, file_name: str = None):
        """Загружает данные из файла и начинает работу с ними"""
        if not file_name:
            file_name = QFileDialog().getOpenFileName(parent=self,
                                                      caption="Select a file",
                                                      directory=os.path.join(os.getcwd(), "Data"),
                                                      filter="All Files (*);;Cluster files (*.data)",
                                                      initialFilter="Cluster files (*.data)",
                                                      options=QFileDialog.Options())
            file_name = file_name[0]

            if not file_name:
                pass

        try:
            if file_name == "":
                return

                # Проверка расширения файла
            if not file_name.endswith('.data'):
                QMessageBox.warning(self, "Wrong file format",
                                    'Invalid file extension. The extension should be ".data"')
                return

            self.setAllApplicationToDefault()

            data_loader = DataLoader(self, file_name)
            successfully = data_loader.loadData()
            if not successfully:
                return
            self.dim = data_loader.getDim()


            self.tour = Tour(self, data_loader.getDim(), data_loader.getData(), data_loader.getLabels(),
                             data_loader.getLabelsDict())
            self.opengl_widget.setTour(self.tour)
            self.opengl_widget.setSize(2)  # Магическое число, отвечает за размер отрисовки точек в opengl_widget

            # Смена заголовка программы
            self.setWindowTitle(f"Data_analysis - {file_name}")
            self.file_name = file_name

            # Смена значения отображаемой размерности
            self.spin_box_dim.setValue(data_loader.dim)

            self.restoreClusterTable()

        except FileNotFoundError:
            pass

    def restoreClusterTable(self):
        """Восстанавливает состояние таблицы, в случае если открыт файл с указанием меток кластеров"""
        if self.tour:
            unique_labels = np.sort(np.unique(self.tour.labels))
            for label in unique_labels:
                if label == 0:
                    continue

                count = np.sum(self.tour.labels == label)
                label_name = self.tour.getLabelsDict().get(label, None)
                self.insertNewRowToClusterTable(label, count, label_name=label_name)

    def onClusterNameChanged(self, item):
        if item.column() == 1:
            row = item.row()
            cluster_number = int(self.cluster_table.item(row, 0).text())
            self.tour.addClusterLabel(cluster_number, item.text())
            # print(item.text())

    def disableEllipseButtons(self):
        """Деактивирует кнопки управления рисованием (для режима манипулирования точками)"""
        self.draw_or_move_button.setText("Add Cluster")

        self.add_cluster_button.setEnabled(False)
        self.reset_drawing_button.setEnabled(False)
        self.spin_box_cluster_number.setEnabled(False)
        # Выключает только кнопки взаимодействия с эллипсом
        self.disableEllipseManipulateButtons()

        self.ellipse_radio_button.setEnabled(False)
        self.line_radio_button.setEnabled(False)
        self.ellipse_radio_button.setChecked(False)

    def enableEllipseButtons(self):
        """Делает активными кнопки управления рисованием (для режима рисования)"""
        self.draw_or_move_button.setText("Manipulate")

        self.add_cluster_button.setEnabled(True)
        self.reset_drawing_button.setEnabled(True)
        self.spin_box_cluster_number.setEnabled(True)
        # Включает только кнопки взаимодействия с эллипсом
        self.enableEllipseManipulateButtons()

        self.ellipse_radio_button.setEnabled(True)
        self.line_radio_button.setEnabled(True)
        self.ellipse_radio_button.setChecked(True)

    def setAllApplicationToDefault(self):
        """Сбрасывает настройки программы к базовым, нужно перед открытием новых файлов"""
        self.setWindowTitle("Data_analysis")
        self.spin_box_dim.setValue(0)
        self.spin_box_cluster_number.setValue(1)
        self.tour = None
        self.file_name = None
        self.dim = None
        self.setDefaultClusterTable()

        self.disableEllipseButtons()

        self.opengl_widget.setDefaultState()
        self.slider.setValue(115)
        self.indicateString.setText("")

    def setDefaultClusterTable(self):
        """Просто удаляет строчки из таблицы (без взаимодействия с данными кластеров)"""
        for row in range(self.cluster_table.rowCount() - 1, -1, -1):
            self.cluster_table.removeRow(row)

    def savePointsAndStatistics(self) -> bool:
        """Сохранение всех данных в 2 файла"""

        if not self.tour:
            QMessageBox.information(self, "Warning", "It is no data to save!", QMessageBox.Ok)
            return False

        options = QFileDialog.Options()
        file_name_labels, _ = QFileDialog.getSaveFileName(self, "Save File", os.path.join(os.getcwd(), "Cluster"),
                                                          "All Files (*);;Cluster files (*.data)",
                                                          "Cluster files (*.data)", options=options)

        if not file_name_labels:
            return False

        if file_name_labels.endswith('.stat'):
            file_name_labels = file_name_labels.rsplit('.', 1)[0]
            print(file_name_labels)

        # Получение только имени файла без абсолютного пути
        file_base_name = os.path.basename(file_name_labels)
        file_base_name_clean = file_base_name.rsplit('.', 1)[0]

        print(self.dim)
        data_saver = DataSaver(self.tour.getData(), self.tour.getLabels(), self.dim, self.tour.getLabelsDict())

        if file_name_labels:
            success_save_clusters = data_saver.saveLabels(file_name_labels)
            if not success_save_clusters:
                QMessageBox.critical(self, "Error", "Failed to save the data. Please try again.", QMessageBox.Ok)
                return False

        # Удаление расширения .data и добавление расширения .stat
        file_name_stat = file_name_labels.rsplit('.', 1)[0] + '.stat'

        if file_name_stat:
            success_save_statistics = data_saver.saveStatistics(file_name_stat)
            if not success_save_statistics:
                QMessageBox.critical(self, "Error", "Failed to save the statistics. Please try again.", QMessageBox.Ok)
                return False

        self.file_name = file_name_labels
        QMessageBox.information(self, "Success",
                                f"{file_base_name_clean}.data and {file_base_name_clean}.stat saved successfully!",
                                QMessageBox.Ok)

        # Смена название окна на текущий сохранённый файл
        self.setWindowTitle(f"Data_analysis - {file_name_labels}")
        return True

    def closeFile(self):
        """При закрытии файла сбрасывает настройки окна к базовым"""
        self.setAllApplicationToDefault()

    def stop(self):
        if self.opengl_widget.mode == State.DRAW:
            self.resetDrawing()
            self.disableEllipseButtons()

        self.setManipulateMode()

    def start(self):
        if self.opengl_widget.mode == State.DRAW:
            self.resetDrawing()
            self.disableEllipseButtons()
        self.setMoveMode()

    def showCurrentDirectoryInFileManager(self):
        file_path = self.file_name
        if not file_path:
            if os.path.exists(os.path.join(os.getcwd(), "Data")):
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.join(os.getcwd(), "Data")))
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.getcwd()))
            return

        file_path = os.path.dirname(self.file_name)
        if os.path.exists(file_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))

    def insertNewRowToClusterTable(self, cluster_number: int, points_in_cluster: int, label_name: str = None) -> None:
        """Добавляет новую строчку с заданными параметрами в таблицу кластеров."""
        row_position = self.cluster_table.rowCount()

        self.cluster_table.insertRow(row_position)

        self.cluster_table.setItem(row_position, 0, QTableWidgetItem(f"{cluster_number}"))
        if label_name:
            self.cluster_table.setItem(row_position, 1, QTableWidgetItem(label_name))
        else:
            self.cluster_table.setItem(row_position, 1, QTableWidgetItem("Cluster"))
        self.cluster_table.setItem(row_position, 2, QTableWidgetItem(f"{points_in_cluster} points"))

    # def
    def setMoveMode(self):
        # if
        self.opengl_widget.mode = State.MOVE

    def setManipulateMode(self):
        self.opengl_widget.mode = State.MANIPULATE

    def setDrawMode(self):
        self.opengl_widget.mode = State.DRAW

    def changeMode(self):
        """Смена режима работы кнопкой рисования"""
        if self.opengl_widget.mode == State.DRAW:
            self.setManipulateMode()
            self.resetDrawing()

            self.disableEllipseButtons()
        else:
            self.setDrawMode()

            self.enableEllipseButtons()

    def addCluster(self):
        """Добавляет кластер в таблицу и в память"""
        if self.tour and self.opengl_widget.mode == State.DRAW and (
                self.opengl_widget.ellipse or self.opengl_widget.line):
            points_in_cluster = self.opengl_widget.addCluster(self.spin_box_cluster_number.value())

            if points_in_cluster != 0:
                # Добавление кластера в таблицу
                row_position = self.cluster_table.rowCount()

                for row in range(row_position):
                    if int(self.cluster_table.item(row, 0).text()) == self.spin_box_cluster_number.value():
                        # Если нашли среди строк таблицы заданный кластер
                        old_number = int(self.cluster_table.item(row, 2).text().split()[0])
                        self.cluster_table.item(row, 2).setText(f"{old_number + points_in_cluster} points")
                        return

                # Добавляем новую строку
                self.insertNewRowToClusterTable(self.spin_box_cluster_number.value(), points_in_cluster)

    def resetDrawing(self):
        """Убирает всю нарисованную геометрию"""
        # if self.opengl_widget.mode == "draw":
        self.opengl_widget.line = None
        self.opengl_widget.ellipse = None

    def setSplineTypeLine(self):
        """Переключает на режим рисования линии"""
        self.line_radio_button.setChecked(True)
        self.ellipse_radio_button.setChecked(False)
        self.opengl_widget.setSplineTypePolyline()
        self.opengl_widget.ellipse = None
        self.disableEllipseManipulateButtons()

    def setSplineTypeEllipse(self):
        """Переключает на режим рисования эллипса"""
        self.ellipse_radio_button.setChecked(True)
        self.line_radio_button.setChecked(False)
        self.opengl_widget.setSplineTypeEllipse()
        self.opengl_widget.line = None
        self.enableEllipseManipulateButtons()

    def disableEllipseManipulateButtons(self):
        self.decrease_ellipse_height.setEnabled(False)
        self.increase_ellipse_height.setEnabled(False)
        self.increase_ellipse_wight.setEnabled(False)
        self.decrease_ellipse_wight.setEnabled(False)
        self.turn_ellipse_up.setEnabled(False)
        self.turn_ellipse_down.setEnabled(False)
        self.turn_ellipse_right.setEnabled(False)
        self.turn_ellipse_left.setEnabled(False)

    def enableEllipseManipulateButtons(self):
        self.decrease_ellipse_height.setEnabled(True)
        self.increase_ellipse_height.setEnabled(True)
        self.increase_ellipse_wight.setEnabled(True)
        self.decrease_ellipse_wight.setEnabled(True)
        self.turn_ellipse_up.setEnabled(True)
        self.turn_ellipse_down.setEnabled(True)
        self.turn_ellipse_right.setEnabled(True)
        self.turn_ellipse_left.setEnabled(True)

    def setDrawButtonsInvisible(self, ellipse_controls_only=False):
        self.decrease_ellipse_height.setVisible(False)
        self.increase_ellipse_height.setVisible(False)
        self.increase_ellipse_wight.setVisible(False)
        self.decrease_ellipse_wight.setVisible(False)
        self.turn_ellipse_up.setVisible(False)
        self.turn_ellipse_down.setVisible(False)
        self.turn_ellipse_right.setVisible(False)
        self.turn_ellipse_left.setVisible(False)

        if ellipse_controls_only:
            return

        self.add_cluster_button.setVisible(False)
        self.reset_drawing_button.setVisible(False)
        self.spin_box_cluster_number.setVisible(False)

        self.ellipse_radio_button.setVisible(False)
        self.line_radio_button.setVisible(False)
        # self.ellipse_radio_button.setVisible(True)

    def setDrawButtonsVisible(self):
        self.decrease_ellipse_height.setVisible(True)
        self.increase_ellipse_height.setVisible(True)
        self.increase_ellipse_wight.setVisible(True)
        self.decrease_ellipse_wight.setVisible(True)
        self.turn_ellipse_up.setVisible(True)
        self.turn_ellipse_down.setVisible(True)
        self.turn_ellipse_right.setVisible(True)
        self.turn_ellipse_left.setVisible(True)

        self.add_cluster_button.setVisible(True)
        self.reset_drawing_button.setVisible(True)
        self.spin_box_cluster_number.setVisible(True)

        self.ellipse_radio_button.setVisible(True)
        self.line_radio_button.setVisible(True)

    def increaseEllipseHeight(self):
        """Увеличивает большую полуось эллипса"""
        if self.opengl_widget.ellipse:
            self.opengl_widget.ellipse.increaseHeight()

    def decreaseEllipseHeight(self):
        """Уменьшает большую полуось эллипса"""
        if self.opengl_widget.ellipse:
            self.opengl_widget.ellipse.decreaseHeight()

    def increaseEllipseWight(self):
        """Увеличивает малую полуось эллипса"""
        if self.opengl_widget.ellipse:
            self.opengl_widget.ellipse.increaseWight()

    def decreaseEllipseWight(self):
        """Уменьшает малую полуось эллипса"""
        if self.opengl_widget.ellipse:
            self.opengl_widget.ellipse.decreaseWight()

    def turnEllipseUp(self):
        """Двигает нарисованный эллипс вверх"""
        if self.opengl_widget.ellipse:
            self.opengl_widget.ellipse.turnUp()

    def turnEllipseDown(self):
        """Двигает нарисованный эллипс вниз"""
        if self.opengl_widget.ellipse:
            self.opengl_widget.ellipse.turnDown()

    def turnEllipseRight(self):
        """Двигает нарисованный эллипс вправо"""
        if self.opengl_widget.ellipse:
            self.opengl_widget.ellipse.turnRight()

    def turnEllipseLeft(self):
        """Двигает нарисованный эллипс влево"""
        if self.opengl_widget.ellipse:
            self.opengl_widget.ellipse.turnLeft()

    def showContextMenu(self, pos):
        """Контекстное меню для удаления кластеров"""
        contextMenu = QMenu(self)
        deleteAction = contextMenu.addAction("Delete Cluster")
        contextMenu.addSeparator()
        deleteAllAction = contextMenu.addAction("Delete All")
        action = contextMenu.exec_(self.cluster_table.mapToGlobal(pos))

        if action == deleteAction:
            self.deleteRow(pos)
        elif action == deleteAllAction:
            self.deleteAllClusters()

    def deleteRow(self, pos):
        """Удаляет конкретный кластер
         получает координаты нажатия мыши для выбора конкретной строки"""
        row = self.cluster_table.rowAt(pos.y())
        if row >= 0:
            cluster_number = int(self.cluster_table.item(row, 0).text())
            self.tour.deleteCluster(cluster_number)
            self.cluster_table.removeRow(row)

    def deleteAllClusters(self):
        """Удаляет все кластеры из таблицы и из данных.
         Для оптимизации кусок кода повторяется из setDefaultClusterTable"""
        for row in range(self.cluster_table.rowCount() - 1, -1, -1):
            # при удалении предыдущей строки все последующие сдвигаются, поэтому удаляем с конца
            cluster_number = int(self.cluster_table.item(row, 0).text())
            self.tour.deleteCluster(cluster_number)
            self.cluster_table.removeRow(row)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()

    if len(sys.argv) > 1:
        file_to_open = sys.argv[1]
        window.openFile(file_to_open)

    window.show()
    sys.exit(app.exec_())
