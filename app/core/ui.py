import sys
from typing import Union, Iterable, Callable

import numpy
from PyQt5 import QtCore
from PyQt5.Qt import QApplication, QDesktopWidget
from PyQt5.Qt import QImage, QPixmap, QColor, QPainter, QPen
from PyQt5.Qt import pyqtSignal as Signal
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLayout, QLabel, QSlider
from PyQt5.QtWidgets import QWidget

from .utils import create_context_and_queue


def to_pixmap(data: numpy.ndarray):
    image = QImage(data.data, *data.shape[:-1], QImage.Format_ARGB32)
    pixmap = QPixmap()
    # noinspection PyArgumentList
    pixmap.convertFromImage(image)
    return pixmap


def mb_state(m_event):
    return bool(QtCore.Qt.LeftButton & m_event.buttons()), bool(QtCore.Qt.RightButton & m_event.buttons())


def v_stack(*args, cm=(0, 0, 0, 0)):
    l = QVBoxLayout()
    l.setContentsMargins(*cm)
    l.setSpacing(2)
    for a in args:
        if isinstance(a, QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l


def h_stack(*args, cm=(0, 0, 0, 0)):
    l = QHBoxLayout()
    l.setContentsMargins(*cm)
    l.setSpacing(0)
    for a in args:
        if isinstance(a, QLayout):
            l.addLayout(a)
        else:
            l.addWidget(a)
    return l


def stack(*args, kind="v", cm=(0, 0, 0, 0), sp=0):
    if kind == "v":
        l = v_stack(*args)
        l.setSpacing(sp)
        l.setContentsMargins(*cm)
    elif kind == "h":
        l = h_stack(*args)
        l.setSpacing(sp)
        l.setContentsMargins(*cm)
    else:
        raise ValueError("Unknown kind of stack: \"{}\"".format(kind))
    return l


class Target2D:

    def __init__(self, color: QColor, shape: tuple = (True, True)):
        self._color = color
        self._shape = shape
        self._pos = (-1, -1)
        self._setPosCalled = False

    def setPosCalled(self):
        return self._setPosCalled

    def shape(self):
        return self._shape

    def pos(self):
        return self._pos

    def setPos(self, pos: tuple):
        self._setPosCalled = True
        self._pos = pos

    def draw(self, w: int, h: int, painter: QPainter):
        pen = QPen(self._color, 1)
        painter.setPen(pen)
        if self._shape[1]:
            painter.drawLine(0, self._pos[1], w, self._pos[1])
        if self._shape[0]:
            painter.drawLine(self._pos[0], 0, self._pos[0], h)


class Image2D(QLabel):

    selectionChanged = Signal(tuple, tuple)

    def _onMouseEvent(self, event):
        if not any(self._target._shape):
            return
        left, right = mb_state(event)
        if left:
            self._target.setPos((event.x(), event.y()))
            self.repaint()
            if self._target.setPosCalled():
                vals = self.targetReal()
                self.selectionChanged.emit(
                    (vals[0] if self._target.shape()[0] else None,
                     vals[1] if self._target.shape()[1] else None),
                    (left, right))

    def mousePressEvent(self, event):
        super(Image2D, self).mousePressEvent(event)
        self._onMouseEvent(event)

    def mouseMoveEvent(self, event):
        super(Image2D, self).mouseMoveEvent(event)
        self._onMouseEvent(event)

    def paintEvent(self, QPaintEvent):
        super(Image2D, self).paintEvent(QPaintEvent)
        self._target.draw(self.width(), self.height(), QPainter(self))

    def __init__(self,
                 targetColor: QColor = QtCore.Qt.red,
                 targetShape: tuple = (True, True),
                 spaceShape: tuple = (-1.0, 1.0, -1.0, 1.0),
                 textureShape: tuple = (1, 1),
                 invertY: bool = True
                 ):
        super().__init__()
        self.setMouseTracking(True)
        self._target = Target2D(targetColor, targetShape)
        self._spaceShape = spaceShape
        self._invertY = invertY
        self._textureShape = textureShape
        self._textureDataReference = None

    def spaceShape(self) -> tuple:
        return self._spaceShape

    def setSpaceShape(self, spaceShape: tuple) -> None:
        self._spaceShape = spaceShape

    def textureShape(self) -> tuple:
        return self._textureShape

    def setTexture(self, data: numpy.ndarray) -> None:
        self._textureDataReference = data
        self._textureShape = data.shape[:-1]
        self.setPixmap(to_pixmap(data))

    def targetPx(self) -> tuple:
        return self._target.pos()

    def setTargetPx(self, targetLocation: tuple) -> None:
        self._target.setPos(targetLocation)

    def targetReal(self) -> tuple:
        x, y = self._target.pos()
        x = self._spaceShape[0] + x / self._textureShape[0] * (self._spaceShape[1] - self._spaceShape[0])
        if self._invertY:
            y = self._spaceShape[2] + (self._textureShape[1] - y) / \
                self._textureShape[1] * (self._spaceShape[3] - self._spaceShape[2])
        else:
            y = self._spaceShape[2] + y / self._textureShape[1] * (self._spaceShape[3] - self._spaceShape[2])
        x = numpy.clip(x, self._spaceShape[0], self._spaceShape[1])
        y = numpy.clip(y, self._spaceShape[2], self._spaceShape[3])
        return x, y

    def setTargetReal(self, targetLocation: tuple) -> None:
        x, y = targetLocation
        x = (x - self._spaceShape[0]) / (self._spaceShape[1] - self._spaceShape[0])*self._textureShape[0]
        y = self._textureShape[1] - \
            (y - self._spaceShape[2]) / (self._spaceShape[3] - self._spaceShape[2])*self._textureShape[1]
        self._target.setPos((x, y))
        self.repaint()


class ParameterizedImageWidget(QWidget):

    selectionChanged = Signal(tuple, tuple)

    valueChanged = Signal(tuple)

    def __init__(self, bounds: tuple, names: tuple, shape: tuple, targetColor: QColor = Qt.red, textureShape=(1,1)):
        super().__init__()

        if names is None and shape is None:
            names = ("", "")
            shape = (True, True)

        if names is None:
            names = ("", "")

        if shape is None:
            shape = tuple(bool(el) if el is not None else False for el in names)

        self._bounds = bounds
        self._names = names
        self._shape = shape
        self._positionLabel = QLabel()
        self._imageWidget = Image2D(targetColor=targetColor, targetShape=shape, spaceShape=bounds, textureShape=textureShape)

        self._imageWidget.selectionChanged.connect(lambda val, _: self.valueChanged.emit(val))
        self._imageWidget.selectionChanged.connect(self.selectionChanged)
        self._imageWidget.selectionChanged.connect(self.updatePositionLabel)

        self.setLayout(v_stack(
            self._imageWidget,
            self._positionLabel
        ))

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        # self.updatePositionLabel(self._imageWidget._target.)

    def updatePositionLabel(self, value, buttons):
        self._positionLabel.setText("  |  ".join(
            filter(lambda x: x is not None, (
                None if sh is None or vl is None else "{0} = {1:.4f}".format(nm, vl)
                for sh, nm, vl in zip(self._shape, self._names, value)
            ))
        ))

    def setImage(self, image: numpy.ndarray):
        self._imageWidget.setTexture(image)

    def value(self):
        return self._imageWidget.targetReal()

    def setValue(self, targetValue: tuple):
        if targetValue is None:
            self._imageWidget.setTargetPx((-1, -1))
            self._positionLabel.setText("")
        else:
            self._imageWidget.setTargetReal(targetValue)
            self.updatePositionLabel(targetValue, (False, False))

    def setShape(self, shape):
        self._shape = shape
        self._imageWidget._target._shape = shape


class SimpleApp(QWidget):

    DEFAULT_IMAGE_SIZE = (256, 256)

    def __init__(self, title):
        import json
        self.app = QApplication(sys.argv)
        # noinspection PyArgumentList
        super().__init__(parent=None)
        self.setWindowTitle(title)
        self.configFile = None
        self.config = None
        if len(sys.argv) > 1:
            self.configFile = sys.argv[1]
            print("Loading config from file:", self.configFile)
            try:
                with open(self.configFile) as f:
                    self.config = json.load(f)
            except Exception as e:
                raise RuntimeWarning("Cannot load configuration from file %s: %s" % (self.configFile, str(e)))
            else:
                print("Loaded configuration:", self.config)
        self.ctx, self.queue = create_context_and_queue(self.config)

    def run(self):
        screen = QDesktopWidget().screenGeometry()
        self.show()
        x = ((screen.width() - self.width()) // 2) if screen.width() > self.width() else 0
        y = ((screen.height() - self.height()) // 2) if screen.height() > self.height() else 0
        self.move(x, y)
        sys.exit(self.app.exec_())


class RealSlider(QSlider):
    valueChanged = Signal(float)

    def __init__(self, bounds: tuple, horizontal=True, steps=10000):
        super().__init__()
        self._steps = steps
        self._bounds = bounds
        self.setOrientation(Qt.Vertical if not horizontal else Qt.Horizontal)
        self.setMinimum(0)
        self.setMaximum(self._steps)

        super().valueChanged.connect(lambda _: self.valueChanged.emit(self.value()))

    def setValue(self, v):
        super().setValue(int((v - self._bounds[0]) / (self._bounds[1] - self._bounds[0]) * self._steps))

    def value(self):
        return float(super().value()) / self._steps * (self._bounds[1] - self._bounds[0]) + self._bounds[0]


class IntegerSlider(QSlider):

    def __init__(self, bounds: tuple, horizontal=True):
        super().__init__()
        self.setOrientation(Qt.Vertical if not horizontal else Qt.Horizontal)
        self.setMinimum(bounds[0])
        self.setMaximum(bounds[1])


def createSlider(sliderType: str, bounds: tuple,
                 horizontal: bool = True,
                 withLabel: str = None, labelPosition: str = "left",
                 withValue: Union[float, int, None] = None,
                 connectTo: Union[Iterable, Callable, None] = None,
                 putToLayout: QLayout = None
                 ) -> tuple:
    sliderType = sliderType.lower()
    if sliderType in {"real", "r", "float"}:
        slider = RealSlider(bounds, horizontal)
    elif sliderType in {"int", "integer", "i", "d"}:
        slider = IntegerSlider(bounds, horizontal)
    else:
        raise ValueError("Unknown slider type: {}".format(sliderType))
    if withValue is not None:
        slider.setValue(withValue)

    layout = None
    if withLabel is not None:
        positions = {"left", "top", "right"}
        if labelPosition not in positions:
            raise ValueError("Label position must be one of: {}".format(positions))
        supportsValues = True
        try:
            withLabel.format(0.42)
        except:
            supportsValues = False
        label = QLabel(withLabel)

        if supportsValues:
            def setVal(val): label.setText(withLabel.format(val))

            setVal(withValue)
            slider.valueChanged.connect(setVal)

        if putToLayout is None:
            layout = (QHBoxLayout if labelPosition in {"left", "right"} else QVBoxLayout)()
        else:
            layout = putToLayout
        if labelPosition in {"right"}:
            layout.addWidget(slider)
            layout.addWidget(label)
        else:
            layout.addWidget(label)
            layout.addWidget(slider)

    if connectTo is not None:
        if isinstance(connectTo, Iterable):
            [slider.valueChanged.connect(fn) for fn in connectTo]
        else:
            slider.valueChanged.connect(connectTo)

    return slider, slider if layout is None else layout
