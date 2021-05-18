from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QPushButton, QApplication, QMenu, QAction, QFileDialog, QLabel, \
    QWidget, QInputDialog
from PyQt5.QtGui import QIcon, QColor, QPixmap, qRgb

import sys
import cv2
import matplotlib.pyplot as plt
from math import exp, sqrt, pi, atan2


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pixmap = ''
        self.initUI()

        #self.pixmap = QPixmap('/Users/aaa/Desktop/zdjecie.png') #do usuniecia
        #self.refresh() #do usuniecia

    def initUI(self):
        self.resize(1200, 800)  # set size of window
        self.center()
        self.setWindowTitle('Image Processing')
        self.createWindowLabel()
        self.createMenu()
        self.createTollbar()
        self.show()

    def createWindowLabel(self):
        self.workspace = QWidget(self)
        self.label = QLabel(self.workspace)
        self.label.setGeometry(0, 0, self.width(), self.height())
        self.label.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label.setScaledContents(False)
        self.label.setWordWrap(True)
        self.setCentralWidget(self.workspace)

    def createMenu(self):
        menubar = self.menuBar()

        # File menu
        fileMenu = menubar.addMenu('File')
        openAct = QAction('Open', self)
        openAct.triggered.connect(self.open)
        fileMenu.addAction(openAct)
        saveAct = QAction('save', self)
        saveAct.triggered.connect(self.save)
        fileMenu.addAction(saveAct)

        # Colors menu
        colorsMenu = menubar.addMenu('Colors')

        saturateAct = QAction('Saturation', self)
        saturateAct.triggered.connect(self.saturate)
        colorsMenu.addAction(saturateAct)

        desaturateAct = QAction('Desaturation', self)
        desaturateAct.triggered.connect(self.desaturation)
        colorsMenu.addAction(desaturateAct)

        negativeAct = QAction('Negative', self)
        negativeAct.triggered.connect(self.negative)
        colorsMenu.addAction(negativeAct)

        brightnessAct = QAction('Brightness', self)
        brightnessAct.triggered.connect(self.brightness)
        colorsMenu.addAction(brightnessAct)

        linearContrastAct = QAction('Linear Contrast', self)
        linearContrastAct.triggered.connect(self.linearContrast)
        colorsMenu.addAction(linearContrastAct)

        invertAct = QAction('Invert', self)
        invertAct.triggered.connect(self.invert)
        colorsMenu.addAction(invertAct)

        invertAct = QAction('Grayscale', self)
        invertAct.triggered.connect(self.grayscale)
        colorsMenu.addAction(invertAct)

        # Filters menu
        filtersMenu = menubar.addMenu('Filters')

        meanAct = QAction('Mean', self)
        meanAct.triggered.connect(self.mean)
        filtersMenu.addAction(meanAct)

        gaussAct = QAction('Gauss', self)
        gaussAct.triggered.connect(self.gauss)
        filtersMenu.addAction(gaussAct)

        cannyAct = QAction('Canny', self)
        cannyAct.triggered.connect(self.canny)
        filtersMenu.addAction(cannyAct)

        # Histogram menu
        histogramMenu = menubar.addMenu('Histogram')

        histogramAct = QAction('Histogram', self)
        histogramAct.triggered.connect(self.histogram)
        histogramMenu.addAction(histogramAct)

    def createTollbar(self):
        scaleUpAct = QAction(QIcon('images/zoomin.png'), 'Scale up', self)
        undoAct = QAction(QIcon('images/undo.png'), 'Undo', self)
        scaleDownAct = QAction(QIcon('images/zoomout.png'), 'Scale down', self)
        scaleUpAct.triggered.connect(self.scaleup)
        undoAct.triggered.connect(self.undo)
        scaleDownAct.triggered.connect(self.scaledown)
        self.toolbar = self.addToolBar("Scale")
        self.toolbar.addAction(scaleUpAct)
        self.toolbar.addAction(scaleDownAct)
        self.toolbar.addAction(undoAct)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def save(self):  # nie działa
        filename = QFileDialog.getSaveFileName(self, 'Open file', './',
                                               "PGM (*.pgm);;PBM (*pbm);;PPM (*.ppm);;PNG (*.png)")
        if filename[0]:
            self.image.saves(filename)

    def open(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file', './', "Image files (*.pgm *.pbm *.ppm *.png)")
        self.pixmap = QPixmap(filename[0])
        self.refresh()

    def undo(self):
        self.refresh()

    def refresh(self):
        self.label.setPixmap(self.pixmap)

    def scaleup(self):
        self.pixmap = self.pixmap.scaled(int(self.pixmap.width() * 1.1), int(self.pixmap.height() * 1.1),
                                         QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)

    def scaledown(self):
        self.pixmap = self.pixmap.scaled(int(self.pixmap.width() * 0.9), int(self.pixmap.height() * 0.9),
                                         QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)

    def saturate(self):
        ratio, pressed = QInputDialog.getDouble(self, "Set value", "Value:", 10, -100, 100, 2)
        if pressed:
            image = self.pixmap.toImage()
            width = self.pixmap.width()
            height = self.pixmap.height()

            for x in range(width):
                for y in range(height):
                    pixel = image.pixel(x, y)

                    r, g, b, a = QColor(pixel).getRgb()
                    h, s, v = rgb_to_hsv(r, g, b)
                    s += (ratio / 100)
                    if s > 1:
                        s = 1
                    if s < 0:
                        s = 0
                    r, g, b = hsv_to_rgb(h, s, v)

                    image.setPixel(x, y, qRgb(r, g, b))

            self.pixmap = QPixmap.fromImage(image)
            # self.refresh()
            self.label.setPixmap(self.pixmap)

    def desaturation(self):
        pressed = QPushButton("Desaturation")
        if pressed:
            image = self.pixmap.toImage()
            width = self.pixmap.width()
            height = self.pixmap.height()

            for x in range(width):
                for y in range(height):
                    pixel = image.pixel(x, y)

                    r, g, b, a = QColor(pixel).getRgb()

                    r1 = 0.299 * r + 0.587 * g + 0.114 * b
                    g1 = 0.299 * r + 0.587 * g + 0.114 * b
                    b1 = 0.299 * r + 0.587 * g + 0.114 * b

                    h, s, v = rgb_to_hsv(r1, g1, b1)

                    r1, g1, b1 = hsv_to_rgb(h, s, v)

                    image.setPixel(x, y, qRgb(r1, g1, b1))

            self.pixmap = QPixmap.fromImage(image)
            self.refresh()

    def negative(self):
        # ratio, pressed = QInputDialog.getDouble(self, "Set saturation procentage", "Value:", 100, 0, 200, 2)
        pressed = QPushButton("Negative")
        if pressed:
            image = self.pixmap.toImage()
            width = self.pixmap.width()
            height = self.pixmap.height()

            for x in range(width):
                for y in range(height):
                    pixel = image.pixel(x, y)

                    r, g, b, a = QColor(pixel).getRgb()

                    r1 = 255 - r
                    g1 = 255 - g
                    b1 = 255 - b

                    image.setPixel(x, y, qRgb(r1, g1, b1))

            self.pixmap = QPixmap.fromImage(image)
            self.refresh()

    def brightness(self):
        ratio, pressed = QInputDialog.getDouble(self, "Set value", "Value:", 10, -50, 50, 2)
        if pressed:
            image = self.pixmap.toImage()
            width = self.pixmap.width()
            height = self.pixmap.height()

            for x in range(width):
                for y in range(height):
                    pixel = image.pixel(x, y)

                    r, g, b, a = QColor(pixel).getRgb()
                    h, s, v = rgb_to_hsv(r, g, b)
                    v += (ratio / 100)
                    if v > 1:
                        v = 1
                    if v < 0:
                        v = 0
                    r, g, b = hsv_to_rgb(h, s, v)

                    image.setPixel(x, y, qRgb(r, g, b))

            self.pixmap = QPixmap.fromImage(image)
            self.label.setPixmap(self.pixmap)
            # self.refresh()

    def linearContrast(self):
        ratio, pressed = QInputDialog.getDouble(self, "Set saturation procentage", "Value:", 10, -100, 100, 2)
        if pressed:
            image = self.pixmap.toImage()
            width = self.pixmap.width()
            height = self.pixmap.height()

            f = (259 * (ratio + 255)) / (255 * (259 - ratio))

            for x in range(width):
                for y in range(height):
                    pixel = image.pixel(x, y)

                    r, g, b, a = QColor(pixel).getRgb()

                    r1 = (f * (r - 128)) + 128
                    g1 = (f * (g - 128)) + 128
                    b1 = (f * (b - 128)) + 128

                    h, s, v = rgb_to_hsv(r1, g1, b1)

                    r1, g1, b1 = hsv_to_rgb(h, s, v)

                    image.setPixel(x, y, qRgb(r1, g1, b1))

            self.pixmap = QPixmap.fromImage(image)
            self.label.setPixmap(self.pixmap)

    def invert(self):
        self.refresh()

    def gauss_kernel(self, size, sigma):
        kernel = []
        sum = 0
        radius = size // 2
        #mean = 0
        xPos = 0
        yPos = 0

        for x in range(0, size):
                kernel.append(['x'] * size)

        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                #g = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) / (2 * pi * sigma * sigma)
                g = (0.5 * pi * sigma * sigma) * exp(-0.5 * ((x * x + y * y) / sigma * sigma))
                kernel[xPos][yPos] = g
                yPos += 1
                sum += g
            yPos = 0
            xPos += 1

        for x in range(0, size):
            for y in range(0, size):
                kernel[x][y] /= sum
                kernel[x][y] = round(kernel[x][y], 6)

        return kernel

    def gauss(self):
        size, pressed = QInputDialog.getInt(self, "Set kernel size", "Size:", 3, 3, 10, 2)
        sigma, pressed = QInputDialog.getDouble(self, "Set sigma", "Value: ", 1.0, 0, 10, 1)
        if pressed:
            image = self.pixmap.toImage()
            width = self.pixmap.width()
            height = self.pixmap.height()
            pixmap2 = image.copy()

            radius = size // 2
            kernel = self.gauss_kernel(size, sigma)

            for x in range(width):
                for y in range(height):
                    rSum = 0
                    gSum = 0
                    bSum = 0
                    kernelSum = 0

                    for i in range(size):
                        for j in range(size):
                            xPos = x + i - radius
                            yPos = y + j - radius

                            if xPos < 0 or xPos > width - 1 or yPos < 0 or yPos > height - 1:
                                continue

                            pixel = image.pixel(xPos, yPos)
                            r, g, b, a = QColor(pixel).getRgb()

                            rSum += r * kernel[i][j]
                            gSum += g * kernel[i][j]
                            bSum += b * kernel[i][j]
                            kernelSum += kernel[i][j]

                    r = rSum / kernelSum
                    g = gSum / kernelSum
                    b = bSum / kernelSum
                    pixmap2.setPixel(x, y, qRgb(r, g, b))

            self.pixmap = QPixmap.fromImage(pixmap2)
            self.label.setPixmap(self.pixmap)


    def mean(self):
        size, pressed = QInputDialog.getInt(self, "Set kernel size", "Size:", 3, 3, 10, 2)
        if pressed:
            image = self.pixmap.toImage()
            width = self.pixmap.width()
            height = self.pixmap.height()
            pixmap2 = image.copy()

            radius = size // 2

            for x in range(width):
                for y in range(height):
                    rSum = 0
                    gSum = 0
                    bSum = 0
                    kernelSum = 0

                    for i in range(size):
                        for j in range(size):
                            xPos = x + i - radius
                            yPos = y + j - radius

                            if xPos < 0 or xPos > width - 1 or yPos < 0 or yPos > height - 1:
                                continue

                            pixel = image.pixel(xPos, yPos)
                            r, g, b, a = QColor(pixel).getRgb()

                            rSum += r
                            gSum += g
                            bSum += b
                            kernelSum += 1

                    r = rSum / kernelSum
                    g = gSum / kernelSum
                    b = bSum / kernelSum

                    pixmap2.setPixel(x, y, qRgb(r, g, b))

            self.pixmap = QPixmap.fromImage(pixmap2)
            self.label.setPixmap(self.pixmap)

    def grayscale(self):
        pressed = QPushButton("Grayscale")
        if pressed:
            image = self.pixmap.toImage()
            width = self.pixmap.width()
            height = self.pixmap.height()

            for x in range(width):
                for y in range(height):

                    pixel = image.pixel(x, y)
                    r, g, b, a = QColor(pixel).getRgb()

                    grayscale = 0.299 * r + 0.587 * g + 0.114 * b

                    image.setPixel(x, y, qRgb(grayscale, grayscale, grayscale))

            self.pixmap = QPixmap.fromImage(image)
            self.label.setPixmap(self.pixmap)

    def canny(self):
        # Krok 1
        self.grayscale()
        self.gauss()

        HT, pressed = QInputDialog.getInt(self, "Set high treshold value", "High treshold value:", 20, 0, 100, 1)
        LT, pressed = QInputDialog.getInt(self, "Set low treshold value", "Low treshold value:", 10, 0, 100, 1)
        HT /= 100
        LT /= 100

        if pressed:
            image = self.pixmap.toImage()
            width = self.pixmap.width()
            height = self.pixmap.height()

            kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            kernelY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
            size = 3

            radius = 3 // 2

            magnitude = [0] * width
            angle = [0] * width
            magnitudeSuppressed = [0] * width
            for x in range(width):
                magnitude[x] = [0] * height
                angle[x] = [0] * height
                magnitudeSuppressed[x] = [0] * height

            #Krok 2 i 3 Obliczenie gradientu i katu nachylenia gradientu
            for x in range(width):
                for y in range(height):
                    Gx = 0
                    Gy = 0

                    for i in range(size):
                        for j in range(size):

                            xPos = x + i - radius
                            yPos = y + j - radius

                            if xPos < 0 or xPos > width - 1 or yPos < 0 or yPos > height - 1:
                                continue

                            pixel = image.pixel(xPos, yPos)
                            r, g, b, a = QColor(pixel).getRgb()
                            h, s, v = rgb_to_hsv(r, g, b)

                            Gx += v * kernelX[i][j]
                            Gy += v * kernelY[i][j]

                    mag = sqrt(Gx * Gx + Gy * Gy)
                    magnitude[x][y] = mag
                    #angle[x][y] = atan2(Gy, Gx) + pi
                    angle[x][y] = atan2(Gy, Gx) * 180 / pi
                    if angle[x][y] < 0:
                        angle[x][y] += 180

            #Krok 4 non maxima suppression
            p1 = 0
            p2 = 0
            for x in range(width):
                for y in range(height):
                    if (0 <= angle[x][y] < 22.5) or (157.5 <= angle[x][y] <= 180):
                        if y + 1 > height - 1 or y - 1 < 0:
                            continue
                        p1 = magnitude[x][y + 1]
                        p2 = magnitude[x][y - 1]

                    elif 22.5 <= angle[x][y] < 67.5:
                        if x + 1 > width - 1 or x - 1 < 0 or y + 1 > height - 1 or y - 1 < 0:
                            continue
                        p1 = magnitude[x + 1][y - 1]
                        p2 = magnitude[x - 1][y + 1]

                    elif 67.5 <= angle[x][y] < 112.5:
                        if x + 1 > width - 1 or x - 1 < 0:
                            continue
                        p1 = magnitude[x + 1][y]
                        p2 = magnitude[x - 1][y]

                    elif (112.5 <= angle[x][y] < 157.5):
                        if x + 1 > width - 1 or x - 1 < 0 or y + 1 > height - 1 or y - 1 < 0:
                            continue
                        p1 = magnitude[x - 1][y - 1]
                        p2 = magnitude[x + 1][y + 1]

                    if magnitude[x][y] >= p1 and magnitude[x][y] >= p2:
                        magnitudeSuppressed[x][y] = magnitude[x][y]
                    else:
                        magnitudeSuppressed[x][y] = 0

            # Krok 5: progowanie krawedzi

            #HT = 0.3
            #LT = 0.1
            for x in range(width):
                for y in range(height):
                    if magnitudeSuppressed[x][y] > HT:
                        magnitudeSuppressed[x][y] = 1

                    elif magnitudeSuppressed[x][y] < LT:
                        magnitudeSuppressed[x][y] = 0

                    #Mapa spojnosci

                    #elif x - 1 >= 0 and x + 1 <= width - 1 and y - 1 >= 0 and y + 1 <= width - 1:
                     #   map = magnitudeSuppressed[x - 1][y] * 4 + magnitudeSuppressed[x + 1][y] * 4 + magnitudeSuppressed[x][y - 1] * 4 + magnitudeSuppressed[x][y + 1] * 4
                      #  if map > LT:
                       #     magnitudeSuppressed[x][y] = 0.4
                        #else:
                         #   magnitudeSuppressed[x][y] = 0

                    else:
                        magnitudeSuppressed[x][y] = 0.4

            for x in range(width):
                for y in range(height):
                    if magnitudeSuppressed[x][y] == 0.4:
                        if x - 1 < 0 or x + 1 > width - 1 or y - 1 < 0 or y + 1 > height - 1:
                            continue
                        if magnitudeSuppressed[x - 1][y] == 1 or magnitudeSuppressed[x + 1][y] == 1 or magnitudeSuppressed[x][y - 1] == 1 or magnitudeSuppressed[x][y - 1] == 1 or magnitudeSuppressed[x - 1][y - 1] == 1 or magnitudeSuppressed[x - 1][y + 1] == 1 or magnitudeSuppressed[x + 1][y - 1] == 1 or magnitudeSuppressed[x + 1][y + 1] == 1:
                            magnitudeSuppressed[x][y] = 1
                        else:
                            magnitudeSuppressed[x][y] = 0


            for x in range(width):
                for y in range(height):
                    pixel = image.pixel(x, y)
                    r, g, b, a = QColor(pixel).getRgb()

                    h, s, v = rgb_to_hsv(r, g, b)

                    v = magnitudeSuppressed[x][y]

                    r, g, b = hsv_to_rgb(h, s, v)

                    image.setPixel(x, y, qRgb(r, g, b))



            self.pixmap = QPixmap.fromImage(image)
            self.label.setPixmap(self.pixmap)





    def histogram(self):

        filename = QFileDialog.getOpenFileName(self, 'Open file', './', "Image files (*.pgm *.pbm *.ppm *.png)")

        file = filename[0]

        image = cv2.imread(file, 0)

        histogramm = cv2.calcHist([image], [0], None, [256], [0, 256])

        plt.plot(histogramm)
        plt.show()


def rgb_to_hsv(r, g, b):
    r /= 255
    g /= 255
    b /= 255

    v = max(r, g, b)
    Xmin = min(r, g, b)
    c = v - Xmin

    if c == 0:
        h = 0
    elif v == r:
        h = 60 * (g - b) / c
    elif v == g:
        h = 60 * (2 + ((b - r) / c))  
    elif v == b:
        h = 60 * (4 + ((r - g) / c)) 

    if v == 0:
        s = 0
    else:
        s = c / v

    return h, s, v


def hsv_to_rgb(h, s, v):
    h = h / 60
    c = v * s
    m = v - c
    x = c * (1 - abs((h % 2) - 1))

    if h >= 0 and h <= 1:
        r = c + m
        g = x + m
        b = m
    elif h > 1 and h <= 2:
        r = x + m
        g = c + m
        b = m
    elif h > 2 and h <= 3:
        r = m
        g = c + m
        b = x + m
    elif h > 3 and h <= 4:
        r = m
        g = x + m
        b = c + m
    elif h > 4 and h <= 5:
        r = x + m
        g = m
        b = c + m
    elif h > 5 and h <= 6:
        r = c + m
        g = m
        b = x + m
    else:
        r = m
        g = m
        b = m

    r *= 255
    g *= 255
    b *= 255

    return r, g, b


def main():
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
