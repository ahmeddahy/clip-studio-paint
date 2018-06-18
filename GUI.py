import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from Motion import Reader
from Motion import Motion
from Motion import SVMClassifier
from Noise import FeaturesExtraction
from Noise import CrossCorrelation
from Noise import Localize
from Noise import NoiseFeatures
from Noise import GaussianMixtureClassifier
from Noise import ForgeryDetermination
from Noise import InputOutput
import cv2
import numpy as np


def start_window():
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QDialog()
    window.setWindowTitle('Video Forgery Detection')
    window.setGeometry(500, 100, 500, 360)
    label2 = QtWidgets.QLabel(window)
    label2.setPixmap(QtGui.QPixmap('background.jpg'))
    label2.resize(500, 360)
    noise_button = QtWidgets.QPushButton(window)
    noise_button.setText('Noise')
    noise_button.resize(400, 100)
    noise_button.move(50, 50)
    noise_button.clicked.connect(Noise_window)
    motion_button = QtWidgets.QPushButton(window)
    motion_button.setText('Motion')
    motion_button.resize(400, 100)
    motion_button.move(50, 210)
    motion_button.clicked.connect(Motion_window)
    window.show()
    app.exec()


class Noise_window(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.title = 'Noise'
        self.left = 450
        self.top = 50
        self.width = 600
        self.height = 500
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.label2 = QtWidgets.QLabel(self)
        self.label2.setPixmap(QtGui.QPixmap('background.jpg'))
        self.label2.resize(600, 500)
        browse_btn = QtWidgets.QPushButton(self)
        browse_btn.setText('Browse')
        browse_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        browse_btn.resize(100, 50)
        browse_btn.move(480, 80)
        browse_btn.clicked.connect(self.Browse_label)
        self.browse_txtbox = QtWidgets.QLineEdit(self)
        self.browse_txtbox.setFont(QtGui.QFont("Times", 15, QtGui.QFont.Bold))
        self.browse_txtbox.resize(440, 50)
        self.browse_txtbox.move(20, 80)
        displayvideo_btn = QtWidgets.QPushButton(self)
        displayvideo_btn.setText('Display Video')
        displayvideo_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        displayvideo_btn.resize(150, 50)
        displayvideo_btn.move(50, 150)
        displayvideo_btn.clicked.connect(self.display_video)
        preprocess_btn = QtWidgets.QPushButton(self)
        preprocess_btn.setText('Preprocessing')
        preprocess_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        preprocess_btn.resize(150, 50)
        preprocess_btn.move(400, 150)
        preprocess_btn.clicked.connect(self.preprocessing)
        noise_features_btn = QtWidgets.QPushButton(self)
        noise_features_btn.setText('Noise Features')
        noise_features_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        noise_features_btn.resize(150, 50)
        noise_features_btn.move(225, 220)
        noise_features_btn.clicked.connect(self.noise_features)
        detect_btn = QtWidgets.QPushButton(self)
        detect_btn.setText('Detect')
        detect_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        detect_btn.resize(150, 50)
        detect_btn.move(225, 400)
        detect_btn.clicked.connect(self.detect)
        self.show()
        self.exec()

    def Browse_label(self):
        window = QtWidgets.QDialog()
        self.path, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Single File', QtCore.QDir.rootPath(),
                                                             '*')
        self.browse_txtbox.setText(self.path)
        InputOutput_obj = InputOutput.Read(self.path)
        InputOutput_obj.ReadVideo()
        self.Video = InputOutput_obj.GetVideo()

    def display_video(self):
        for i in range(1, len(self.Video)):
            cv2.imshow('Video', self.Video[i])
            cv2.waitKey(30)
        cv2.destroyAllWindows()

    def preprocessing(self):
        for i in range(1, len(self.Video)):
            gray_image = cv2.cvtColor(self.Video[i], cv2.COLOR_BGR2GRAY)
            cv2.imshow('Gray Frames In The Video', gray_image)
            cv2.waitKey(30)
        cv2.destroyAllWindows()

    def noise_features(self):
        FeaturesExtraction_obj = FeaturesExtraction.FeaturesExtraction(self.Video)
        FeaturesExtraction_obj.WaveletDenoising()
        for i in range(1, len(self.Video)):
            cv2.imshow('Noise Features In The Video', FeaturesExtraction_obj.DenoisedFrames[i])
            cv2.waitKey(30)
        cv2.destroyAllWindows()

    def detect(self):
        FeaturesExtraction_obj = FeaturesExtraction.FeaturesExtraction(self.Video)
        FeaturesExtraction_obj.WaveletDenoising()
        FeaturesExtraction_obj.CorrelationCoefficient()
        self.Features = FeaturesExtraction_obj.GetCorrelationCoefficient()
        Classify_obj = ForgeryDetermination.Classify(self.Features)
        forged = Classify_obj.GaussianMixture()
        if (forged == True):
            self.fake()
        else:
            self.original()

    def fake(self):
        buttonReply = QtWidgets.QMessageBox.question(self, 'Result',
                                                     "The video is forged" + "\n" + "Do you want to display forged object in the video ?",
                                                     QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                     QtWidgets.QMessageBox.No)
        if buttonReply == QtWidgets.QMessageBox.Yes:
            Localize_obj = Localize.Localize(self.Features)
            Localize_obj.Thershold()
            nVideo, Result = Localize_obj.Localization(self.Video)
            for i in range(1, len(nVideo)):
                cv2.imshow('Forged Video', nVideo[i])
                cv2.waitKey(30)
            cv2.destroyAllWindows()

    def original(self):
        buttonReply = QtWidgets.QMessageBox.about(self, 'Result', "The video is Authentic")


class Motion_window(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.title = 'Motion'
        self.left = 450
        self.top = 50
        self.width = 600
        self.height = 500
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.label2 = QtWidgets.QLabel(self)
        self.label2.setPixmap(QtGui.QPixmap('background.jpg'))
        self.label2.resize(600, 500)
        browse_btn = QtWidgets.QPushButton(self)
        browse_btn.setText('Browse')
        browse_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        browse_btn.resize(100, 50)
        browse_btn.move(480, 80)
        browse_btn.clicked.connect(self.Browse_label)
        self.browse_txtbox = QtWidgets.QLineEdit(self)
        self.browse_txtbox.setFont(QtGui.QFont("Times", 15, QtGui.QFont.Bold))
        self.browse_txtbox.resize(440, 50)
        self.browse_txtbox.move(20, 80)
        displayvideo_btn = QtWidgets.QPushButton(self)
        displayvideo_btn.setText('Display Video')
        displayvideo_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        displayvideo_btn.resize(150, 50)
        displayvideo_btn.move(50, 150)
        displayvideo_btn.clicked.connect(self.display_video)
        preprocess_btn = QtWidgets.QPushButton(self)
        preprocess_btn.setText('Preprocessing')
        preprocess_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        preprocess_btn.resize(150, 50)
        preprocess_btn.move(400, 150)
        preprocess_btn.clicked.connect(self.preprocessing)
        baseframe_btn = QtWidgets.QPushButton(self)
        baseframe_btn.setText('Base Frame')
        baseframe_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        baseframe_btn.resize(150, 50)
        baseframe_btn.move(50, 300)
        baseframe_btn.clicked.connect(self.calc_base_frame)
        Motionvideo_btn = QtWidgets.QPushButton(self)
        Motionvideo_btn.setText('Motion Video')
        Motionvideo_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        Motionvideo_btn.resize(150, 50)
        Motionvideo_btn.move(400, 300)
        Motionvideo_btn.clicked.connect(self.Motion_Residue)
        detect_btn = QtWidgets.QPushButton(self)
        detect_btn.setText('Detect')
        detect_btn.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        detect_btn.resize(150, 50)
        detect_btn.move(225, 400)
        detect_btn.clicked.connect(self.detect)
        self.show()
        self.exec()

    def Browse_label(self):
        window = QtWidgets.QDialog()
        self.path, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Single File', QtCore.QDir.rootPath(),
                                                             '*')
        self.browse_txtbox.setText(self.path)
        InputOutput_obj = InputOutput.Read(self.path)
        InputOutput_obj.ReadVideo()
        self.Video = InputOutput_obj.GetVideo()

    '''def display_video(self):
        self.__frames, self.__fps = Reader.read_video(path)  # get video frames and frame rat'''

    def display_video(self):
        for i in range(1, len(self.Video)):
            cv2.imshow('Video', self.Video[i])
            cv2.waitKey(30)
        cv2.destroyAllWindows()

    def preprocessing(self):
        for i in range(1, len(self.Video)):
            gray_image = cv2.cvtColor(self.Video[i], cv2.COLOR_BGR2GRAY)
            cv2.imshow('Gray Frames In The Video', gray_image)
            cv2.waitKey(30)
        cv2.destroyAllWindows()

    def calc_base_frame(self):
        gray_frames = []
        for i in range(0, len(self.Video)):
            gray_image = cv2.cvtColor(self.Video[i], cv2.COLOR_BGR2GRAY)
            gray_image = np.array(gray_image).astype(int)
            gray_frames.append(gray_image)
        m = Motion()
        Baseframe = m.calcBaseFrame(gray_frames)
        Baseframe = np.uint8(Baseframe)
        cv2.imshow('Base Frame', Baseframe)
        cv2.waitKey(1000)

    def Motion_Residue(self):
        gray_frames = []
        for i in range(0, len(self.Video)):
            gray_image = cv2.cvtColor(self.Video[i], cv2.COLOR_BGR2GRAY)
            gray_image = np.array(gray_image).astype(int)
            gray_frames.append(gray_image)
        m = Motion()
        Baseframe = m.calcBaseFrame(gray_frames)
        Baseframe = np.uint8(Baseframe)
        for i in range(0, len(gray_frames)):
            frameResidue = gray_frames[i] - Baseframe
            frameResidue = np.uint8(frameResidue)
            cv2.imshow('Motion Residue', frameResidue)
            cv2.waitKey(30)
        cv2.destroyAllWindows()

    def detect(self):
        m = Motion()
        m.computeFrameFeatures(self.path)
        seconds = m.getFakeTime()
        if (seconds.__len__() > 0):
            self.fake(seconds)
        else:
            self.original()

    def fake(self, seconds):
        buttonReply = QtWidgets.QMessageBox.question(self, 'Result',
                                                     "Do you want to display forged frames in the video ?",
                                                     QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                     QtWidgets.QMessageBox.No)
        if buttonReply == QtWidgets.QMessageBox.Yes:
            nvideo = self.Video
            for i in range(0, len(seconds)):
                fra = round(seconds[i])
                fra *= 30
                for j in range(fra, fra + 30):
                    cv2.rectangle(nvideo[j], (2, 2), (318, 238), (0, 0, 255), 2)
            for i in range(1, len(nvideo)):
                cv2.imshow('Forged Video', nvideo[i])
                cv2.waitKey(30)
            cv2.destroyAllWindows()

    def original(self):

        buttonReply = QtWidgets.QMessageBox.about(self, 'Result', "The video is Authentic")


start_window()
