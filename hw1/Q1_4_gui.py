# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\hw1.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(718, 415)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.btn1_1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn1_1.setGeometry(QtCore.QRect(40, 50, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.btn1_1.setFont(font)
        self.btn1_1.setCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.btn1_1.setCheckable(False)
        self.btn1_1.setObjectName("btn1_1")
        self.btn1_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn1_2.setGeometry(QtCore.QRect(40, 100, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.btn1_2.setFont(font)
        self.btn1_2.setCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.btn1_2.setCheckable(False)
        self.btn1_2.setObjectName("btn1_2")
        self.btn1_3 = QtWidgets.QPushButton(self.centralwidget)
        self.btn1_3.setGeometry(QtCore.QRect(40, 150, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.btn1_3.setFont(font)
        self.btn1_3.setCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.btn1_3.setCheckable(False)
        self.btn1_3.setObjectName("btn1_3")        
        self.btn1_4 = QtWidgets.QPushButton(self.centralwidget)
        self.btn1_4.setGeometry(QtCore.QRect(40, 200, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.btn1_4.setFont(font)
        self.btn1_4.setCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        self.btn1_4.setCheckable(False)
        self.btn1_4.setObjectName("btn1_4")
        
        self.btn2_1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn2_1.setGeometry(QtCore.QRect(240, 50, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn2_1.setFont(font)
        self.btn2_1.setObjectName("btn2_1")
        self.btn2_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn2_2.setGeometry(QtCore.QRect(240, 100, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn2_2.setFont(font)
        self.btn2_2.setObjectName("btn2_2")
        
        self.btn4_1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn4_1.setGeometry(QtCore.QRect(240, 180, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_1.setFont(font)
        self.btn4_1.setObjectName("btn4_1")
        self.btn4_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn4_2.setGeometry(QtCore.QRect(240, 230, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_2.setFont(font)
        self.btn4_2.setObjectName("btn4_2")
        self.btn4_3 = QtWidgets.QPushButton(self.centralwidget)
        self.btn4_3.setGeometry(QtCore.QRect(240, 280, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_3.setFont(font)
        self.btn4_3.setObjectName("btn4_3")
        self.btn4_4 = QtWidgets.QPushButton(self.centralwidget)
        self.btn4_4.setGeometry(QtCore.QRect(240, 330, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_4.setFont(font)
        self.btn4_4.setObjectName("btn4_4")

        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(430, 50, 261, 321))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_3.setObjectName("frame_3")
        
        self.frame_5 = QtWidgets.QFrame(self.frame_3)                        
        self.frame_5.setGeometry(QtCore.QRect(10, 30, 241, 231))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_5.setObjectName("frame_5")
        
        self.frame_6 = QtWidgets.QFrame(self.frame_5)                      
        self.frame_6.setGeometry(QtCore.QRect(10, 30, 221, 141))
        self.frame_6.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_6.setObjectName("frame_6")

        self.btn3_1 = QtWidgets.QPushButton(self.frame_5)                   
        self.btn3_1.setGeometry(QtCore.QRect(20, 180, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn3_1.setFont(font)
        self.btn3_1.setObjectName("btn3_1")
        self.btn3_2 = QtWidgets.QPushButton(self.frame_3)
        self.btn3_2.setGeometry(QtCore.QRect(30, 270, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn3_2.setFont(font)
        self.btn3_2.setObjectName("btn3_2")

        
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(20, 20, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(220, 20, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(420, 20, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(220, 140, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        
        
        self.label_3_1 = QtWidgets.QLabel(self.frame_3)
        self.label_3_1.setGeometry(QtCore.QRect(10, 10, 21, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_3_1.setFont(font)
        self.label_3_1.setObjectName("label_3_1")

        

        self.label_3_p = QtWidgets.QLabel(self.frame_5)                      ##################
        self.label_3_p.setGeometry(QtCore.QRect(10, 0, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3_p.setFont(font)
        self.label_3_p.setObjectName("label_3_p")
        
        self.label_3_a = QtWidgets.QLabel(self.frame_6)                  ##########################
        self.label_3_a.setGeometry(QtCore.QRect(10, 10, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3_a.setFont(font)
        self.label_3_a.setObjectName("label_3_a")
        
        self.label_3_s = QtWidgets.QLabel(self.frame_6)                   ########################
        self.label_3_s.setGeometry(QtCore.QRect(10, 40, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3_s.setFont(font)
        self.label_3_s.setObjectName("label_3_s")
        
        self.label_3_x = QtWidgets.QLabel(self.frame_6)
        self.label_3_x.setGeometry(QtCore.QRect(10, 70, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3_x.setFont(font)
        self.label_3_x.setObjectName("label_3_x")
        
        self.label_3_y = QtWidgets.QLabel(self.frame_6)
        self.label_3_y.setGeometry(QtCore.QRect(10, 100, 47, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3_y.setFont(font)
        self.label_3_y.setObjectName("label_3_y")

        self.label_3_d = QtWidgets.QLabel(self.frame_6)
        self.label_3_d.setGeometry(QtCore.QRect(180, 10, 21, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_3_d.setFont(font)
        self.label_3_d.setObjectName("label_3_d")
        
        self.label_3_p_x = QtWidgets.QLabel(self.frame_6)
        self.label_3_p_x.setGeometry(QtCore.QRect(180, 70, 31, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_3_p_x.setFont(font)
        self.label_3_p_x.setObjectName("label_3_p_x")
        
        self.label_3_p_y = QtWidgets.QLabel(self.frame_6)
        self.label_3_p_y.setGeometry(QtCore.QRect(180, 100, 31, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        self.label_3_p_y.setFont(font)
        self.label_3_p_y.setObjectName("label_3_p_y")
        
        self.edtAngle = QtWidgets.QLineEdit(self.frame_6)
        self.edtAngle.setGeometry(QtCore.QRect(60, 10, 113, 20))
        self.edtAngle.setObjectName("edtAngle")
        self.edtScale = QtWidgets.QLineEdit(self.frame_6)
        self.edtScale.setGeometry(QtCore.QRect(60, 40, 113, 20))
        self.edtScale.setObjectName("edtScale")
        self.edtTx = QtWidgets.QLineEdit(self.frame_6)
        self.edtTx.setGeometry(QtCore.QRect(60, 70, 113, 20))
        self.edtTx.setObjectName("edtTx")
        self.edtTy = QtWidgets.QLineEdit(self.frame_6)
        self.edtTy.setGeometry(QtCore.QRect(60, 100, 113, 20))
        self.edtTy.setObjectName("edtTy")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 718, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn1_1.setText(_translate("MainWindow", "1.1 Load Image"))
        self.btn1_2.setText(_translate("MainWindow", "1.2 Color Conversion"))
        self.btn1_3.setText(_translate("MainWindow", "1.3 Image Flipping"))
        self.btn1_4.setText(_translate("MainWindow", "1.4 Blending"))
        self.btn2_1.setText(_translate("MainWindow", "2.1 Global Threshold"))
        self.btn2_2.setText(_translate("MainWindow", "2.2 Local Threshold"))
        self.btn3_1.setText(_translate(
            "MainWindow", "3.1 Rotation, Scaling, Translation"))
        self.btn3_2.setText(_translate(
            "MainWindow", "3.2 Perspective Transform"))
        self.btn4_1.setText(_translate("MainWindow", "4.1 Gaussian"))
        self.btn4_2.setText(_translate("MainWindow", "4.2 Sobel X"))
        self.btn4_3.setText(_translate("MainWindow", "4.3 Sobel Y"))
        self.btn4_4.setText(_translate("MainWindow", "4.4 Magnitude"))
        
        self.label_1.setText(_translate("MainWindow", "1. Image Processing"))
        self.label_2.setText(_translate("MainWindow", "2. Adaptive Threshold"))
        self.label_3.setText(_translate(
            "MainWindow", "3. Image Transformation"))
        self.label_3_1.setText(_translate("MainWindow", "3.1 Rot, Scale, Translate"))
        self.label_3_p.setText(_translate("MainWindow", "Parameters"))
        self.label_3_a.setText(_translate("MainWindow", "Angle:"))
        self.label_3_s.setText(_translate("MainWindow", "Scale:"))
        self.label_3_x.setText(_translate("MainWindow", "Tx:"))
        self.label_3_y.setText(_translate("MainWindow", "Ty:"))
        self.label_3_d.setText(_translate("MainWindow", "deg"))
        self.label_3_p_x.setText(_translate("MainWindow", "pixel"))
        self.label_3_p_y.setText(_translate("MainWindow", "pixel"))
        self.label_4.setText(_translate("MainWindow", "4. Convolution"))



class Ui_InputWindow(object):
    def setupUi(self, InputWindow):
        InputWindow.setObjectName("InputWindow")
        InputWindow.resize(260, 50)
        self.centralwidget = QtWidgets.QWidget(InputWindow)
        self.centralwidget.setObjectName("centralwidget")

        #self.edtDirection = QtWidgets.QInputDialog(self.centralwidget)
        #self.edtDirection.setGeometry(QtCore.QRect(120, 10, 113, 30))
        #self.edtDirection.setObjectName("edtAngle")
        self.edtAngle = QtWidgets.QLineEdit(self.centralwidget)
        self.edtAngle.setGeometry(QtCore.QRect(120, 10, 113, 30))
        self.edtAngle.setObjectName("edtAngle")
        
        InputWindow.setCentralWidget(self.centralwidget)
        #self.menubar = QtWidgets.QMenuBar(InputWindow)
        #self.menubar.setGeometry(QtCore.QRect(0, 0, 200, 21))
        #self.menubar.setObjectName("menubar")
        #InputWindow.setMenuBar(self.menubar)
        #self.statusbar = QtWidgets.QStatusBar(InputWindow)
        #self.statusbar.setObjectName("statusbar")
        #InputWindow.setStatusBar(self.statusbar)

        self.retranslateUi(InputWindow)
        QtCore.QMetaObject.connectSlotsByName(InputWindow)

    def retranslateUi(self, InputWindow):
        _translate = QtCore.QCoreApplication.translate
        InputWindow.setWindowTitle(_translate("InputWindow", "InputWindow"))
        #self.label_2.setText(_translate("InputWindow", "direction:"))
