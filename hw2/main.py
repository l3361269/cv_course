import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
from mainwindow import Ui_MainWindow

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import time
import os


class mainWin(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(mainWin,self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()


    def onBindingUI(self): 
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click) 
        self.btn3_1.clicked.connect(self.on_btn3_1_click) 
        self.btn3_2.clicked.connect(self.on_btn3_2_click) 
        self.btn4_1.clicked.connect(self.on_btn4_1_click)


    def on_btn1_1_click(self):
        imgLeft = cv2.imread('imL.png', 0)
        imgRight = cv2.imread('imR.png', 0)
        # Initialize the stereo block matching object 
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
        # Compute the disparity image
        disparity = stereo.compute(imgLeft, imgRight)
        # Normalize the image for representation
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Display the result
        cv2.imshow('disparitytest', np.hstack((imgLeft, imgRight, disparity)))
        

    def on_btn2_1_click(self):
        ALGO='MOG'
        if ALGO == 'MOG2':
            backSub = cv2.createBackgroundSubtractorMOG2()
        else:
            backSub = cv2.createBackgroundSubtractorKNN()

        capture = cv2.VideoCapture('bgSub.mp4')
        if not capture.isOpened:
            print('Unable to open: ' + 'bgSub.mp4')
            exit(0)

        while True:
            ret, frame = capture.read()
            if frame is None:
                break
                
            fgMask = backSub.apply(frame)
            cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            
            cv2.imshow('original vedio', frame)
            cv2.imshow('foreground', fgMask)
            keyboard = cv2.waitKey(30)


    def on_btn3_1_click(self):
        self.capture = cv2.VideoCapture('featureTracking.mp4')
        # Take first frame and find corners in it
        ret, self.old_frame = self.capture.read()
        self.old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
            
        self.p0=[]
        def detect_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.rectangle(self.old_frame,(x-10,y-10),(x+11,y+11),(0,0,255),1)
                user_points = np.empty([1, 1, 2], dtype=float)
                user_points[0][0] = [x,y]
                self.p0.append(user_points[0])
                
        # set the mouse call back
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame",detect_points)

        while (1):
            cv2.imshow('frame', self.old_frame)
            if cv2.waitKey(1)&len(self.p0)==7:
                cv2.imshow('frame', self.old_frame)
                cv2.waitKey(1)
                self.p0=np.array(self.p0,dtype=np.float32)
                break


    def on_btn3_2_click(self):
        # Create some random colors
        color = np.random.randint(0,255,(100,3))
        # Create a mask image for drawing purposes
        mask = np.zeros_like(self.old_frame)
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (21,21),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # start the processing
        p0=self.p0
        old_gray=self.old_gray
        while(1):
            ret,frame = self.capture.read()
            if frame is None:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame=cv2.rectangle(frame,(int(a-10),int(b-10)),(int(a+11),int(b+11)),(0,0,255),1)
            img = cv2.add(frame,mask)
            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff

            if k == 27:
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        #cv2.destroyAllWindows()
        self.capture.release()


    def on_btn4_1_click(self):
        def draw(img, corners, imgpts):
            imgpts = np.int32(imgpts).reshape(-1,2)
            neighbor=imgpts[-1]
            for i in imgpts[1:]:
                img = cv2.line(img, tuple(imgpts[0]), tuple(i),(0,0,255),5)
                img = cv2.line(img, tuple(i), tuple(neighbor),(0,0,255),5)
                neighbor=i
            return img

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        axis = np.float32([[3,3,-4], [1,1,0], [1,5,0], [5,5,0],[5,1,0]])

        os.mkdir('output')
        #for fname in glob.glob('*.bmp'):
        for n in range(1,6):
            fname='{}.bmp'.format(n)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8,11),None)
            
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
                
                # calibrate camera
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
                # Find the rotation and translation vectors.
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

                img = draw(img,corners2,imgpts)
                cv2.imwrite('output/'+fname, img)
                cv2.namedWindow('img_{}'.format(n),cv2.WINDOW_NORMAL)
                cv2.resizeWindow('img_{}'.format(n), img.shape[0]-80,img.shape[1]-80)
                cv2.imshow('img_{}'.format(n),img)

                cv2.waitKey(500)
                time.sleep(0.5)
                cv2.destroyAllWindows()

if __name__=='__main__':
    app=QApplication(sys.argv)
    window = mainWin()
    window.show()
    sys.exit(app.exec_())
