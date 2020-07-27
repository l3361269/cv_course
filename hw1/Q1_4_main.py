# -*- coding: utf-8 -*-

import sys
import math
from Q1_4_gui import Ui_MainWindow, Ui_InputWindow
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication
import numpy as np
from scipy import signal


class InputWindow(QMainWindow, Ui_InputWindow):
    def __init__(self, parent=None):
        super(InputWindow, self).__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.onBindingUI()
        self.img = None
        self.degree_img = None

    def onBindingUI(self):
        self.edtAngle.textChanged.connect(self.sync_lineEdit)
    
    def sync_lineEdit(self, text):
        height = self.img.shape[0]
        width = self.img.shape[1]

        get_input = float(text) - 180
        print('get:', end=' ')
        print(get_input)
        
        get_range_min = get_input - 10
        get_range_max = get_input + 10

        flat_img = self.img.flatten()
        for i in range(height*width):
            if self.degree_img[i] > get_range_max or self.degree_img[i] < get_range_min:
                flat_img[i] = 0
        cv2.imshow('show range img', np.reshape(flat_img, (height, width)))
      

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()
        self.original_img = None
        self.img = None
        self.flipped_img = None
        self.mainwindow2 = None
        self.window_name = ''
        self.pos_record = [0, 0, 0, 0]
        self.current_record_ix = 0

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click) #4-1
        self.btn2_2.clicked.connect(self.on_btn2_2_click) #4-2
        self.btn3_1.clicked.connect(self.on_btn3_1_click) #5-1
        self.btn3_2.clicked.connect(self.on_btn3_2_click) #5-2
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn4_3.clicked.connect(self.on_btn4_3_click)
        self.btn4_4.clicked.connect(self.on_btn4_4_click)

    # button for loading image
    def on_btn1_1_click(self):
        filename = './images/dog.bmp'
        img = cv2.imread(filename)
        height = img.shape[0]
        width = img.shape[1]
        print('loading image : ' + filename)
        print('Height = %d' %(height))
        print('Width = %d' %(width))
        cv2.imshow('dog image', img)
        


    def on_btn1_2_click(self):
        filename = './images/color.png'
        srcBGR = cv2.imread(filename)
        destRBG = srcBGR[...,[1,2,0]]
        cv2.imshow('converted image', destRBG)
        

    def on_btn1_3_click(self):
        filename = './images/dog.bmp'
        img = cv2.imread(filename)
        horizontal_img = cv2.flip(img, 1)
        cv2.imshow('flipped image', horizontal_img)
        

    def on_btn1_4_click(self):
        filename = './images/dog.bmp'
        self.img = cv2.imread(filename)
        self.flipped_img = cv2.flip(self.img, 1)
        self.window_name = 'image blending'
        
        cv2.imshow(self.window_name, self.flipped_img)
        cv2.createTrackbar('ratio', self.window_name, 0, 100, self.on_trackBar_changed)
        

    def on_trackBar_changed(self, val):
        img_ratio = val / 100
        blended_img = cv2.addWeighted(self.img, img_ratio, self.flipped_img, 1-img_ratio, 0)
        cv2.imshow(self.window_name, blended_img)
        

    def on_btn2_1_click(self):
        filename = './images/QR.png'
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # 300 * 300 / grayscale
        cv2.imshow('original image', img)
        retval, dst_img	= cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow('thresholded image', dst_img)
        


    def on_btn2_2_click(self):
        filename = './images/QR.png'
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # 300 * 300 / grayscale
        cv2.imshow('original image', img)
        dst_img	= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -1)
        cv2.imshow('thresholded image', dst_img)
        


    def on_btn4_1_click(self):
        filename = './images/School.jpg'
        img = cv2.imread(filename)
        img_B=img[:,:,0]
        img_G=img[:,:,1]
        img_R=img[:,:,2]
        img_Gray=img_B*0.114+img_G*0.587+img_R*0.299
        height = img.shape[0]
        width = img.shape[1]
        #3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        #filter (convolution)
        bordered_img = cv2.copyMakeBorder(img_Gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE) 
        gray_gaussian_img = np.zeros((height, width))
        for y in range(height):
            tmp1 = np.convolve(gaussian_kernel[0], bordered_img[y], 'same')
            tmp2 = np.convolve(gaussian_kernel[1], bordered_img[y+1], 'same')
            tmp3 = np.convolve(gaussian_kernel[2], bordered_img[y+2], 'same')
            for x in range(width):
                gray_gaussian_img[y][x] = tmp1[x] + tmp2[x] + tmp3[x]
        max_value = max(gray_gaussian_img.flatten())
        min_value = min(gray_gaussian_img.flatten())
        total_range = max_value - min_value
        tmp_arr = [ int( (i-min_value)/total_range*255 ) for i in gray_gaussian_img.flatten() ]
        gray_gaussian_img= np.array(tmp_arr, dtype = np.uint8)
        gray_gaussian_img= np.reshape(gray_gaussian_img, (height, width))

        cv2.imshow('Gaussian smooth filter', gray_gaussian_img)
    
    def on_btn4_2_click(self):
        filename = './images/School.jpg'
        img = cv2.imread(filename)
        img_B=img[:,:,0]
        img_G=img[:,:,1]
        img_R=img[:,:,2]
        img_Gray=img_B*0.114+img_G*0.587+img_R*0.299
        height = img.shape[0]
        width = img.shape[1]
        
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        vertical_edge_img = np.zeros((height, width)) # 300 * 300 / grayscale
        bordered_img = cv2.copyMakeBorder(img_Gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE) # 302 * 302 / grayscale

        # sobel-x
        for y in range(height):
            c1 = np.convolve(sobel_x[0], bordered_img[y], 'valid')
            c2 = np.convolve(sobel_x[1], bordered_img[y+1], 'valid')
            c3 = np.convolve(sobel_x[2], bordered_img[y+2], 'valid')
            for x in range(width):
                vertical_edge_img[y][x] = c1[x] + c2[x] + c3[x]

        vertical_edge_img = np.absolute(vertical_edge_img)
        #normalize
        max_value = max(vertical_edge_img.flatten())
        min_value = min(vertical_edge_img.flatten())
        total_range = max_value - min_value
        tmp_arr = [ int( (i-min_value)/total_range*255 ) for i in vertical_edge_img.flatten() ]
        vertical_norm_flat_img = np.array(tmp_arr, dtype = np.uint8)
        vertical_norm_img = np.reshape(vertical_norm_flat_img, (height, width))
        
        cv2.imshow('vertical edge', vertical_norm_img)

    def on_btn4_3_click(self):
        filename = './images/School.jpg'
        img = cv2.imread(filename)
        img_B=img[:,:,0]
        img_G=img[:,:,1]
        img_R=img[:,:,2]
        img_Gray=img_B*0.114+img_G*0.587+img_R*0.299
        height = img.shape[0]
        width = img.shape[1]
        
        sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        horizontal_edge_img = np.zeros((height, width)) # 300 * 300 / grayscale
        bordered_img = cv2.copyMakeBorder(img_Gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE) # 302 * 302 / grayscale
        # sobel-y
        for y in range(height):
            c1 = np.convolve(sobel_y[0], bordered_img[y], 'valid')
            c2 = np.convolve(sobel_y[2], bordered_img[y+2], 'valid')
            for x in range(width):
                horizontal_edge_img[y][x] = c1[x] + c2[x]

        horizontal_edge_img = np.absolute(horizontal_edge_img)
        max_value = max(horizontal_edge_img.flatten())
        min_value = min(horizontal_edge_img.flatten())
        total_range = max_value - min_value

        tmp_arr = [ int( (i-min_value)/total_range*255 ) for i in horizontal_edge_img.flatten() ]
        horizontal_norm_flat_img = np.array(tmp_arr, dtype = np.uint8)
        horizontal_norm_img = np.reshape(horizontal_norm_flat_img, (height, width))
        
        cv2.imshow('horizontal edge', horizontal_norm_img)

    def on_btn4_4_click(self):
        filename = './images/School.jpg'
        img = cv2.imread(filename)
        img_B=img[:,:,0]
        img_G=img[:,:,1]
        img_R=img[:,:,2]
        img_Gray=img_B*0.114+img_G*0.587+img_R*0.299
        height = img.shape[0]
        width = img.shape[1]
        
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        vertical_edge_img = np.zeros((height, width)) # 300 * 300 / grayscale
        bordered_img = cv2.copyMakeBorder(img_Gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE) # 302 * 302 / grayscale

        # sobel-x
        for y in range(height):
            c1 = np.convolve(sobel_x[0], bordered_img[y], 'valid')
            c2 = np.convolve(sobel_x[1], bordered_img[y+1], 'valid')
            c3 = np.convolve(sobel_x[2], bordered_img[y+2], 'valid')
            for x in range(width):
                vertical_edge_img[y][x] = c1[x] + c2[x] + c3[x]

        vertical_edge_img = np.absolute(vertical_edge_img)
        #normalize
        max_value = max(vertical_edge_img.flatten())
        min_value = min(vertical_edge_img.flatten())
        total_range = max_value - min_value
        tmp_arr = [ int( (i-min_value)/total_range*255 ) for i in vertical_edge_img.flatten() ]
        vertical_norm_flat_img = np.array(tmp_arr, dtype = np.uint8)
        vertical_norm_img = np.reshape(vertical_norm_flat_img, (height, width))
        
        sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        horizontal_edge_img = np.zeros((height, width)) # 300 * 300 / grayscale
        bordered_img = cv2.copyMakeBorder(img_Gray, 1, 1, 1, 1, cv2.BORDER_REPLICATE) # 302 * 302 / grayscale
        # sobel-y
        for y in range(height):
            c1 = np.convolve(sobel_y[0], bordered_img[y], 'valid')
            c2 = np.convolve(sobel_y[2], bordered_img[y+2], 'valid')
            for x in range(width):
                horizontal_edge_img[y][x] = c1[x] + c2[x]

        horizontal_edge_img = np.absolute(horizontal_edge_img)
        max_value = max(horizontal_edge_img.flatten())
        min_value = min(horizontal_edge_img.flatten())
        total_range = max_value - min_value

        tmp_arr = [ int( (i-min_value)/total_range*255 ) for i in horizontal_edge_img.flatten() ]
        horizontal_norm_flat_img = np.array(tmp_arr, dtype = np.uint8)
        horizontal_norm_img = np.reshape(horizontal_norm_flat_img, (height, width))
        
        #combine x and y
        combined_img = np.sqrt(cv2.add(abs(0.5*np.float32(horizontal_norm_img)), abs(0.5*np.float32(vertical_norm_img))))

        abs_combined_img = np.absolute(combined_img)
        max_value = max(abs_combined_img.flatten())
        min_value = min(abs_combined_img.flatten())
        total_range = max_value - min_value
        tmp_arr = [ int( ((i-min_value)/total_range) * 255 ) for i in abs_combined_img.flatten() ] # normalize

        thres_flat_img = np.array(tmp_arr, dtype = np.uint8)
        thres_img = np.reshape(thres_flat_img, (height, width))
        cv2.imshow('magnitude', thres_img)


    def on_btn3_1_click(self):
        # edtAngle, edtScale. edtTx, edtTy to access to the ui object
        filename = './images/OriginalTransform.png'
        img = cv2.imread(filename)
        cv2.imshow('original image', img)
        height = img.shape[0]
        width = img.shape[1]

        get_angle = float(self.edtAngle.text())
        get_scale = float(self.edtScale.text())
        get_tx = float(self.edtTx.text())
        get_ty = float(self.edtTy.text())
        central_x = 130
        central_y = 125

        rotate_matrix = cv2.getRotationMatrix2D(((central_x, central_y)), get_angle, 1)
        rot_img = cv2.warpAffine(img, rotate_matrix, (width, height))
        rot_sca_img = cv2.resize(rot_img, None, fx=get_scale, fy=get_scale, interpolation=cv2.INTER_LINEAR)
        translation_matrix = np.float32([[1,0,get_tx],[0,1,get_ty]])
        rot_sca_tra_img = cv2.warpAffine(rot_sca_img, translation_matrix, (width, height))
        cv2.imshow('rotation, scaling & translation img', rot_sca_tra_img)
        

    def on_btn3_2_click(self):
        filename = './images/OriginalPerspective.png'
        img = cv2.imread(filename)
        self.window_name = 'original image'
        self.original_img = img.copy()
        self.img = img
        cv2.imshow(self.window_name, img)
        cv2.setMouseCallback(self.window_name, self.get_mouse_position)

    def get_mouse_position(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.pos_record[self.current_record_ix] = [x, y]
            cv2.circle(self.img,(x,y), 20, (255,0,0), -1)
            cv2.imshow(self.window_name, self.img)
            self.current_record_ix += 1
            self.current_record_ix %= 4
            if self.current_record_ix == 0:
                pts1 = np.float32(self.pos_record)
                pts2 = np.float32([[20, 20],[450, 20],[450, 450],[20, 450]])

                perspective_matrix = cv2.getPerspectiveTransform(pts1,pts2)
                dst_img = cv2.warpPerspective(self.original_img, perspective_matrix, (470, 470))
                cv2.imshow('perspective result image', dst_img)
                

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

