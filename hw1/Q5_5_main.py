import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.uic import loadUi
from Q5_5_gui import Ui_MainWindow
#from Q5_main import LeNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(1,6,(5,5),padding=2) #->32*32
        self.conv3=nn.Conv2d(6,16,(5,5))
        self.conv5=nn.Linear(16*5*5, 120)
        self.fc6=nn.Linear(120,84)
        self.fc7=nn.Linear(84,10)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,kernel_size=(2,2), stride=2)
        x=F.relu(self.conv3(x))
        x=F.max_pool2d(x,kernel_size=(2,2), stride=2)
        x=x.view(-1, self.num_flat_features(x))
        x=F.relu(self.conv5(x))
        x=F.relu(self.fc6(x))
        x=self.fc7(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

from torch.autograd import Variable 
data=datasets.MNIST('./data', train=False, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = Variable(torch.unsqueeze(data.test_data, dim=1).float(), requires_grad=False)

model = LeNet()
model.load_state_dict(torch.load('mnist_LeNet.pt'))
model.eval()


class mainWin(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(mainWin,self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()
    
    def onBindingUI(self): 
        self.pushButton.clicked.connect(self.on_btn_click)
        self.lineEdit.textChanged.connect(self.getText)
    

    def getText(self,text):
        self.text=text
        #print('t',self.text)
        #print(self.text)
        self.inf=int(self.text)
        #print(inf)
        self.output=model(data[self.inf].view(1,1,28,28))
        self.output=F.softmax(self.output)
        pred = self.output.argmax(dim=1, keepdim=True)
    
    def on_btn_click(self):
        #print('ok')
        #print(pred)
        plt.figure()
        plt.imshow(data[self.inf].view(28,28))
        plt.show()
        plt.figure()
        x=[0,1,2,3,4,5,6,7,8,9]
        plt.bar(x, self.output.detach().numpy()[0])
        plt.xticks(ticks=x)
        plt.show()


if __name__=='__main__':
    app=QApplication(sys.argv)
    #main_win=loadUi('Q5_gui.ui')
    window = mainWin()
    #print(window.lineEdit.text())
    #main_win.pushButton.clicked.connect(predict)
    #print('predict')
    #print(main_win.lineEdit.text())
    window.show()
    sys.exit(app.exec_())
