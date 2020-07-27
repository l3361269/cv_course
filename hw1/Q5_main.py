#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import numpy as np
import pandas as pd
from keras.datasets import mnist
import matplotlib.pyplot as plt

def show_image(index_range):
    fig = plt.gcf()
    fig.set_size_inches(15,160)
    ind=1
    for i in index_range:
        ax=plt.subplot(1,10,ind)
        ax.imshow(X_Train[i], cmap='binary')
        ax.set_title('Lable'+str(y_Train[i]), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ind+=1
    plt.show()

(X_Train, y_Train), (X_Test, y_Test) = mnist.load_data()
show_image(np.random.randint(len(X_Train),size=10))


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(1,6,(5,5),padding=2) #->1*32*32
        self.conv3=nn.Conv2d(6,16,(5,5))
        self.conv5=nn.Linear(16*5*5, 120)
        self.fc6=nn.Linear(120,84)
        self.fc7=nn.Linear(84,10)
    def forward(self,x):
        x=F.relu(self.conv1(x)) #->6*28*28
        x=F.max_pool2d(x,kernel_size=(2,2), stride=2) #->6*14*14
        x=F.relu(self.conv3(x)) #->16*10*10
        x=F.max_pool2d(x,kernel_size=(2,2), stride=2) #->16*5*5
        x=x.view(-1, self.num_flat_features(x)) #->16*1*25
        x=F.relu(self.conv5(x)) #->1*120
        x=F.relu(self.fc6(x)) #->1*84
        x=self.fc7(x) #->1*10
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(model,train_loader,device,optimizer,epoch):
    model.train()
    loss_e=[]
    for batch_index, (data,target) in enumerate(train_loader):
        data, target =data.to(device), target.to(device)
        optimizer.zero_grad()
        output=model(data)
        #loss=F.nll_loss(output,target,reduction='mean')
        loss=nn.CrossEntropyLoss()
        loss=loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_index%1000==0:
            print('epoch:',epoch,' training...')
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               #epoch, batch_index * len(data), len(train_loader.dataset),
               #100. * batch_index / len(train_loader), loss.item()))
        loss_e.append(loss.item())
    
    model.eval()
    train_loss=0
    correct=0
    with torch.no_grad():
        for batch_index,(data,target) in enumerate(train_loader):
            data,target=data.to(device),target.to(device)
            output=model(data)
            #test_loss+=F.nll_loss(output,target,reduction='sum').item()  # sum up batch loss
            loss=nn.CrossEntropyLoss(reduction='sum')
            train_loss+=loss(output,target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss/=len(train_loader.dataset)
    #print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        #train_loss, correct, len(train_loader.dataset),
        #100. * correct / len(train_loader.dataset)))
    return loss_e, train_loss, correct / len(train_loader.dataset)
            
def test(model,test_loader,device):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for batch_index,(data,target) in enumerate(test_loader):
            data,target=data.to(device),target.to(device)
            output=model(data)
            #test_loss+=F.nll_loss(output,target,reduction='sum').item()  # sum up batch loss
            loss=nn.CrossEntropyLoss(reduction='sum')
            test_loss+=loss(output,target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss/=len(test_loader.dataset)
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        #test_loss, correct, len(test_loader.dataset),
        #100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

def main(loss,train_loss,train_accu,test_accu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("cuda" if torch.cuda.is_available() else "cpu")
    model=LeNet().to(device)
    batch_size=32
    learning_rate=0.001
    epoch=50
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    

    for e in range(epoch):
        training=train(model,train_loader,device,optimizer,e)
        loss.append(training[0])
        train_loss.append(training[1])
        train_accu.append(training[2])
        test_accu.append(test(model,test_loader,device))
    
    torch.save(model.state_dict(), "mnist_LeNet.pt")

    return loss,train_loss,train_accu,test_accu

if __name__ == '__main__':
    loss=[]
    train_loss=[]
    train_accu=[]
    test_accu=[]
    loss,train_loss,train_accu,test_accu=main(loss,train_loss,train_accu,test_accu)


    plt.plot(range(len(loss[0])),loss[0])
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    plt.show()

    plt.subplot(2,1,1)
    plt.plot(range(len(train_accu)),train_accu ,color='b',label='Training')
    plt.plot(range(len(train_accu)),test_accu ,color='y',label='Testing')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy (%)')
    plt.subplot(2,1,2)
    plt.plot(range(len(train_loss)),train_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()





