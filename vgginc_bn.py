import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms 
import torchvision.models as models
import time
import os
from torchvision.models.vgg import model_urls



"""data = ImageFolder(root = '/data/train', transform= transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
    ]))
trainloader = torch.utils.data.Dataloader(data, batch_size =4, shuffle= True)
"""
class VGG_16(nn.Module):

    def __init__(self, in_chan, no_classes = 0,pretrain = True):
        
        super(VGG_16,self).__init__()
        self.input_chan = in_chan
        self.out_dim = no_classes
        if pretrain:
            model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
            vgg16 = models.vgg16_bn(pretrained=True)
            self.conv_opt = nn.Sequential(*list(vgg16.features))
            self.fc_dims = [7*7*512, 4096, 4096, 1000,128, self.out_dim]	
            self.fc = []
            self.fc_final = []
            self.fc_final0 = []
            self.fc_final1 = []
            self.fc_final2 = [] 
            self.fc_final3 = []
            self.fc_final4 = []
            self.fc_final5 = []  
            self.fc_final6 = []
            self.fc_final7 = []
            self.fc_final8 = []  
   
        else:
            self.conv_channels = [self.input_chan, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            self.fc_dims = [7*7*512, 4096, 4096, 1000,128,self.out_dim]
            self.ker = 3
            self.stride = 1
            self.pad = 1
            self.mp_ker = 2
            self.mp_stride = 2
            self.conv = []
            self.fc = []
            self.fc_final = []
            #self.final = []
            for i in range(len(self.conv_channels)-1):
                self.conv.append(nn.Conv2d(in_channels = self.conv_channels[i], out_channels = self.conv_channels[i+1], kernel_size = self.ker,  stride = self.stride, padding = self.pad))
                self.conv.append(nn.ReLU())
                if(self.conv_channels != self.input_chan and (self.conv_channels[i]-self.conv_channels[i+1]) != 0):
                    self.conv.append(nn.MaxPool2d(kernel_size = self.mp_ker, stride = self.mp_stride))
            self.conv.append(nn.MaxPool2d(kernel_size = self.mp_ker, stride = self.mp_stride))

            self.conv_opt = nn.Sequential(*self.conv)

        for i in range(len(self.fc_dims)-2):
            self.fc.append(nn.Linear(in_features = self.fc_dims[i], out_features = self.fc_dims[i+1]))
            #self.fc.append(nn.Dropout(0.5))
            #if(i != (len(self.fc_dims)-1)):
            self.fc.append(nn.ReLU())
            self.fc.append(nn.Dropout(0.5))
        #self.fc.append(nn.Softmax(dim=1))
        self.fc_opt = nn.Sequential(*self.fc)
        self.fc_final0.append(nn.Linear(128,5))#self.fc_dims[5]))
        self.final0 = nn.Sequential(*self.fc_final0)
        self.fc_final1.append(nn.Linear(128,5))
        self.final1 = nn.Sequential(*self.fc_final1)
        self.fc_final2.append(nn.Linear(128,5))
        self.final2 = nn.Sequential(*self.fc_final2)
        self.fc_final3.append(nn.Linear(128,5))#self.fc_dims[5]))
        self.final3 = nn.Sequential(*self.fc_final3)
        self.fc_final4.append(nn.Linear(128,5))
        self.final4 = nn.Sequential(*self.fc_final4)
        self.fc_final5.append(nn.Linear(128,5))
        self.final5 = nn.Sequential(*self.fc_final5)
        self.fc_final6.append(nn.Linear(128,5))#self.fc_dims[5]))
        self.final6 = nn.Sequential(*self.fc_final6)
        self.fc_final7.append(nn.Linear(128,5))
        self.final7 = nn.Sequential(*self.fc_final7)
        self.fc_final8.append(nn.Linear(128,5))
        self.final8 = nn.Sequential(*self.fc_final8)
 
    def forward(self, x, strm=0):
        #res = []
        x = self.conv_opt(x)
        x = x.view(x.size(0), -1)
        rep = self.fc_opt(x)
        #m = nn.Softmax(dim=1)
        #r = F.relu(self.fc_opt[12])
        #print(np.shape(rep))
        r  = nn.ReLU()
        rep1 = r(rep)
        #n = nn.Linear(128,self.fc_dims[5])
        #res = []
        m = nn.Dropout(0.5)
        out = m(rep1)
        if(strm == 1):
          res = self.final0(out)
          out_dist = res
          val = res
        elif(strm == 2):
          out_dist = self.final0(out)
          res = self.final1(out)
          val = torch.cat((self.final0(m(rep1)),res),1)
        elif(strm == 3):
          out_dist = torch.cat((self.final0(out),self.final1(out)),1)
          res = self.final2(out)
          val = torch.cat((self.final0(m(rep1)),self.final1(m(rep1)),res),1)
        elif(strm == 4):
          out_dist = torch.cat((self.final0(out),self.final1(out),self.final2(out)),1)
          res = self.final3(out)
          val = torch.cat((self.final0(m(rep1)),self.final1(m(rep1)),self.final2(m(rep1)),res),1)
        elif(strm == 5):
          out_dist = torch.cat((self.final0(out),self.final1(out),self.final2(out),self.final3(out)),1)
          res = self.final4(out)
          val = torch.cat((self.final0(m(rep1)),self.final1(m(rep1)),self.final2(m(rep1)),self.final3(m(rep1)),res),1)
        elif(strm == 6):
          out_dist = torch.cat((self.final0(out),self.final1(out),self.final2(out),self.final3(out),self.final4(out)),1)
          res = self.final5(out)
          val = torch.cat((self.final0(m(rep1)),self.final1(m(rep1)),self.final2(m(rep1)),self.final3(m(rep1)),self.final4(m(rep1)),res),1)
        elif(strm == 7):
          out_dist = torch.cat((self.final0(out),self.final1(out),self.final2(out),self.final3(out),self.final4(out),self.final5(out)),1)
          res = self.final6(out)
          val = torch.cat((self.final0(m(rep1)),self.final1(m(rep1)),self.final2(m(rep1)),self.final3(m(rep1)),self.final4(m(rep1)),self.final5(m(rep1)),res),1)
        elif(strm == 8):
          out_dist = torch.cat((self.final0(out),self.final1(out),self.final2(out),self.final3(out),self.final4(out),self.final5(out),self.final6(out)),1)
          res = self.final7(out)
          val = torch.cat((self.final0(m(rep1)),self.final1(m(rep1)),self.final2(m(rep1)),self.final3(m(rep1)),self.final4(m(rep1)),self.final5(m(rep1)),self.final6(m(rep1)),res),1)
        elif(strm == 9):
          out_dist = torch.cat((self.final0(out),self.final1(out),self.final2(out),self.final3(out),self.final4(out),self.final5(out),self.final6(out),self.final7(out)),1)
          res = self.final8(out)
          val = torch.cat((self.final0(m(rep1)),self.final1(m(rep1)),self.final2(m(rep1)),self.final3(m(rep1)),self.final4(m(rep1)),self.final5(m(rep1)),self.final6(m(rep1)),self.final7(m(rep1)),res),1)
 
        elif(strm == 0):
          #while(idx != strm):
          res = torch.cat((self.final0(out),self.final1(out),self.final2(out),self.final3(out),self.final4(out),self.final5(out),self.final6(out),self.final7(out),self.final8(out)),1) 
          #res = np.asarray(res)
          #res = torch.from_numpy(res)
          val = res
          out_dist = res
        #out = n(rep1)
        #res = np.asarray(res)
        #rep = np.asarray(rep)
          #val = np.asarray(val)
          #out_dist = np.asarray(out_dist)
        #out_dist = torch.div(out_dist,2)
        s = nn.Softmax(dim=1)
        res = s(res)
        val = s(val)
        #out_dist =s(out_dist)
        """
        if(dist_id == 1):
          out_dist ={ '1':self.final0(out)}
        elif(dist_id == 2):
          out_dist = {'1':self.final0(out),'2':self.final1(out)}
        elif(dist_id == 3):
          out_dist = {'1':self.final0(out),'2':self.final1(out),'3':self.final2(out)}
        #print(np.shape(x)) 
        #r = m(r)
        """
        #print('d',np.shape(out_dist))
        return (val,rep,val,out_dist)





if __name__ == "__main__":
    rand_data =  torch.rand(1,3,224,224)
    rand_data = Variable(rand_data)
    input_chan = int(input("Enter number of input channels"))
    no_clss = int(input("Enter number of classes"))
    model = VGG_16(input_chan,no_clss)
    clss_prob,rep = model(rand_data)
    clss_prob = (torch.squeeze(clss_prob))
    clss_prob = clss_prob.cpu().detach().numpy()
    clss_label = np.argmax(clss_prob)
    print(clss_label)
    print(torch.mean(rep,1))
    print(np.cov(rep.cpu().detach().numpy()))










