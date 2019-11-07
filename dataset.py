import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from vgginc import VGG_16
import torch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#prev = '/home/SharedData/Divakar/project1/curriculum/previous_inc.pth.tar'

class inc_data(Dataset):

    def __init__(self, path, device, prev, stream, clas):

        self.path = path
        self.device = device
        self.prev = prev
        self.stream = str(stream)
        self.clas = str(clas)
        self.path = os.path.join(self.path,self.stream)
        self.path = os.path.join(self.path,self.clas)
        self.mod = 0
        self.model = VGG_16(3).to(self.device)
        self.model.load_state_dict(torch.load(self.prev))
    def __len__(self):
        length = 0
        count = 0 
        #for path in (self.path):
        #print(self.path)
        for _,dirnames,filenames in os.walk(self.path):
                 #count += 1
                 #if(count%10==0):
                  # print("#")
            length += len(filenames)
        #print(length)
        #length = sum([len(files) for r, d, files in os.walk(self.path)])

        return length #len(os.walk(self.path).next[1])

    def __getitem__(self,index):
        
        #mod = 0
        img_path = os.path.join(self.path,str(index)+str('.')+str('jpg'))
        #print(img_path)
        #label = 0
        img = Image.open(img_path).resize((224,224))
        img = np.array(img).transpose(2,0,1)
        img = img.astype(np.float32)/255.0
        #print(self.stream)
        #label = 40+int(self.clas)
        self.mod = (int(self.stream)-1)*5 
        #self.model = self.model.to(self.device)
        #self.model.load_state_dict(torch.load(self.prev))
        #print(self.mod)
        label = (self.mod + int(self.clas))#(int(self.stream)-1)*10 +int(self.clas)
        label = label-1
        self.model.eval()
        with torch.no_grad():
             image = torch.from_numpy(img)
             image =  image.unsqueeze(0)
             image = image.to(self.device)
             _,_,_,dist = self.model(image,int(self.stream))
             #del(image)
             #del(self.model)
             #dist = dist.to('cpu')

        return img,label,dist
