import os
import sys
import shutil
import argparse
import time

import math
import numpy as np
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

from model_vgg import VGG_16
from logger import Tensorboard
from Bc import BCoeff
from cosine import cos
from Curr_data import curric_data
from dataset_bn import inc_data
from data import curric_data_shuff 
from ImageFolder_bn import ImageFolder1
from MemFolder_bn import DistFolder 

parser = argparse.ArgumentParser(description='VGG16 Training')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint(default: none)')
parser.add_argument('--gpu', default='cuda:0', type=str, metavar='gpu_id', help='GPU id ')
parser.add_argument('--batchsize', default='1', type = int, metavar='batch_size', help='size of batch')
parser.add_argument('--lr', default='1e-06', type=float, metavar='lr', help='learning rate')
parser.add_argument('--streams', default='9', type=int, metavar='streams', help='number of data streams')
parser.add_argument('--incstep',default='5', type=int, metavar='incstep', help='number of classes per incremental step')
parser.add_argument('--max_epoch', default='40', type=int, metavar='max_epoch', help='maximum number of base epochs in each stream')
parser.add_argument('--wd', default='1e-04',type=float, metavar='wd', help='weight decay')
args = parser.parse_args()

#device
device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
print(device)
#device = torch.device("cuda:1"if torch.cuda.is_available() else "cpu")

#define the paths to data directories
main = '/home/SharedData/Divakar/project1/data_step_3/train'
val = '/home/SharedData/Divakar/project1/data_step_3/validation'
fullval = '/home/SharedData/Divakar/project1/data_step_3/validation'
mem = '/home/SharedData/Divakar/project1/data_step_3/mem_step3'
shuff = '/home/SharedData/Divakar/project1/data_step_3/temp_step3'

#paths to checkpoints and saved model
save_path = '/home/SharedData/Divakar/project1/curriculum/ablation/saved_step3.pth.tar'
PATH = '/home/SharedData/Divakar/project1/curriculum/ablation/checkpoint_step3.pth.tar'
best = '/home/SharedData/Divakar/project1/curriculum/ablation/best_step3.pth.tar'
old = '/home/SharedData/Divakar/project1/curriculum/ablation/previous_step3.pth.tar'
tensorboard = Tensorboard('/home/SharedData/Divakar/project1/curriculum/ablation_logs/step_3_oct10')
wd = args.wd
batch = args.batchsize
lr = args.lr
streams = args.streams
inc_step = args.incstep
max_epoch = args.max_epoch

valdata = ImageFolder(root = fullval, transform = transforms.Compose([
transforms.Resize(224), 
transforms.ToTensor()
 ]))
valloader = torch.utils.data.DataLoader(valdata, batch_size=batch,shuffle = True)

#num_of_curr_classes = 3
#prev = 3

sel = 0.3
is_check = False
#streams = 9
dim = 128
num_of_clas = np.repeat(inc_step,streams)#[5,5,5,5,5,5,5,5,5]
list_of_dataloaders = []
list_of_valloaders = []
#create list of datasets per class for both training and validation
for stream in range(1, streams+1):
            num_of_curr_classes = 0
            newpath = os.path.join(main,str(stream))
            valpath = os.path.join(val,str(stream))
            for _, dirnames, filenames in os.walk(newpath):
                num_of_curr_classes += len(dirnames)

            for clas in range(1,num_of_curr_classes+1):

                tempset = curric_data(main, stream, clas)
                temp = torch.utils.data.DataLoader(tempset, batch_size=batch, shuffle=True)
                list_of_dataloaders.append(temp)

                tempval = curric_data(val, stream, clas)
                tempv = torch.utils.data.DataLoader(tempval, batch_size=batch, shuffle=True)
                list_of_valloaders.append(tempv)


#num_of_curr_classes = 5
def validate_model(model, valloader, criterion):

    model.eval()
    
    with torch.no_grad():
         running_loss = 0
         total = 0
         correct = 0
         for i, val in enumerate(valloader):
             (image,label) = val
             
             if torch.cuda.is_available():
                image = image.to(device)
                label = label.to(device)

             out, _, _,_ = model(image)
             loss = criterion(out,label)
             running_loss += loss.item()
             _, predicted = torch.max(out,1)
             correct += (predicted == label).sum().item() 
       
         n = len(valloader.dataset)
         print('Validation loss {}, Validation accuracy {}%'.format(running_loss/n, 100*correct/n))
         return 100*correct/n


def validate(model, list_of_valloaders, criterion, stream, mod):

    n = 0
    m = 0
    total = 0
    correct = 0
    running_loss = 0
    val_correct = 0
    model.eval()
    for idx in range(1, stream+1):
        newpath = os.path.join(main, str(idx))
        
        num_of_curr_classes = 0
        for _,dirnames, filenames in os .walk(newpath):
            num_of_curr_classes += len(dirnames)
        #model.eval()
        m = (idx-1)*inc_step


        with torch.no_grad():
            
             #total = 0
             #correct = 0
             for clas in range(1, num_of_curr_classes+1):
                 for indx, data in enumerate(list_of_valloaders[m+clas-1]):
                     image, label = data
                     
                     if torch.cuda.is_available():
                        image = image.to(device)
                        label = label.to(device)
                        #model = model.to(device)
                     _, _,out,_ = model(image, idx)
                     loss = criterion(out, label)
                     running_loss += loss.item()
                     _, predicted = torch.max(out,1)
                     correct += (predicted==label).sum().item()
                     del(image)
                     del(label)
                 n += len(list_of_valloaders[m+clas-1].dataset)
        val_correct += correct
        total += n
    print("Validation loss :{}, Validation accuracy:{}%".format(running_loss/total, 100*val_correct/total))
    return(100*val_correct/total)
#def loss_func():




def get_features(list_of_dataloaders, stream, criterion):
    premodel = VGG_16(3,args.streams,args.incstep).to(device)
    premodel.load_state_dict(torch.load(old))
    premodel.eval()
    num_of_curr_classes = 0
    newpath = os.path.join(main, str(stream))
    
    for _,dirnames,filename in os.walk(newpath):
        num_of_curr_classes += len(dirnames)
    

    mod = (stream -1) * inc_step


    out = np.zeros((560,dim,inc_step))#num_of_curr_classes))
    mat = np.array([], dtype=np.float32).reshape(0,dim)
    print('Extracting features from existing model') 
    for clas in range(1, num_of_curr_classes+1):
        idx = mod+clas-1
        #print(idx)
        mat = np.array([], dtype=np.float32).reshape(0,dim)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(list_of_dataloaders[idx]):
                 
                image, label = batch_data

                if torch.cuda.is_available():
                   
                   image = image.to(device)
                   label = label.to(device)
                
                _, rep,_,_ = premodel(image,stream)
                #dctny[path] = dist 
                del(image)
                del(label)   
                rep = rep.cpu()            
                mat = torch.from_numpy(mat).to('cpu')
                mat = torch.cat((mat,rep),0)
                mat = mat.cpu().numpy()

        out[:,:,clas-1] = mat
    del(premodel)
    return out,num_of_curr_classes,idx

def store_mem(stream,check=False):
   if(check == False):
    newpath = os.path.join(main, str(stream))
    num_of_curr_classes = 0
    dest = os.path.join(shuff, str(stream)) 
    if os.path.exists(dest) and os.path.isdir(dest):
        shutil.rmtree(dest)

    
    for _, dirnames, filenames in os.walk(newpath):
        #print('filenames',filenames)
        num_of_curr_classes += len(dirnames)
    
    mod = (stream-1)*inc_step

    for clas in range(1, num_of_curr_classes+1):

        claspath = os.path.join(newpath, str(clas))
        files = os.listdir(claspath)
        #print('files',files)
        filenames = random.sample(files,int(sel*len(files)))
        dest = os.path.join(mem, str(stream))
        #dest1 = os.path.join(mem, str(mod+clas))
        #if not os.path.exists(dest1):
        #       os.makedirs(dest1)
        dest2 = os.path.join(dest,str(mod+clas))#(mem, str(mod+clas))
        if not os.path.exists(dest2):
               os.makedirs(dest2)
        #for the_file in os.listdir(dest2):
        #    file_path = os.path.join(dest2, the_file)
        #    try:
        #        if os.path.isfile(file_path):
         #           os.unlink(file_path)
         #   except Exception as e:
         #        print(e)
        #shutil.rmtree(dest)
        for fname in filenames:
            src = os.path.join(claspath,fname) 
            shutil.copy2(src,dest2)
            #shutil.copy2(src,dest1)
    #data = DistFolder(root = dest, stream= stream,device=device,prev=old, transform = transforms.Compose([
                                                                                #transforms.Resize(224), 
                                                                                #transforms.ToTensor()
                                                                                #]))
    #data = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True)
    #order.append(data)
   
   else:
    newpath = os.path.join(main, str(stream))
    num_of_curr_classes = 0

    
    for _, dirnames, filenames in os.walk(newpath):
        #print('filenames',filenames)
        num_of_curr_classes += len(dirnames)
    
    mod = (stream-1)*inc_step

    for clas in range(1, num_of_curr_classes+1):

        claspath = os.path.join(newpath, str(clas))
        files = os.listdir(claspath)
        #print('files',files)
        filenames = random.sample(files,int(sel*len(files)))
        dest = os.path.join(mem, str(stream))
        dest = os.path.join(dest,str(clas))#(mem, str(clas))
        if not os.path.exists(dest):
               os.makedirs(dest)
        for the_file in os.listdir(dest):
            file_path = os.path.join(dest, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                 print(e)
        #shutil.rmtree(dest)
        for fname in filenames:
            src = os.path.join(claspath,fname) 
            shutil.copy2(src,dest)
       
   #return order


  
def finetune(model, stream, criterion,optimizer):

         model.train()
         finedata = ImageFolder1(root = mem,stream=stream,device=device,prev=old,transform= transforms.Compose([
          transforms.Resize(224),
          transforms.ToTensor()
    
          ]))
         fineloader = torch.utils.data.DataLoader(finedata, batch_size = 10, shuffle= True)
         for g in optimizer.param_groups:
             g['lr']=lr/10
             g['betas']=(0.9,0.99)#optimizer = torch.optim.Adam([{"params": model.parameters()},{"params": model.fc_opt[4].parameters(),"lr":0.001}], lr=(lr/10), betas=(0.9,0.999), eps=1e-08,weight_decay=wd,amsgrad=False)
         corr = 0
         #running_dist_loss = 0
         lngth = 0
         ft = True
         print("Fine Tuning")
         for e in range(30):
             #running_dist_loss = 0


             for i, d in enumerate(fineloader):
               #if(i >3):
               #  break;
               img, true, lbl_dist  = d
               print(lbl_dist.shape)
               optimizer.zero_grad()
               if torch.cuda.is_available():
                    img = img.to(device)
                    lbl_dist = lbl_dist.to(device)
                    true = true.to(device)

               out,_,_,out_dist = model(img,stream)
               #print(out_dist.shape)
               #del(img)
               #print(true)
               #lbl_dist = lbl_dist.squeeze(0)
               #out_dist = out_dist.unsqueeze(1)
               #print(lbl_dist.shape)
               #criterion = nn.CrossEntropyLoss()
               #loss_fn_kd = loss_fn_kd.to(device)
               loss_fine = loss_fn_kd(out, out_dist, true, lbl_dist,stream,ft)#criterion(torch.pow(out_dist,0.5),torch.pow(lbl, 0.5))
               loss_fine.backward()
               #torch.nn.utils.clip_grad_norm_(model.parameters(),1)
               optimizer.step()
         optimizer.zero_grad()
         del(img)
         del(lbl_dist)
         del(true)
         del(optimizer)
         torch.cuda.empty_cache()


def gen_curriculum(sim):
    
    curriculum = np.argmax(sim,axis=1)#np.argsort(-1*np.amax(sim,axis=1))#np.argmax(sim,axis=1)
    #c = [0,1,0,3]
    #c = [0,2,3,1,2,1]
    #idx = [np.argwhere(i[0]==c)for i in np.array(np.unique(c, return_counts=True)).T if i[1]>=0]
    #print(4+idx[0])
    #print(idx)
    #print(4+np.argsort(c))
    #print(idx[0][0])
    #print(idx[2][0])
    return curriculum


def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)



def  prep_data(curric,stream):
    
    mod = (stream-1)*inc_step 
    order = [] 
    curr = [] 

    src = os.path.join(mem,str(stream-1))
    dest = os.path.join(shuff, str(stream-1))
    if os.path.exists(dest) and os.path.isdir(dest):
        shutil.rmtree(dest)
    #shutil.copytree(src,dest)#.copy2(src,dest)
    copyDirectory(src,dest)

    for strm in range(1,stream):
        num = 0
        m = (strm-1)*inc_step
        path = os.path.join(shuff,str(strm))
        if(strm==1):
           data = DistFolder(root=path, stream=(stream),device=device,prev=old, transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]))
           data = torch.utils.data.DataLoader(data,batch_size=batch,shuffle=True)
        else:  

           for _, dirnames, filenames in os.walk(newpath):
               #print('filenames',filenames)
               num += len(dirnames)
           if(num != 0):
              data = ImageFolder1(root=path, stream=(stream),device=device,prev=old, transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]))
              data = torch.utils.data.DataLoader(data,batch_size=batch,shuffle=True)
           else:
              tempset = inc_data(path,device, old, stream, clas)
              data = torch.utils.data.DataLoader(tempset, batch_size=batch, shuffle=True)
              
        order.append(data)
        curr.append(m+(np.arange(0,inc_step,1)))
        store_mem(strm,check=True)
    """
    src = os.path.join(mem,str(stream-1))
    dest = os.path.join(shuff, str(stream-1))
    if os.path.exists(dest) and os.path.isdir(dest):
       shutil.rmtree(dest)
    #shutil.copytree(src,dest)#.copy2(src,dest)
    copyDirectory(src,dest)
    """ 
    #order = []
    count = 0
    arg = np.argsort(curric)
    idx = [np.argwhere(i[0]==curric)for i in np.array(np.unique(curric,return_counts=True)).T if i[1]>=0]
    while(count < len(idx)):
        if(len(idx[count]) > 1):
          for i in range(1,len(idx[count])):
              arr = np.delete(arg,np.argwhere(arg ==idx[count][i] ))
              arg = arr
          data_loc, stream, clas,dctnry = shuffle(idx[count], stream, shuffle=True)
          #tempset = curric_data_shuff(data_loc, stream, clas, dctnry)
          #data = torch.utils.data.DataLoader(tempset, batch_size=50, shuffle=True)
          data = ImageFolder1(root = data_loc, stream= stream,device=device,prev=old, transform = transforms.Compose([
                                                                                transforms.Resize(224), 
                                                                                transforms.ToTensor()
                                                                                ]))
          data = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True)
         
          order.append(data)
          count += 1
          #curr.append((mod+arr))
        else:
          arr = idx[count]
          data_loc, stream, clas,_ = shuffle(idx[count],stream, shuffle=False)
          tempset = inc_data(data_loc,device, old, stream, clas)
          data = torch.utils.data.DataLoader(tempset, batch_size=batch, shuffle=True)
          order.append(data)
          count += 1
    curr.append((mod+arr))
    return order,curr
        
          
def shuffle(arr, stream, shuffle):
    
    mod = (stream-1)*inc_step
    print('arr', arr)
    arr = 1+arr.flatten()
    if(shuffle == True):
      dctnry = {}
      dest_idx = arr[0]
      for i in range(len(arr)):
          tag = mod+arr[i]
          tag2 = mod+dest_idx
          dctnry[str(i)]= str(tag)
          path = os.path.join(main,str(stream))
          path = os.path.join(path, str(arr[i]))
          files = os.listdir(path)
          dest1 = os.path.join(shuff,str(stream))
          dest1 = os.path.join(dest1, str(tag2))
          dest = os.path.join(dest1,str(tag))
          if not os.path.exists(dest):
                 os.makedirs(dest)
          for fname in files:
              src = os.path.join(path,fname)
              shutil.copy2(src,dest)
          filenames = random.sample(files,int(sel*len(files)))
          mem1 = os.path.join(mem,str(stream))
          mem1 = os.path.join(mem1, str(tag2))
          memry = os.path.join(mem1,str(tag))
          if not os.path.exists(memry):
                 os.makedirs(memry)
         
          for fname in filenames:
              src = os.path.join(path,fname)
              shutil.copy2(src,memry)
      
      return dest1, stream, dest_idx, dctnry

    else:
      dest_idx = arr[0]
     
      dctnry = {}   
      path = os.path.join(main,str(stream))
      path = os.path.join(path, str(dest_idx))
      files = os.listdir(path)
      dest1 = os.path.join(shuff,str(stream))
      dest = os.path.join(dest1, str(dest_idx))
      if not os.path.exists(dest):
             os.makedirs(dest)
      for fname in files:
          src = os.path.join(path,fname)
          shutil.copy2(src,dest)
      filenames = random.sample(files,int(sel*len(files)))
      mem1 = os.path.join(mem,str(stream))
      #mem1 = os.path.join(mem1, str(tag2))
      memry = os.path.join(mem1,str(dest_idx))
      if not os.path.exists(memry):
             os.makedirs(memry)

      for fname in filenames:
          src = os.path.join(path,fname)
          shutil.copy2(src,memry)
 
      return shuff, stream, dest_idx, dctnry





def save_checkpoint(state, is_best, filename):
    
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best)


def save_previous(state, is_best, save_prev, filename):
    
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best)
    if (save_prev and is_best):
        shutil.copyfile(filename,old)





def loss_fn_kd(outputs, dist, labels, logits, stream, ft):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = 2
    if(ft == False):
       #alpha =  ((stream-1)*(inc_step*sel*490))
       alpha = 1#0.95 #alpha/(alpha+(inc_step*490))

       KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(dist/T, dim=1),F.log_softmax(logits/T, dim=1))  + F.cross_entropy(outputs,labels) #*(1.- alpha)


    else:
     
        KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(dist/T, dim=1),F.log_softmax(logits/T, dim=1))  + F.cross_entropy(outputs,labels) 

    return KD_loss



def stream_train(model, max_epoch, list_of_dataloaders,strm, criterion, mod, is_check, start_epoch,optimizer):
   num_of_curr_classes = 0
   n = 0
   best_acc = 0
   acc = 0
   prev_acc = 0

   #for g in optimizer.param_groups:
    #   g['lr']= 1e-06 
    #   g['betas']=(0.9,0.99)



   scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
   model.train()
   if(is_check):
     start = start_epoch
   else:
     start = (strm-1)*max_epoch
 
   for epoch in range(start,max_epoch):#for stream in range(1,strm+1):
    time_start = time.time()
    running_loss = 0
    total = 0
    correct = 0
    n = 0
    num_of_curr_classes = 0
    for stream in range(1,strm+1):
        
        mod = (stream-1)*inc_step
        
        newpath = os.path.join(main,str(stream))
        
        num_of_curr_classes = 0
        for _,dirnames,filename in os.walk(newpath):
            num_of_curr_classes += len(dirnames)
           
        print("Epoch {}/{} stream {}/{} running".format(epoch+1, strm*max_epoch,stream,strm))
        for clas in range(1,num_of_curr_classes+1):
            print('Training class {}'.format(clas))
            for batch_idx, batch_data in enumerate(list_of_dataloaders[mod+clas-1]):
                #print(len(list_of_dataloaders[mod+clas].dataset))                
                t1 =  time.time()
                
                image, label = batch_data
 
                optimizer.zero_grad()
                if torch.cuda.is_available():
                   
                   image = image.to(device)
                   label = label.to(device)
                   model = model.to(device)
 
                out,_,_,dist = model(image,stream)
                out.add(1e-8)
                loss = criterion(out,label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
                #torch.cuda.empty_cache()
                running_loss += loss.item()
                _ , predicted = torch.max(out,1)


                correct += (predicted == label).sum().item()
                t = time.time()-t1
                print("{}/{} iteration , loss: {} Elapsed time: {}".format(batch_idx+1, len(list_of_dataloaders[mod+clas-1])/batch, loss.item(),t))
            n += len(list_of_dataloaders[mod+clas-1].dataset)  
        print(correct)
        print(n) 
     
    print('{}/{} Epoch, Loss:{} Elapsed Time:{} Training Accuracy:{} %'.format(epoch+1, strm*max_epoch, running_loss/n, time.time()-time_start, 100*correct/n))
    acc = validate(model,list_of_valloaders,criterion,stream,mod)
    scheduler.step()
    #if(acc>97 and (acc-prev_acc)<= 1e-02):
    #   return acc,epsilon,is_best,mod
    prev_acc = acc
    is_best = acc> best_acc
    best_acc = max(acc, best_acc)
    epsilon = abs(acc-best_acc)
    tensorboard.log_scalar('Loss'+str(strm),running_loss/n, epoch+1)
    tensorboard.log_scalar('Training Accuracy'+str(strm), 100*correct/n, epoch+1)
    tensorboard.log_scalar('Validation Accuracy'+str(strm),acc, epoch+1)
    if(stream == strm):
      s = strm+1
    else:
      s = strm
    if(epoch == max_epoch-1):
     torch.save(model.state_dict(),old)
     old_feat_mat,_,_ = get_features( list_of_dataloaders, strm, criterion)
    else:
     old_feat_mat = []
    save_checkpoint({
                    'stream': s,
                    'epoch' : epoch,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss' : loss,
                    'acc' : acc,
                    #'mod' : mod,
                    'old_feat_mat': old_feat_mat,
                    'best_acc' : best_acc
           },is_best,PATH)
    save_prev = True
    save_previous(model.state_dict(),is_best,save_prev,PATH)
   tensorboard.log_scalar('Accuracy',best_acc,strm) #change to best_acc
   return (acc, epsilon, is_best, mod)





def stream_curric_train(model, max_epoch, strm, criterion, clas_order, curric,  old_feat_mat, is_check, start_epoch, optimizer):

   model.train()
   mod = 0 
   for g in optimizer.param_groups:
       g['lr']= lr 
       g['betas']=(0.9,0.99)
        
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)
   if(is_check):
     start = start_epoch
   else:
     start = (strm-1)*max_epoch#(2*(strm-1)-1)*max_epoch
   """
   distdata = DistFolder(root = mem, stream=strm, device=device, prev=old,transform= transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
    
    ]))
   distloader = torch.utils.data.DataLoader(distdata, batch_size = batch, shuffle= True)
   """
   #max_epoch = (10*(strm-1))+max_epoch
   for epoch in range(max_epoch):
    prev_acc = 0
    best_acc = 0
    time_start = time.time()
    running_loss = 0
    total = 0
    correct = 0
    n = 0
    ft = False
    for stream in range(strm,strm+1):
        newpath = os.path.join(main,str(stream))
        num_of_curr_classes= 0
        #print(clas_order)
        clas_order = np.asarray(clas_order)
        curriculum = clas_order.flatten()#clas_order[stream-1]#.astype(int)
        curriculum = curriculum.tolist() 
        for _,dirnames,filename in os.walk(newpath):
           num_of_curr_classes += len(dirnames)
        
        mod = (stream-1)*inc_step

        #curriculum = mod+curriculum
        curric = np.asarray(curric)
        curric = curric.flatten()
        curric = curric.tolist()
        print("Stream {}/{} Epoch {}/{}  running".format(stream,strm,epoch+1, strm*max_epoch))
        print('c',curric)
        #print('l',len(curriculum))
        #print('curr',curriculum)
        for clas,order in enumerate(curric):
            #print('Training class {}'.format(mod+order+1))
            #if (clas+1 > num_of_curr_classes):
            #   break;
            print('Training class {}'.format(order+1))
            for batch_idx, batch_data in enumerate(curriculum[clas]):
                
                t1 =  time.time()
                
                image, label, lbl_dist = batch_data
                #print('l1',label)
                optimizer.zero_grad()
                if torch.cuda.is_available():
                   lbl_dist = lbl_dist.to(device)
                   image = image.to(device)
                   label = label.to(device)
                #print(label)
                out,_,_,dist= model(image,stream)
                #out.add(1e-8)
                #print('out{}, size{}'.format(out, np.shape(out)))
                lbl_dist = lbl_dist.squeeze(0)
                #loss_fn_kd = loss_fn_kd.to(device)
                loss = loss_fn_kd(out, dist, label, lbl_dist, stream, ft)#criterion(out,label)#edit the loss to cross distillation
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
                #torch.cuda.empty_cache()
                running_loss += loss.item()
                _ , predicted = torch.max(out,1)
                #print('grad',loss.grad)
                #print('data',loss.data)

                correct += (predicted == label).sum().item()
                t = time.time()-t1
                #print('b', batch_idx)
                print("{}/{} iteration , loss: {} Elapsed time: {}".format(batch_idx+1,len(curriculum[clas].dataset)/batch,loss.item(),t))
            n += len(curriculum[clas].dataset)  
        print(correct)
        print(n)
    """ 
    if((epoch+1) != 0):#((epoch+1)%2 == 0):
                  
         corr = 0
         #running_dist_loss = 0
         lngth = 0
         print("Distillation")
         for e in range(1):
             running_dist_loss = 0


             for i, d in enumerate(distloader):
               #if(i >3):
               #  break;
               img, true , lbl_dist = d
               #print(lbl)
               optimizer.zero_grad()
               if torch.cuda.is_available():
                    img = img.to(device)
                    lbl_dist = lbl_dist.to(device)
                    true = true.to(device)

               out,_,_,out_dist = model(img,stream)
               lbl_dist = lbl_dist.squeeze(0)
               #lbl = lbl.squeeze(0)
               #print('3',np.shape(lbl))
               loss_dist = loss_fn_kd(out, out_dist, true, lbl_dist)#criterion(torch.pow(out_dist,0.5),torch.pow(lbl, 0.5))
               loss_dist.backward()
               #torch.nn.utils.clip_grad_norm_(model.parameters(),1)
               optimizer.step()
               #torch.cuda.empty_cache()
               running_dist_loss += loss_dist.item()
               _, pred = torch.max(out_dist,1)
               #lbl = lbl.type(torch.LongTensor)
               #lbl = lbl.to(device)
               #corr += (pred==lbl).sum().item() 
                
             #lngth = len(distloader.dataset)
             lngth += len(distloader.dataset)
             print("Epoch {}/{} Distillation loss {}".format(epoch+1,max_epoch,running_dist_loss/lngth))#,100*corr/lngth) )
             #tensorboard.log_scalar('Distillation loss', running_dist_loss/lngth, epoch+1)
    """ 
    print('{}/{} Epoch, Loss:{} Elapsed Time:{} Training Accuracy:{} %'.format(epoch+1,(strm)*max_epoch, running_loss/n, time.time()-time_start, 100*correct/n))
    train_acc = 100*correct/n
    acc = validate(model,list_of_valloaders,criterion,stream,mod)
    scheduler.step()
    #if(acc>90 and (acc-prev_acc)<1e-01):
    # return acc,epsilon,is_best,mod
    prev_acc = acc
    is_best = acc> best_acc
    best_acc = max(acc, best_acc)
    epsilon = abs(acc-best_acc)
    tensorboard.log_scalar('Loss'+str(strm),running_loss/n, epoch+1)
    tensorboard.log_scalar('Training Accuracy'+str(strm), 100*correct/n, epoch+1)
    tensorboard.log_scalar('Validation Accuracy'+str(strm),acc, epoch+1)
    save_checkpoint({
                    'stream': strm,
                    'epoch' : epoch,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss' : loss,
                    'acc'  : acc,
                    #'mod'  : mod,
                    'old_feat_mat': old_feat_mat,
                    'best_acc' : best_acc
            },is_best,PATH)
    mod = mod+clas        
    save_prev = True
    save_previous(model.state_dict(),is_best,save_prev,PATH)
   tensorboard.log_scalar('Accuracy',best_acc,strm)
   return (acc, epsilon, is_best, mod)




def train(model, criterion, max_epoch, save_path,optimizer):

    start_epoch = 0
    order = [] #args.parser.parse_args()
    mod = 0
    best_acc = 0
    is_check = False
    stream = 1
    #optimizer = torch.optim.Adam(model.parameters())   
    new_out_list = []
    old_out_list = []
    clas_order = []
    dctnry = {}
    curric = []
    if args.resume:
      if os.path.isfile(args.resume):
            
            print("=> loading checkpoint '{}'".format(args.resume))
            is_check = True
            checkpoint = torch.load(args.resume)
            stream = checkpoint['stream']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_acc = checkpoint['best_acc']
            acc = checkpoint['acc']
            #mod = checkpoint['mod']
            old_feat_mat = checkpoint['old_feat_mat']
            print("=> loaded checkpoint '{}' (epoch'{}')".format(args.resume, checkpoint['epoch']))

      else:

            print("=> no checkpoint found '{}'".format(args.resume))


    while (stream != streams+1):
          num_of_curr_classes = 0
          print("Incoming Stream {} of data".format(stream))
          if(stream == 1):
            #model = VGG_16(3).to(device)
            #optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(0.9,0.999),eps=1e-08, weight_decay=wd,amsgrad=False) 
            acc, epsilon, is_best, mod = stream_train(model, max_epoch, list_of_dataloaders, stream, criterion, mod, is_check, start_epoch,optimizer)
            optimizer.zero_grad()
            #del(optimizer)
            torch.cuda.empty_cache()
            #store_mem(stream)
            #dctnry = get_logits(model,stream)
            #torch.save(model.state_dict(),old)
            old_feat_mat,_,mod = get_features(list_of_dataloaders, stream, criterion)
            #order = init_order(model,list_of_dataloaders,stream,criterion)
            store_mem(stream)
            #clas_order.append(order)#clas_order.append(np.arange(0,inc_step,1))#(order) #use order where classes which have most accuracy comes first and so on..
            
            curric.append(np.arange(0,inc_step,1))
            stream += 1
           
          elif (stream != 1):
            
            prevpath = os.path.join(main, str(stream-1))
            
            new_feat_mat, curr,_ = get_features(list_of_dataloaders, stream, criterion)
            #prev = inc_step
            
            BC_matrix = np.zeros((inc_step,inc_step))#((curr, prev))
            
            old_feat_mat = np.array(old_feat_mat).reshape(560,128,inc_step)#prev)
            #print(np.shape(old_feat_mat))
            for i in range(inc_step):
                for j in range(inc_step):
                      BC_matrix[i,j] = cos(new_feat_mat[:,:,i], old_feat_mat[:,:,j])
            sim = BC_matrix
            
            curriculum = gen_curriculum(sim)
            order, curric = prep_data(curriculum,stream)
            #clas_order.append(order)#clas_order[stream-1, :] = order #modify to clas_order.append(order) ?
            #curric.append(arr)
            #print('arr',arr)
            #print('curric', curric)
            acc, _, is_best, mod = stream_curric_train(model, max_epoch, stream, criterion, order, curric, old_feat_mat, is_check, start_epoch, optimizer)#remove list_of_dataloaders and use clas_order instead ?
            #optimizer.zero_grad()
            #del(optimizer)
            torch.cuda.empty_cache()
            #fine_tune(stream) #later development
            #store_mem(stream)
            #dctnry = get_logits(model, stream)
            #finetune(model,stream,criterion,optimizer) 
            #torch.save(model.state_dict(),old)
            finetune(model,stream,criterion,optimizer) 

            old_feat_mat = new_feat_mat.copy()
            new_feat_mat.fill(0)
            stream += 1 


    print('Training Completed')
    print('Curriculum: {}'.format(curric))
    print('Validating over whole data')
    val_acc =  validate_model(model, valloader, criterion)
    print('Saving model')    
    torch.save(model.state_dict(), save_path)
    tensorboard.close()
    



if __name__ == '__main__':
   
    max_epoch = int(input('Enter maximum epoch'))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    model = VGG_16(3,args.streams,args.incstep).to(device)

    #optimizer = torch.optim.Adam([{"params":model.parameters()}],lr=1e-04, betas=(0.9,0.999),eps=1e-08, weight_decay=1e-04,amsgrad=False)    
    #optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=wd)#
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-06, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-04, amsgrad=False)

    #model = VGG_16(3).to(device)
    train(model, criterion, max_epoch, save_path,optimizer)
