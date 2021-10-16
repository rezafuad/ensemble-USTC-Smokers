import os
import sys
import pandas as pd
import numpy as np
import time
import copy
import math

from model import *
import torch
from torch.utils.data import Subset, DataLoader 
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler



### training opt
opt_batchsize = 32 #
opt_name = 'ft_ensemblev2_resnet18_resnet18half_mobilenet_squeezenet'
opt_netname = 'ft_ensemblev2_resnet18_resnet18half_mobilenet_squeezenet'
opt_dropout = 0 #.5
opt_stride = 2
opt_lr = 0.05 #1


dir_name = os.path.join('./model',opt_name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
###


### to bypass image loader
def loader(path):
    return 0
### create the dataset for stats
dataset = datasets.ImageFolder(root='../data/', loader=loader)
### count examples per each class
### Cloud  Dust  Haze  Land  Seaside  Smoke
classinfo = { 0: [], 1: [], 2: [], 3: [], 4: [], 5: [] }
for ind in range(len(dataset)):
    img, label = dataset[ind]
    if len(classinfo[label]) == 0:
        classinfo[label] = [1, ind, 0]
    else:
        classinfo[label][0] += 1
        if ind > classinfo[label][2]:
            classinfo[label][2] = ind
        ###
    ###
###
print("--> Class Info:", classinfo)
#sys.exit()
### Split dataset into train and val (64 % for training, 16 % for validation, and 20 % for testing)
train_indices = []
val_indices = []
test_indices = []
for k in classinfo:
    rndind = np.random.permutation(classinfo[k][0])
    p80percent = int( round( 0.64 * classinfo[k][0] + 0.2 ) )   # additional constant to match the training/val/test split as in the original paper
    p16percent = int( round( 0.16 * classinfo[k][0] + 0.275 ) ) # additional constant to match the training/val/test split as in the original paper
    p20percent = classinfo[k][0] - p80percent - p16percent
    #print(k, len(rndind), p80percent, p16percent, p20percent)
    train_indices += [b+classinfo[k][1] for b in rndind[0:p80percent]]
    val_indices += [b+classinfo[k][1] for b in rndind[p80percent:p80percent+p16percent]]
    test_indices += [b+classinfo[k][1] for b in rndind[p80percent+p16percent::]]
###
### count examples per each class (training dan validation set)
classinfotrain = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 }
classinfoval   = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 }
classinfotest  = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 }
for ind in train_indices:
    img, label = dataset[ind]
    classinfotrain[label] += 1
###
for ind in val_indices:
    img, label = dataset[ind]
    classinfoval[label] += 1
###
for ind in test_indices:
    img, label = dataset[ind]
    classinfotest[label] += 1
###

print("--> Training per class   :", classinfotrain)
print("--> Validation per class :", classinfoval)
print("--> Testing per class    :", classinfotest)


print("--> Creating subset ...")
## recreating dataset
datasettrain = datasets.ImageFolder(root='../data/', 
                            transform=transforms.Compose([transforms.Resize((256,256)),
                                                          transforms.ToTensor()]))
datasetval   = datasets.ImageFolder(root='../data/', 
                            transform=transforms.Compose([transforms.Resize((256,256)),
                                                          transforms.ToTensor()]))
datasettest   = datasets.ImageFolder(root='../data/', 
                            transform=transforms.Compose([transforms.Resize((256,256)),
                                                          transforms.ToTensor()]))
trainDataset = Subset(datasettrain, train_indices)
valDataset = Subset(datasetval, val_indices)
testDataset = Subset(datasettest, test_indices)

## save val and test indices
with open(os.path.join("./model", opt_name, "valtestindices.npy"), "wb") as f:
    np.save(f, val_indices)
    np.save(f, test_indices)
###

## calculate mean and std
#print("--> Calculate mean and std of training dataset ...")
#meanRGB = [np.mean(np.array(x), axis=(1,2)) for x,_ in trainDataset]
#stdRGB = [np.std(np.array(x), axis=(1,2)) for x,_ in trainDataset]
#mean = [np.mean([m[0] for m in meanRGB]), np.mean([m[1] for m in meanRGB]), np.mean([m[2] for m in meanRGB])]
#std  = [np.mean([m[0] for m in stdRGB]), np.mean([m[1] for m in stdRGB]), np.mean([m[2] for m in stdRGB])]
#print("       mean =", mean, "- std =", std)
mean = [0.308602, 0.29497108, 0.28711048]
std = [0.15585361, 0.15004204, 0.14360557]
#sys.exit()

## re arrange transforms 
train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomPerspective(),
                    transforms.RandomRotation(10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
val_transform = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
trainDataset.dataset.transform = train_transform
valDataset.dataset.transform = val_transform
testDataset.dataset.transform = val_transform


###
dataloaders = { 'train': DataLoader(trainDataset, batch_size=opt_batchsize, shuffle=True, num_workers=8, pin_memory=False),
                'val': DataLoader(valDataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=False) }
datasizes = { 'train': len(trainDataset), 'val': len(valDataset) }


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## loss history
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    ###
    fig.savefig( os.path.join('./model',opt_name,'train.svg'))
###


def save_network(network, epoch_label, test=False):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',opt_name,save_filename)
    if test:
        torch.save(network, save_path)
    else:
        torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available() and not test:
        network.cuda(device) #gpu_ids[0])
    ###
###

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    print("--> Starting the training process ... ")

    best_model_wts = copy.deepcopy(model.cpu().state_dict())
    model.cuda(device) #gpu_ids[0])
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('    Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('    ' + '-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            ###

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    ###
                ###

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            ###

            if phase == 'train':
                scheduler.step()
            ###

            epoch_loss = running_loss / datasizes[phase]
            epoch_acc = running_corrects.double() / datasizes[phase]

            print('    {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
                #if epoch % 2 == 1:
                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.cpu().state_dict())
                    model.cuda(device)
                ###
                save_network(model, epoch)
                ###
                draw_curve(epoch)
                #
                time_elapsed = time.time() - since
                print('    Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            ###
        ###
        print()
    ###

    time_elapsed = time.time() - since
    print('--> Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('    Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    save_network(best_model_wts, 'best', True)

    return model
###



#model = ft_resnet18(6, opt_dropout, opt_stride)
#model = ft_resnet18half(6, opt_dropout, opt_stride)
#model = ft_mobilenetv2(6, opt_dropout, opt_stride)
#model = ft_squeezenet(6, opt_dropout, opt_stride)

#model = ft_ensemble_resnet18_resnet18half(6, opt_dropout, opt_stride)
#model = ft_ensemble_resnet18_mobilenet(6, opt_dropout, opt_stride)
#model = ft_ensemble_resnet18_squeezenet(6, opt_dropout, opt_stride)
#model = ft_ensemble_mobilenet_squeezenet(6, opt_dropout, opt_stride)
#model = ft_ensemble_resnet18_mobilenet_squeezenet(6, opt_dropout, opt_stride)
#model = ft_ensemble_resnet18_resnet18half_mobilenet_squeezenet(6, opt_dropout, opt_stride)

#model = ft_ensemblev2_resnet18_resnet18half(6, opt_dropout, opt_stride)
#model = ft_ensemblev2_resnet18_mobilenet(6, opt_dropout, opt_stride)
#model = ft_ensemblev2_resnet18_squeezenet(6, opt_dropout, opt_stride)
#model = ft_ensemblev2_mobilenet_squeezenet(6, opt_dropout, opt_stride)
#model = ft_ensemblev2_resnet18_mobilenet_squeezenet(6, opt_dropout, opt_stride)
model = ft_ensemblev2_resnet18_resnet18half_mobilenet_squeezenet(6, opt_dropout, opt_stride)

print(model)


#ignored_params = list(map(id, model.classifier.parameters() ))
ignored_params = list(map(id, model.model1.classifier.parameters() ))
ignored_params += list(map(id, model.model2.classifier.parameters() ))
ignored_params += list(map(id, model.model3.classifier.parameters() ))
ignored_params += list(map(id, model.model4.classifier.parameters() ))

base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
#optimizer_ft = optim.Adam([
    {'params': base_params, 'lr': 0.1*opt_lr},
#    {'params': base_params, 'lr': opt_lr},
    #{'params': model.classifier.parameters(), 'lr': opt_lr}
    {'params': model.model1.classifier.parameters(), 'lr': opt_lr},
    {'params': model.model2.classifier.parameters(), 'lr': opt_lr},
    {'params': model.model3.classifier.parameters(), 'lr': opt_lr},
    {'params': model.model4.classifier.parameters(), 'lr': opt_lr},
#    ], weight_decay=0, betas=(0.9,0.999), eps=1e-8, amsgrad=False)
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

# model to gpu
model = model.cuda(device)
#model = torch.nn.parallel.DataParallel(model)

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()

### training process
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)

### testing process
print("--> Testing process")
model.eval()
### testing variable
#running_loss = 0.0
running_corrects = 0
stats_perclass = [ [0,0], [0,0], [0,0], [0,0], [0,0], [0,0] ]
predictions_scores = []
###
predslabels = []
### Iterate over testing data.
for inputs, labels in testDataset: #DataLoader(testDataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
    inputs = torch.unsqueeze(inputs.to(device), dim=0)
    #labels = torch.IntTensor([labels]) #.to(device)
    
    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        #loss = criterion(outputs, labels)   
    ###

    # statistics
    #running_loss += loss.item() #* inputs.size(0)
    running_corrects += (preds == labels) #torch.sum(preds == labels.data)
    stats_perclass[labels][0] += 1
    if preds == labels:
        stats_perclass[labels][1] += 1
    ###
    predslabels.append([labels, preds.item()])
###
#test_loss = running_loss / len(testDataset)
test_acc = running_corrects.double() / len(testDataset)
test_acc = float(test_acc[0]) 
###
print("--> Testing result")
#print('    -> Testing loss: {:4f}'.format(test_loss) )
print('    -> Testing accuracy: {:4f}'.format(test_acc) )
for c in range(6):
    print('    -> Class ' + dataset.classes[c] + ' (corrects/num) : {:d}/{:d}'.format(stats_perclass[c][1], stats_perclass[c][0]) )
###
print("--> Finished ...")

###
with open(os.path.join("./model", opt_name, "testresults.npy"), "wb") as f:
    np.save(f, np.array(predslabels))
###

