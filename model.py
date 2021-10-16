import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torchvision import models
from torch.autograd import Variable
import pretrainedmodels

#from pyramidpooling import SpatialPyramidPooling, TemporalPyramidPooling

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    ###
###

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    ###
###


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        
        self.add_block = add_block
        self.classifier = classifier
    ###

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x
        ###
    ###
###################################################################################################


class ft_resnet18(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_resnet18, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        self.model = model_ft
        self.model.fc = nn.Sequential() #Linear(512, class_num, num_bottleneck=128)
        self.classifier = ClassBlock(512, class_num, droprate, num_bottleneck=1024, relu=True)
    ###
    
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
    ###
###################################################################################################
 

class ft_resnet18half(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_resnet18half, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(model_ft.children())[:-3])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ClassBlock(256, class_num, droprate, num_bottleneck=1024, relu=True)
    ###
    
    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    ###
###################################################################################################


class ft_resnet18prune(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_resnet18half, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, class_num)
    ###
    
    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        return x
    ###
###################################################################################################



class ft_squeezenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_squeezenet, self).__init__()
        model_ft = models.squeezenet1_0(pretrained=True)
        self.model = model_ft
        #self.model.fc = nn.Linear(512, class_num)
        self.model.classifier[1] = nn.Identity()
        self.classifier = ClassBlock(512, class_num, droprate, num_bottleneck=1024, relu=True)
    ###
    
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
    ###
###################################################################################################


class ft_mobilenetv2(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_mobilenetv2, self).__init__()
        model_ft = models.mobilenet_v2(pretrained=True)
        self.model = model_ft
        self.model.classifier[1] = nn.Identity()
        self.classifier = ClassBlock(1280, class_num, droprate, num_bottleneck=1024, relu=True)
    ###
    
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
    ###
###################################################################################################

class ft_ensemble_resnet18_resnet18half(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        self.model1 = ft_resnet18(class_num, droprate, stride)
        self.model2 = ft_resnet18half(class_num, droprate, stride)
        self.softmax = nn.LogSoftmax(dim=1)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        #x1 = self.model1(x)
        #x2 = self.model2(x)
        x2 = self.softmax(self.model2(x))
        x = torch.div(torch.add(x1,x2), 2.0)
        return x
    ###
###################################################################################################


class ft_ensemble_resnet18_squeezenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_ensemble_resnet18_squeezenet, self).__init__()
        self.model1 = ft_resnet18(class_num, droprate, stride)
        self.model2 = ft_squeezenet(class_num, droprate, stride)
        self.softmax = nn.LogSoftmax(dim=1)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        x2 = self.softmax(self.model2(x))
        x = torch.div(torch.add(x1,x2), 2.0)
        return x
    ###
###################################################################################################

class ft_ensemble_resnet18_mobilenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        self.model1 = ft_resnet18(class_num, droprate, stride)
        self.model2 = ft_mobilenetv2(class_num, droprate, stride)
        self.softmax = nn.LogSoftmax(dim=1)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        #x1 = self.model1(x)
        #x2 = self.model2(x)
        x2 = self.softmax(self.model2(x))
        x = torch.div(torch.add(x1,x2), 2.0)
        return x
    ###
###################################################################################################

class ft_ensemble_mobilenet_squeezenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        self.model1 = ft_squeezenet(class_num, droprate, stride)
        self.model2 = ft_mobilenetv2(class_num, droprate, stride)
        self.softmax = nn.LogSoftmax(dim=1)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        #x1 = self.model1(x)
        #x2 = self.model2(x)
        x2 = self.softmax(self.model2(x))
        x = torch.div(torch.add(x1,x2), 2.0)
        return x
    ###
###################################################################################################


class ft_ensemble_resnet18_mobilenet_squeezenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        self.model1 = ft_resnet18(class_num, droprate, stride)
        self.model2 = ft_squeezenet(class_num, droprate, stride)
        self.model3 = ft_mobilenetv2(class_num, droprate, stride)
        self.softmax = nn.LogSoftmax(dim=1)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        x2 = self.softmax(self.model2(x))
        x3 = self.softmax(self.model3(x))
        x = torch.div(torch.add(torch.add(x1,x2),x3), 3.0)
        return x
    ###
###################################################################################################

class ft_ensemble_resnet18_resnet18half_mobilenet_squeezenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        self.model1 = ft_resnet18(class_num, droprate, stride)
        self.model2 = ft_squeezenet(class_num, droprate, stride)
        self.model3 = ft_mobilenetv2(class_num, droprate, stride)
        self.model4 = ft_resnet18half(class_num, droprate, stride)
        self.softmax = nn.LogSoftmax(dim=1)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        x2 = self.softmax(self.model2(x))
        x3 = self.softmax(self.model3(x))
        x4 = self.softmax(self.model4(x))
        x = torch.div(torch.add(torch.add(torch.add(x1,x2),x3),x4), 4.0)
        return x
    ###
###################################################################################################



class ft_ensemblev2_resnet18_resnet18half(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        self.model1 = ft_resnet18(class_num, droprate, stride)
        self.model2 = ft_resnet18half(class_num, droprate, stride)

        self.softmax = nn.LogSoftmax(dim=1)
        self.conv2d = nn.Conv2d(2, 1, 1, bias=False)
        self.conv2d.apply(weights_init_kaiming)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        x2 = self.softmax(self.model2(x))
        
        x1 = x1.view(x1.size(0), 1, x1.size(1), 1)
        x2 = x2.view(x2.size(0), 1, x2.size(1), 1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.conv2d(x)
        x = x.view(x.size(0), x.size(2))
        return x
    ###
###################################################################################################


class ft_ensemblev2_resnet18_mobilenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        self.model1 = ft_resnet18(class_num, droprate, stride)
        self.model2 = ft_mobilenetv2(class_num, droprate, stride)

        self.softmax = nn.LogSoftmax(dim=1)
        self.conv2d = nn.Conv2d(2, 1, 1, bias=False)
        self.conv2d.apply(weights_init_kaiming)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        x2 = self.softmax(self.model2(x))

        x1 = x1.view(x1.size(0), 1, x1.size(1), 1)
        x2 = x2.view(x2.size(0), 1, x2.size(1), 1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.conv2d(x)
        x = x.view(x.size(0), x.size(2))

        return x
    ###
###################################################################################################

class ft_ensemblev2_resnet18_squeezenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        self.model1 = ft_resnet18(class_num, droprate, stride)
        self.model2 = ft_squeezenet(class_num, droprate, stride)

        self.softmax = nn.LogSoftmax(dim=1)
        self.conv2d = nn.Conv2d(2, 1, 1, bias=False)
        self.conv2d.apply(weights_init_kaiming)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        x2 = self.softmax(self.model2(x))

        x1 = x1.view(x1.size(0), 1, x1.size(1), 1)
        x2 = x2.view(x2.size(0), 1, x2.size(1), 1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.conv2d(x)
        x = x.view(x.size(0), x.size(2))

        return x
    ###
###################################################################################################

class ft_ensemblev2_mobilenet_squeezenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        self.model1 = ft_squeezenet(class_num, droprate, stride)
        self.model2 = ft_mobilenetv2(class_num, droprate, stride)

        self.softmax = nn.LogSoftmax(dim=1)
        self.conv2d = nn.Conv2d(2, 1, 1, bias=False)
        self.conv2d.apply(weights_init_kaiming)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        x2 = self.softmax(self.model2(x))

        x1 = x1.view(x1.size(0), 1, x1.size(1), 1)
        x2 = x2.view(x2.size(0), 1, x2.size(1), 1)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.conv2d(x)
        x = x.view(x.size(0), x.size(2))

        return x
    ###
###################################################################################################




class ft_ensemblev2_resnet18_mobilenet_squeezenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        
        self.model1 = ft_resnet18(class_num, droprate, stride)
        self.model2 = ft_squeezenet(class_num, droprate, stride)
        self.model3 = ft_mobilenetv2(class_num, droprate, stride)
       
        self.softmax = nn.LogSoftmax(dim=1)
        self.conv2d = nn.Conv2d(3, 1, 1, bias=False)
        self.conv2d.apply(weights_init_kaiming)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        x2 = self.softmax(self.model2(x))
        x3 = self.softmax(self.model3(x))
     
        x1 = x1.view(x1.size(0), 1, x1.size(1), 1)
        x2 = x2.view(x2.size(0), 1, x2.size(1), 1)
        x3 = x3.view(x3.size(0), 1, x3.size(1), 1)
        
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv2d(x)
        x = x.view(x.size(0), x.size(2))
   
        return x
    ###
###################################################################################################

class ft_ensemblev2_resnet18_resnet18half_mobilenet_squeezenet(nn.Module):
    ###
    
    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        
        self.model1 = ft_resnet18(class_num, droprate, stride)
        self.model2 = ft_resnet18half(class_num, droprate, stride)
        self.model3 = ft_squeezenet(class_num, droprate, stride)
        self.model4 = ft_mobilenetv2(class_num, droprate, stride)
       
        self.softmax = nn.LogSoftmax(dim=1)
        self.conv2d = nn.Conv2d(4, 1, 1, bias=False)
        self.conv2d.apply(weights_init_kaiming)
    ###
    
    def forward(self, x):
        x1 = self.softmax(self.model1(x))
        x2 = self.softmax(self.model2(x))
        x3 = self.softmax(self.model3(x))
        x4 = self.softmax(self.model4(x))
     
        x1 = x1.view(x1.size(0), 1, x1.size(1), 1)
        x2 = x2.view(x2.size(0), 1, x2.size(1), 1)
        x3 = x3.view(x3.size(0), 1, x3.size(1), 1)
        x4 = x4.view(x4.size(0), 1, x4.size(1), 1)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv2d(x)
        x = x.view(x.size(0), x.size(2))
   
        return x
    ###
###################################################################################################






