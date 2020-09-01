# Created by : Sonya Murden
# Date : 31/08/2020
# Updated : 01/09/2020

''' methods:
    load_data(data_directory)
    label_mapping(json_file)
    process_image(image_file)
'''

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json


def load_data(data_directory):
    ''' Input : file path of data
        Transforms data for the training, validation, and testing sets
        Returns : directories and dataloaders
    '''
    data_dir = data_directory
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=False)

    return train_data, test_data, validation_data, trainloader, testloader, validationloader


def label_mapping(json_file):
    ''' Input: mapping file data to name
        Returns : index mapped to name file idx_to_name 
    '''
    with open(json_file, 'r') as f:
        idx_to_name = json.load(f)
        return idx_to_name
    

def process_image(image_file):
    ''' Input: Image for prediction
        Resizes, crops, normalizes and adjusts for prediction method
        Returns : adjusted image
    '''
    img = Image.open(image_file) 
    
    # resize maintaining aspect ratio
    w, h = img.size
    if w < h:
        basewidth = 256
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    else:
        baseheight = 256
        hpercent = (baseheight / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, baseheight), Image.ANTIALIAS)
    
    # crop 224 x 224
    left = (256-224)/2
    top = (256-224)/2
    right = left + 224 
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    np_image = np.array(img)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2,0,1))
    
    return np_image