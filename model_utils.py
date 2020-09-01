# Created by : Sonya Murden
# Date : 31/08/2020
# Updated : 01/09/2020

''' methods:
load_checkpoint(filepath, device, arch='vgg16')
    predict(image_path, model, device, topk=5)
    pretrained_model(arch)
    create_classifier(model, arch_inFeatures, hidden_units, output_cats)
    validation(model, validationloader, criterion, device)
    train_model(model, epochs, trainloader, validationloader, device, optimizer, criterion)
    test_model(model, testloader, device, optimizer, criterion)
    save_checkpoint(model, save_dir, arch, train_data, optimizer, epochs, hidden_units)
'''

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from PIL import Image
from data_utils import process_image
import numpy as np

# for testing delete when new process works
def load_checkpoint(filepath, device, arch='vgg16'):
    ''' Input : file path of checkpoint, arch is set at default vgg16.
        Returns : last saved model checkpoint
    '''
    if device == 'cuda':
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location = 'cpu')
    
    try:
        arch = checkpoint['arch']
    except:
        arch = arch        
    
    model, arch_inFeatures = pretrained_model(arch)
        
    model.class_to_idx = (checkpoint['class_to_idx'])
    model.classifier = (checkpoint['classifier'])
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = (checkpoint['optimizer'])
    
    return model

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to(device)
    
    img = process_image(image_path)
    image_tensor = torch.from_numpy(np.expand_dims(img,axis=0)).type(torch.FloatTensor).to(device)
    model.eval()
    with torch.no_grad():
        logps = model.forward(image_tensor)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        probabilities = [p.item() for p in top_p[0]]
        classes = [idx_to_class[i.item()] for i in top_class[0]]
        
    model.train()
    
    return probabilities, classes


def pretrained_model(arch):
    ''' Input: arch for chosen torchvision.models
        Output: based model with frozen parameters to avoid backprop
    '''
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        arch_inFeatures = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        arch_inFeatures = model.classifier.in_features
    else:
        print("Script can be run with 'vgg16' or 'densenet121', please re-enter.")    
    
    for param in model.parameters():
        param.requires_grad = False

    return model, arch_inFeatures

    
def create_classifier(model, arch_inFeatures, hidden_units, output_cats):
    ''' Input: base pretrained model, in features of arch, n units for hidden layer, n class outputs
        Output: model with prescribed classifier, negative log likelihood loss (criterion)
    '''
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(arch_inFeatures, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.1)),
                          ('fc2', nn.Linear(hidden_units, output_cats)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()

    return model, criterion

def validation(model, validationloader, criterion, device):
    ''' validates model loss and accuracy
        Input: updated pretrained model, validation data, criterion, device (default=gpu)
        Output: validation loss, accuracy
    '''
    validation_loss = 0
    accuracy = 0
    
    for inputs, labels in validationloader:
        # Move input and label tensors to active devide
        inputs, labels = inputs.to(device), labels.to(device)
                        
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
                    
        validation_loss += batch_loss.item()
                    
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    return validation_loss, accuracy       


def train_model(model, epochs, trainloader, validationloader, device, optimizer, criterion):
    ''' Input: updated pretrained model, epochs, training data, validation data, optimizer (incl learning rate), criterion
        Output: Prints training loss, validation loss & validation accuracy, returns trained model
    '''
    model.to(device)
    
    for epoch in range(epochs):
        print(f"Training - epoch {epoch+1}...")
        start = time.time()
        running_loss = 0
        
        for inputs, labels in trainloader:
            # Move input and label tensors to active devide
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        else:
            # Run validation
            model.eval()
            
            with torch.no_grad():
                validation_loss, accuracy = validation(model, validationloader, criterion, device)
                
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validationloader):.3f}.. "
              f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} secs")
        
        running_loss = 0     
        
        model.train()
        
    return model

def test_model(model, testloader, device, optimizer, criterion):
    ''' Inputs: trained model, testloader, device, optimizer (incl learning rate), criterion
        Output: Prints overall model accuracy %, returns tested model
    '''
    print("Running model overall accuracy test....")
    model.eval()
    total = 0
    accuracy = 0
    model.to(device)

    with torch.no_grad():    
        for inputs, labels in testloader:
            # Move input and label tensors to active devide
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    print(f"Test accuracy: {round(100 * accuracy/len(testloader),3)}%")
    
    model.train()
    
    return model


def save_checkpoint(model, save_dir, arch, train_data, optimizer, epochs, hidden_units):
    ''' saves model checkpoint
        Inputs: trained model, arch, dir of training data, optimizer, epochs
        Output: saved checkpoint
    '''
    model.class_to_idx = train_data.class_to_idx

    checkpoint =  {'arch': arch,
                   'hidden_layer': hidden_units,
                   'class_to_idx': model.class_to_idx,
                   'classifier': model.classifier,
                   'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict,
                   'epochs': epochs}
    
    print("Model saved")

    return torch.save(checkpoint, save_dir)
