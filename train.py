# Created by : Sonya Murden
# Date : 31/08/2020
# updated : 01/09/2020

''' Main train : calls methods from data_utils, model_utils and command line arguments using argument_parser
    Trains chosen torchvision.models architecture (user can chose vgg16 or densenet121),
    user input: data_directory (helper:  ./flowers) 
    user input optional: 
    --classes (helper:output units, default=102)
    --save_dir (default = checkpoint.pth)
    --arch (helper: vgg16 or densenet121, default=vgg16)
    --learning_rate (default = 0.009)
    --hidden_units  (default = 500)
    --epochs (default = 3)
    --gpu
    
    output: The training loss, validation loss, and validation accuracy are printed out as a network trains
'''

import torch
from torch import nn
from torch import optim

from model_utils import save_checkpoint, train_model, validation, test_model, create_classifier, pretrained_model
from data_utils import load_data
from argument_parser import get_args_train

#constants 
#output_cats = 102  # number of flower classifications (can make this a command line input for other training)

args = get_args_train()

if (args.device =='gpu' and torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    print("Model should be trained on GPU, enable and select --gpu gpu for training")

train_data, test_data, validation_data, trainloader, testloader, validationloader = load_data(args.data_directory)

pretrain_model, arch_inFeatures = pretrained_model(args.arch)

model, criterion = create_classifier(pretrain_model, arch_inFeatures, args.hidden_units, args.output_cats)

optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

trained_model = train_model(model, args.epochs, trainloader, validationloader, device, optimizer, criterion)

tested_model = test_model(trained_model, testloader, device, optimizer, criterion)

save_checkpoint(trained_model, args.save_directory, args.arch, train_data, optimizer, args.epochs, args.hidden_units)