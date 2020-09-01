# Created by : Sonya Murden
# Date : 01/09/2020


''' command line parser
'''

import argparse

def get_args_predict():
    ''' command line inputs for running predict.py
    '''
    parser = argparse.ArgumentParser(description='Use NN to predict image classification')
    
    parser.add_argument('input', action ='store', help='Image file')
    
    parser.add_argument('checkpoint', action = 'store', default='checkpoint.pth', help='Enter path of model checkpoint')
    
    parser.add_argument('--top_k', type=int, dest='topk', action='store', default=3, help='Enter number of classes to view in result')
    
    parser.add_argument('--category_names', dest='json_file', action='store', default='cat_to_name.json', help='Enter path to mapping file')
    
    parser.add_argument('--gpu', dest='device', action='store', default='gpu', help='Turn on gpu mode. Note: default set to ON')
    
    return parser.parse_args()


def get_args_train():
    ''' command line inputs for running train.py
    '''
    
    parser = argparse.ArgumentParser(description='Train a classifier NN')
    
    parser.add_argument('data_directory', action ='store', help='Enter path to training data')
    
    parser.add_argument('--classes', type=int, dest='output_cats', action ='store', default=102, 
                        help='Enter number of classifications for data. Note: default is 102 for flowers data')
    
    parser.add_argument('--save_dir', dest='save_directory', action = 'store', default='checkpoint.pth', 
                        help='Enter location to save checkpoint i.e. checkpoint.pth')
    
    parser.add_argument('--arch', dest='arch', action='store', default='vgg16',
                        help='Enter pretrained model to use "vgg16" or "densenet121". Note: default is vgg16')
    
    parser.add_argument('--learning_rate', type=float, dest='lr', action='store', default=0.0009, 
                        help='Enter learning rate for training. Note: default is 0.0009')
    
    parser.add_argument('--hidden_units', type=int, dest='hidden_units', action='store', default=500, 
                        help='Enter units for hidden layer output. Note: default is 500')
    
    parser.add_argument('--epochs', type=int, dest='epochs', action='store', default=3, 
                        help='Enter number of training epochs. Note: default is 3')
    
    parser.add_argument('--gpu', dest='device', action='store', default='gpu', 
                        help='Turn on gpu mode. Note: default set to ON')
    
    return parser.parse_args()
