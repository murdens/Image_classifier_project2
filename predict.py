# Created by : Sonya Murden
# Date : 31/08/2020
# Updated : 01/09/2020

''' Main predict : calls methods from data_utils, model_utils and command line arguments using argument_parser
    predicts image classification using a pretrained NN - default is flowers class
    user input: 
    image_path (helper: ./flowers/test/10/07090.jpg)
    checkpoint (helper: saved pretrainedmodel, default = checkpoint.pth)
    user input optional: 
    --topk (helper: top n probablities, default = 3)
    --category_names (helper: json file containing names by index, default = cat_to_name.json)
    --gpu (default = gpu)
    output: predicted class & probability.
'''

import torch
from torch import nn
from torch import optim

from model_utils import load_checkpoint, predict
from data_utils import load_data, label_mapping, process_image
from argument_parser import get_args_predict

    
args = get_args_predict()

if (args.device =='gpu' and torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

loaded_model = load_checkpoint(args.checkpoint, device)
    
probabilities, classes = predict(args.input, loaded_model, device, args.topk)
    
idx_to_name = label_mapping(args.json_file)
labels = [idx_to_name[str(i)] for i in classes]
    
# Print out result
i = 0
while i < args.topk:
    print(f"Image is classified as a {labels[i]} with a probability of {round(probabilities[i]*100,2)}%")
    i += 1