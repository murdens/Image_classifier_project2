# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

### Building the command line application
Application uses a pair of Python scripts that run from the command line. 
It is built to enable loading of image data with a mapping file of categories to names, selection of pretrained architecture and user set hyperparameters.

### Graduating Certificate
[certificate.pdf](https://github.com/murdens/Image_classifier_project2/blob/master/certificate.pdf)

### Specifications
- Train.py, will train a new network on a dataset and save the model as a checkpoint. 
- Predict.py, uses a trained network to predict the class for an input image. 

#### Helpers:
- data_utils
- model_utils
- argument_parser

#### Train a new network on a data set with train.py

Basic usage: python train.py data_directory

Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg13"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu

#### Predict flower name from an image with predict.py along with the probability of that name.

Basic usage: python predict.py /path/to/image checkpoint

Options:
- Return top K most likely classes: python predict.py input checkpoint --top_k 3
- Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python predict.py input checkpoint --gpu
