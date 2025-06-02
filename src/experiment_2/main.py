#  setup!
#  imports!
import logging
import torch
import os
import pandas as pd
import Diagrams
from Datasets import MNIST
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from Model.FixedMaskingEncoder import FixedMaskingEncoder
from Model.Decoder import Decoder
from Model.TrainingVAE import TrainingVAE
from Loss.VAElosses import VAE_loss
from Train.BasicTraining import train_loop
# 
#  constants!
experiment_name = 'experiment_2'
weights_filename = 'experiment_2_masking_vae.pt'
training_losses_filename = 'experiment_2_training.csv'
training_graph_filename = 'experiment_2_training.png'
reconstruction_filename = 'experiment_2_reconstruction.png'
generation_filename = 'experiment_2_generation.png'
imputation_filename = 'experiment_2_imputation.png'
# 
#  logger!
logger = logging.getLogger(__name__)
# 
training_data, testing_data = MNIST()
# 
def train(model): #  
    """ #  
    sets the optimizers and training data up and trains the model!
    returns model tracked training loss and tracked testing loss per epoch.
    """
    # 
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)
    return train_loop(model, VAE_loss, training_data, optimizer, batch_size=256, epochs=2, training_epoch_granularity=3, testing_epoch_dots=10, testing_dataset=testing_data)#TODO up the epochs
# 
def makemodel(): #  
    """makes the empty model"""
    shared_settings = {'embedding_dimensionality':4, 'data_shape':torch.Size([1,28,28])}
    mask = torch.zeros(shared_settings['data_shape'])
    mask[:,14:,:] = 1.0
    encoder_settings = {'layer_dimensionality':256, 'mask':mask} | shared_settings
    decoder_settings = {'layer_dimensionality':256} | shared_settings
    return TrainingVAE(FixedMaskingEncoder(**encoder_settings), Decoder(**decoder_settings))
# 
def main(): #  
    #  start logging
    logging.basicConfig(filename=".log", filemode='w', level=logging.INFO)
    logger.info(f'launching {experiment_name}')
    # 
    #  set up save folder
    #  sus* out names
    experiment_data_dir = None
    save_data_dir = None
    try:
        experiment_data_dir = os.environ['EXPERIMENT_DATA_DIR'] + '/'
        save_data_dir = experiment_data_dir + experiment_name + '/'
    except KeyError as e:
        raise RuntimeError('Failed to load experiment data directory environment variable.') from e
    # 
    #  create our directory if necissary!
    if experiment_name not in os.listdir(experiment_data_dir):
        logger.info(f'directory {save_data_dir} not found.. Summoning.')
        os.mkdir(save_data_dir)
    # 
    # 
    #  initialize model
    model = makemodel()
    # 
    #  aquire model weights and training data
    training_losses_set = False
    #  load the existing file if possible
    if weights_filename in os.listdir(save_data_dir):
        print('Found existing weights file! equiping!')
        model.load_state_dict(torch.load(save_data_dir + weights_filename))
    # 
    #  or train new model weights!
    else:
        print('No weights file found. Training new model!')
        training_losses = train(model)
        training_losses_set = True
        torch.save(model.state_dict(),save_data_dir + weights_filename)
        print('New model trained')
    # 
    # 
    #  Make records of training data
    if(training_losses_set):
        print('saving model trainining data')
        training_losses.to_csv(save_data_dir + training_losses_filename)
        print(f'saving model training graph to {save_data_dir + training_graph_filename}')
        Diagrams.train_figure(training_losses).savefig(save_data_dir + training_graph_filename)
        plt.close()
    # 
    #  make reconstruction graphics
    print(f'saving model reconstructions to {save_data_dir + reconstruction_filename}')
    Diagrams.reconstruction_figure(model).savefig(save_data_dir + reconstruction_filename)
    plt.close()
    # 
    #  make generation graphics
    print(f'saving model generations to {save_data_dir + generation_filename}')
    Diagrams.generation_figure(model).savefig(save_data_dir + generation_filename)
    plt.close()
    # 
# 
if __name__=='__main__':
    main()
