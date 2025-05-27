#  setup!
#  imports!
import logging
import torch
import os
from Datasets import MNIST
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from Model.NonUniformEncoder import NonUniformEncoder
from Model.Decoder import Decoder
from Model.TrainingVAE import TrainingVAE
from Loss.VAElosses import VAE_loss
from Train.BasicTraining import train_loop
# 
#  constants!
experiment_name = 'experiment_1'
weights_filename = 'experiment_1_basic_vae.pt'
training_graph_filename = 'experiment_1_training_validations.pd'
reconstruction_filename = 'experiment_1_reconstruction.png'
generation_filename = 'experiment_1_generation.png'
# 
#  logger!
logger = logging.getLogger(__name__)
# 
# 
def train(model): #  
    """ #  
    sets the optimizers and training data up and trains the model!
    returns model tracked training loss and tracked testing loss per epoch.
    """
    # 
    training_data, testing_data = MNIST()
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)
    return train_loop(model, VAE_loss, training_data, optimizer, batch_size=256, epochs=32, training_epoch_granularity=3, testing_epoch_dots=10, testing_dataset=testing_data)
# 
def makemodel(): #  
    """makes the empty model"""
    shared_settings = {'embedding_dimensionality':4, 'data_shape':torch.Size([1,28,28])}
    non_uniform_encoder_settings = {'layer_dimensionality':256} | shared_settings
    decoder_settings = {'layer_dimensionality':256} | shared_settings
    return TrainingVAE(NonUniformEncoder(**non_uniform_encoder_settings), Decoder(**decoder_settings))
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
    #  aquire model weights
    #  load the existing file if possible
    if weights_filename in os.listdir(save_data_dir):
        print('Found existing weights file! equiping!')
        model.load_state_dict(torch.load(save_data_dir + weights_filename))
    # 
    #  or train new model weights!
    else:
        print('No weights file found. Training new model!')
        training_data = train(model)
        torch.save(model.state_dict(),save_data_dir + weights_filename)
        print('New model trained')
    # 
    # 
    #@title Display reconstructions
    _, testing_data = MNIST()
    quicktest_dataloader = iter(DataLoader(testing_data, batch_size=1))
    rows, columns, scaleup = 2, 4, 2
    model.eval()
    figure = plt.figure(figsize=(columns*2*scaleup, rows*scaleup), layout='constrained')
    figure.set_facecolor('lightyellow')
    figure.set_edgecolor('tan')
    figure.set_linewidth(1)
    figure.suptitle('Reconstruction $x_0:argmax_{x}(p(\\sim q(x_0)))$')
    subfigures = figure.subfigures(nrows=rows, ncols=columns)
    for subfigure in subfigures.ravel():
        display_image, label = next(quicktest_dataloader)
        decoded_image = model(display_image)['decoder_mean'].detach()
        axs = subfigure.subplots(nrows=1, ncols=2)
        axs[0].imshow(display_image.squeeze(), cmap='grey')
        axs[0].set_axis_off()
        axs[0].set_title(f'Original ({label.item()})')
        axs[1].imshow(decoded_image.squeeze(), cmap='grey')
        axs[1].set_axis_off()
        axs[1].set_title(f'Reconstruction')
    plt.show()
# 
if __name__=='__main__':
    main()
