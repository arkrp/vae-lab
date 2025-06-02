import torch
import matplotlib.pyplot as plt
from Datasets import MNIST
from torch.utils.data import DataLoader
def reconstruction_figure(model): #  
    """takes in a full vae and creates a diagram showing reconstructions it makes from the MINST dataset"""
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
    return figure
# 
def generation_figure(model): #  
    """takes in a full vae and creates a diagram showing generations it makes"""
    def random_decoder_sample(decoder):
      return decoder(torch.randn(decoder.embedding_dimensionality).unsqueeze(0))[0]
    model.eval()
    rows, columns, scaleup = 2, 8, 2
    figure = plt.figure(figsize=(columns* scaleup, rows*scaleup), layout='constrained')
    figure.suptitle('Random Draws p(~N(0,I))')
    figure.set_facecolor('lightyellow')
    figure.set_edgecolor('tan')
    figure.set_linewidth(1)
    axs = figure.subplots(nrows=rows, ncols=columns)
    for ax in axs.ravel():
      image = random_decoder_sample(model.decoder).detach()
      ax.imshow(image.squeeze(), cmap='grey')
      ax.set_title('')
      ax.set_axis_off()
    return figure
# 
def train_figure(training_losses): #  
    """generates a training graph from a table of losses per epoch"""
    figure = plt.figure()
    ax = figure.subplots()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.plot(training_losses['epoch'], training_losses['training loss'], label='training loss')
    ax.plot(training_losses['epoch'], training_losses['testing loss'], label='testing loss')
    ax.legend(loc='upper right')
    return figure
# 
