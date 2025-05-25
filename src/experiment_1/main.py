import logging
import Datasets
import torch
import torch.optim as optim
from Model.NonUniformEncoder import NonUniformEncoder
from Model.Decoder import Decoder
from Model.TrainingVAE import TrainingVAE
from Loss import VAE_loss
from Train import train_loop
training_data, testing_data = Datasets.MNIST()
shared_settings = {'embedding_dimensionality':10, 'data_shape':torch.Size([1, 28, 28]), 'layer_dimensionality':128}
weights_filename = 'experiment_1_basic_vae.pt'
training_graph_filename = 'experiment_1_training_validations.pd'
reconstruction_filename = 'experiment_1_reconstruction.png'
generation_filename = 'experiment_1_generation.png'
def main():
    logging.basicConfig(filename=".log", filemode='w', level=logging.INFO)
    shared_settings = {'embedding_dimensionality':4, 'data_shape':torch.Size([1,28,28])}
    non_uniform_encoder_settings = {'layer_dimensionality':256} | shared_settings
    decoder_settings = {'layer_dimensionality':256} | shared_settings
    #model = Training_AE(Encoder(**encoder_settings), Decoder(**decoder_settings))
    model = TrainingVAE(NonUniformEncoder(**non_uniform_encoder_settings), Decoder(**decoder_settings))
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)
    #optimizer = optim.SGD(model.parameters(), lr=2e-4)
    training_loss, testing_loss = train_loop(model, VAE_loss, training_data, optimizer, batch_size=256, epochs=32, training_epoch_granularity=3, testing_epoch_dots=10, testing_dataset=testing_data) # 6 is about enought but 2 for testing
if __name__=="__main__":
    main()
