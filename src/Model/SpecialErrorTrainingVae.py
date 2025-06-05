#TODO make an approprate loss function to deal with the fundemental error
from torch import nn, randn
class SpecialErrorTrainingVAE(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
  def forward(self, input):
    encoder_mean, encoder_stdev, fundemental_error = self.encoder(input)
    encoder_draw = encoder_mean + randn(encoder_mean.size()) * encoder_stdev
    decoder_mean, decoder_stdev = self.decoder(encoder_draw)
    return {
        'encoder_mean':encoder_mean,
        'encoder_stdev':encoder_stdev,
        'fundemental_error':fundemental_error
        'encoder_draw':encoder_draw,
        'decoder_mean':decoder_mean,
        'decoder_stdev':decoder_stdev
        }
