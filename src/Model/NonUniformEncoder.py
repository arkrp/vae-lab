import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)
logger.info('Loading NonUniformEncoder')
class NonUniformEncoder(nn.Module):
  def __init__(self, *, embedding_dimensionality=10, data_shape=torch.Size([1, 28, 28]), layer_dimensionality=128, uniform_stdev=0.1):
    super().__init__()
    self.uniform_stdev = uniform_stdev
    self.embedding_dimensionality = embedding_dimensionality
    self.flatten = nn.Flatten()
    self.estack = nn.Sequential(
        nn.BatchNorm1d(data_shape.numel()),
        nn.Linear(data_shape.numel(), layer_dimensionality),
        nn.BatchNorm1d(layer_dimensionality),
        nn.ReLU(),
        nn.Linear(layer_dimensionality, layer_dimensionality),
        nn.BatchNorm1d(layer_dimensionality),
        nn.ReLU(),
        nn.Linear(layer_dimensionality, embedding_dimensionality*2)
    )
  def forward(self, input):
    network_output = self.estack(self.flatten(input))
    mean, network_stdev = torch.split(network_output, self.embedding_dimensionality, dim=1)
    stdev = self.uniform_stdev + torch.abs(network_stdev)
    return mean, stdev
logger.info('Verifying NonUniformEncoder integrity...')
non_uniform_encoder_settings = {'embedding_dimensionality':2, 'data_shape':torch.Size([5,5]), 'layer_dimensionality':16}
blank_non_uniform_encoder = NonUniformEncoder(**non_uniform_encoder_settings)
try:
  dummy_data = torch.zeros(torch.Size([2])+non_uniform_encoder_settings['data_shape'])
  dummy_mean, dummy_stdev = blank_non_uniform_encoder(dummy_data)
  expected_output_size = torch.Size([2 , non_uniform_encoder_settings['embedding_dimensionality']])
  if (dummy_mean.size() != expected_output_size):
    raise ValueError(f'Unexpected encoder mean output size {dummy_mean.size()}, expected {expected_output_size}')
  if (dummy_stdev.size() != expected_output_size):
    raise ValueError(f'Unexpected encoder stdev output size {dummy_stdev.size()}, expected {expected_output_size}')
except Exception as e:
  raise ValueError('NonUniformEncoder Inoperable. Repair needed.') from e
