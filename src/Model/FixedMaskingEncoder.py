#FixedMaskingEncoder is meant to be a proof of concept drop in replacement for NonUniformEncoder
#TODO make this work by masking off a portion of an image.
#TODO make this use MaskingSubencoders to do its job
#TODO make the sampling work with the modified formulation of combined MaskingSubencoders.
#TODO make a mask inversion function
#  setup
import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)
logger.info('Loading FixedMaskingEncoder')
# 
class FixedMaskingEncoder(nn.Module):
  def __init__(self, *, subencoder, mask=torch.zeros([1, 28, 28])):
    super().__init__()
    if subencoder.data_shape != mask.shape:
        raise ValueError(f'Mask does not match subencoder data shape {subencoder.data_shape=} {mask.shape=}')
    #  save arguments
    self.mask = mask
    self.subencoder = subencoder
    # 
    self.flatten = nn.Flatten()
  def forward(self, input): #   TODO change this to the appropriate forward function
    network_output = self.estack(self.flatten(input))
    mean, network_stdev = torch.split(network_output, self.embedding_dimensionality, dim=1)
    stdev = self.uniform_stdev + torch.abs(network_stdev)
    return mean, stdev
  # 
logger.info('Verifying FixedMaskingEncoder integrity...')
fixed_masking_encoder_settings = {'embedding_dimensionality':2, 'data_shape':torch.Size([5,5]), 'layer_dimensionality':16}
fixed_masking_encoder = FixedMaskingEncoder(**fixed_masking_encoder_settings)
try:
  dummy_data = torch.zeros(torch.Size([2])+fixed_masking_encoder_settings['data_shape'])
  dummy_mean, dummy_stdev = fixed_masking_encoder(dummy_data)
  expected_output_size = torch.Size([2 , fixed_masking_encoder_settings['embedding_dimensionality']])
  if (dummy_mean.size() != expected_output_size):
    raise ValueError(f'Unexpected encoder mean output size {dummy_mean.size()}, expected {expected_output_size}')
  if (dummy_stdev.size() != expected_output_size):
    raise ValueError(f'Unexpected encoder stdev output size {dummy_stdev.size()}, expected {expected_output_size}')
except Exception as e:
  raise ValueError('FixedMaskingEncoder Inoperable. Repair needed.') from e
