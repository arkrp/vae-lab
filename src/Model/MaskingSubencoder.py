#  notes
#basically this is just an encoder but we mask out portions of the image using a mask parameter to the forward.
#note that ones in the mask are the things that are considered blocking out information
#oddly this seemed to be a really really easy modification. Literally just add a mask to a standard encoder. neat!
# 
#  setup!
import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)
logger.info('Loading MaskingSubencoder')
# 
class MaskingSubencoder(nn.Module):
  def __init__(self, *, embedding_dimensionality=10, data_shape=torch.Size([1, 28, 28]), layer_dimensionality=128, uniform_stdev=0.05): #  
    super().__init__()
    #  save arguments
    self.embedding_dimensionality = embedding_dimensionality
    self.data_shape = data_shape
    self.layer_dimensionality = layer_dimensionality
    self.uniform_stdev = uniform_stdev
    # 
    #  make network components!
    self.flatten = nn.Flatten()
    self.estack = nn.Sequential(
        nn.BatchNorm1d(data_shape.numel()*2),
        nn.Linear(data_shape.numel(), layer_dimensionality),
        nn.BatchNorm1d(layer_dimensionality),
        nn.ReLU(),
        nn.Linear(layer_dimensionality, layer_dimensionality),
        nn.BatchNorm1d(layer_dimensionality),
        nn.ReLU(),
        nn.Linear(layer_dimensionality, embedding_dimensionality*2)
    )
    # 
  # 
  def forward(self, input, mask): #  
    #  validate the input
    #  validate input size
    if input[0].size() != self.data_shape:
        raise ValueError(f'input is of inappropriate size {input.size()} when expected size is {self.data_shape}')
    # 
    #  validate mask size
    if mask.size() != self.data_shape:
        raise ValueError(f'mask is of inappropriate size {mask.size()} when expected size is {self.data_shape}')
    # 
    # 
    #  run the network
    network_output = self.estack(
            torch.hstack(
                self.flatten(mask.censor(input)),
                self.flatten(mask)))
    mean, network_stdev = torch.split(network_output, self.embedding_dimensionality, dim=1)
    stdev = self.uniform_stdev + torch.abs(network_stdev)
    # 
    #  return the results
    return mean, stdev
    # 
  # 
#  verify subencoder integrity
logger.info('Verifying MaskingSubencoder integrity...')
masking_subencoder_settings = {'embedding_dimensionality':2, 'data_shape':torch.Size([5,5]), 'layer_dimensionality':16}
masking_subencoder = MaskingSubencoder(**masking_subencoder_settings)
try:
  dummy_data = torch.zeros(torch.Size([2])+masking_subencoder_settings['data_shape'])
  #TODO you need an actual mask class for this to work.
  dummy_mask = torch.zeros(torch.Size([2])+masking_subencoder_settings['data_shape'])
  dummy_mean, dummy_stdev = masking_subencoder(dummy_data, dummy_mask)
  expected_output_size = torch.Size([2 , masking_subencoder_settings['embedding_dimensionality']])
  if (dummy_mean.size() != expected_output_size):
    raise ValueError(f'Unexpected encoder mean output size {dummy_mean.size()}, expected {expected_output_size}')
  if (dummy_stdev.size() != expected_output_size):
    raise ValueError(f'Unexpected encoder stdev output size {dummy_stdev.size()}, expected {expected_output_size}')
except Exception as e:
  raise ValueError('MaskingSubencoder Inoperable. Repair needed.') from e
# 
