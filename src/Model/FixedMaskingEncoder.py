#  notes
#FixedMaskingEncoder is meant to be a proof of concept drop in replacement for NonUniformEncoder
#this assumes that the z distribution is a Identity guassian
#fundemental_error is a variable representing that the combination of the two distributions we are given by the masking is an impossible ditribution. When this occurs we would like to know so we can peanalize it about all else.
# on a sidenote. I am beginning to question the viability of this framework. Not that it won't work, but wouldn't it make more sense to train a seperate reverse
#TODO make this use MaskingSubencoders to do its job
#TODO make a training encoder and training loss which is appropriate for our model and includes the fundemental error.
# 
#  setup
#  import stuff!
import torch
from torch import nn
import logging
import Mask from Helper.Mask
# 
#  logging!
logger = logging.getLogger(__name__)
logger.info('Loading FixedMaskingEncoder')
# 
# 
def combined_latent_distribution(latent_distribution_1, latent_distribution_2): #  
    mean_1, stdev_1 = latent_distribution_1
    mean_2, stdev_2 = latent_distribution_2
    mean_b = torch.zeros(mean_1.size())
    var_1 = stdev_1**2
    var_2 = stdev_2**2
    var_b = torch.ones(stdev_1.size())
    denominator = var_1*var_b + var_2*var_b - var_1*var_2
    just_negatives = torch.min(0,denominator)
    fundemental_error = -just_negatives.sum(axis=1)
    mean_numerator = mean_1*var_2*var_b + mean_2*var_1*var_b + mean_b*var_1*var_2
    var_numerator = var_1*var_2*var_b
    var = torch.abs(var/denominator) # we abs it because anything that would be negative ultimately gets ignored due to fundemental error in loss computation
    mean = mean/denominator
    stdev = torch.sqrt(var)
    return mean, stdev, fundemental_error
# 
class FixedMaskingEncoder(nn.Module): #  
  def __init__(self, *, subencoder, mask=Mask(torch.zeros([1, 28, 28]))): #  
    super().__init__()
    #  Make sure the subencoder is appropriately sized
    if subencoder.data_shape != mask.shape:
        raise ValueError(f'Mask does not match subencoder data shape {subencoder.data_shape=} {mask.shape=}')
    # 
    #  attributes!
    self.mask = mask
    self.subencoder = subencoder
    self.flatten = nn.Flatten()
    # 
  # 
  def forward(self, input): #  
    #  get the individual distributions
    mask1 = self.mask
    mask2 = self.mask.invert()
    masked_input_1 = mask1.censor(input)
    masked_input_2 = mask2.censor(input)
    latent_distribution_1 = self.subencoder(maked_input_1, mask1)
    latent_distribution_2 = self.subencoder(maked_input_2, mask2)
    # 
    #  combine them!
    combined_latent_distribution = fuse_guassian_distributions(
            latent_distribution_1,
            latent_distribution_2)
    # 
    #  return it!
    return combined_latent_distribution
    # 
  # 
# 
#  Check encoder integrity
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
# 
