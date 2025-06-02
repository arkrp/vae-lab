#TODO make this so it works when masking in batches, because it needs to be able to.
#  import stuff!
import torch
# 
def assert_same_size(a, b, error_msg): #  
    if a.size()!=b.size():
        raise ValueError(error_msg)
# 
class Mask(): #  
    def __init__(self, mask_tensor): #  
        #  validate input!
        if not torch.all((mask_tensor==0.0) or (mask_tensor==1.0)):
            raise ValueError(f'Error in creation of Mask, mask_tensor must consist solely of ones and zeroes {mask_tensor=}')
        # 
        #  accept input!
        self.mask_tensor = mask_tensor
        # 
    # 
    def union(self, other_mask): #  
        """this returns the union the Mask with another Mask"""
        assert_same_size(self, other_mask, 'Attempted to union Masks of different size')
        new_mask_tensor = torch.zeros(self.size())
        new_mask_tensor[self.mask_tensor==1.0] = 1.0
        new_mask_tensor[other_mask.mask_tensor==1.0] = 1.0
        return Mask(new_mask_tensor)
    # 
    def conjunction(self, other_mask): #  
        """this returns the conjunction of the Mask with another Mask"""
        assert_same_size(self, other_mask, 'Attempted to conjunction Masks of different size')
        new_mask_tensor = torch.ones(self.size())
        new_mask_tensor[self.mask_tensor==0.0] = 0.0
        new_mask_tensor[other_mask.mask_tensor==0.0] = 0.0
        return Mask(new_mask_tensor)
    # 
    def invert(self): #  
        """returns an inverted copy of the Mask"""
        new_mask_tensor = torch.zeros(self.size())
        new_mask_tensor[self.mask_tensor==0.0] = 1.0
        return Mask(new_mask_tensor)
    # 
    def __repr__(self): #  
        return(f'Mask({self.mask_tensor})')
    # 
    def size(self): #  
        return self.mask_tensor.size()
    # 
    def censor(self, tensor): #  
    """
    Uses the Mask to censor a tensor. Takes in a tensor and returns a censored tensor
    """
    return_value = tensor.clone()
    return_value[self.mask_tensor==1.0] = 0.0
    return return_value
    # 
# 

