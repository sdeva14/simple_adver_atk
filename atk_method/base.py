import torch.nn as nn


class Attack_Base(nn.Module):
    '''
        input
        - atk_method: the attack method
        - model: the target model which we want to attack

    '''
    def __init__(self):

        super().__init__()

        # self.atk_method = atk_method
        # self.model = model
    
        # self.device = device
        # self.targeted = targeted  # whether target specific labels or not

        # TODO: variables if we need to train the framework?
    
    def get_logits(self, inputs, labels=None, *args, **kwargs):
        # if self._normalization_applied is False:
        #     inputs = self.normalize(inputs)
        logits = self.model(inputs).logits
        return logits

    def forward(self, inputs, labels=None, *args):
        '''
        empty for base wrapper
        -> will be used to perform the attack method
        '''
        raise NotImplementedError

    def save(self):
        '''
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.
        (quoted from the "torchattacks" library)



        '''