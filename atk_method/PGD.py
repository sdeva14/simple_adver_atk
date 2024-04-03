import torch.nn as nn
import torch

from .base import Attack_Base

class Attack_PGD(Attack_Base):
    def __init__(self, model, device, targeted=False, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):

        super().__init__()

        self.model = model
        self.device = device
        self.targeted = targeted  # whether target specific labels or not

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        # self.supported_mode = ["default", "targeted"]
    
    def forward(self, images, labels=None, *args):
        '''
        empty for base wrapper
        -> will be used to perform the attack method
        '''

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images