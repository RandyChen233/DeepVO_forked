import torch
import numpy as np
from params import par



def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [-0.5, 0.5] range for each pixel
    perturbed_image = torch.clamp(perturbed_image, -0.5, 0.5)
    # Return the perturbed image
    return perturbed_image


def denorm(device, batch, mean=list(par.img_means), std = list(par.img_stds)):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
        
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)
        
    # De-normalize using mean and std
    # We know that each "batch" has shape ([batch,seq,channel,width,height])
    batch_update = batch.clone()
    
    for channel in range(3):
        batch_update[:, :, channel, :, :] = batch[:, :, channel, :, :] * std[channel] + mean[channel]

    
    return batch_update