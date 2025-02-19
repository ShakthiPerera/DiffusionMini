import torch

def default_loss(x, eps_pred, eps, criterion):
    '''Calculate the default loss for the ddpm'''
        
    loss = criterion(eps_pred, eps)

    return loss, loss