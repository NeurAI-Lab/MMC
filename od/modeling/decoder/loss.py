from torch import nn


criterion_MSE = nn.MSELoss(reduction='mean')


def recon_loss(img, recon_img):
    loss = 0
    loss = criterion_MSE(img, recon_img)

    return loss