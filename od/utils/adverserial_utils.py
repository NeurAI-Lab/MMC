import torch


def forward_transform(image, info):
    image[:, 0, :, :] = (image[:, 0, :, :] - info["mean"][0]) / info["std"][0]
    image[:, 1, :, :] = (image[:, 1, :, :] - info["mean"][1]) / info["std"][1]
    image[:, 2, :, :] = (image[:, 2, :, :] - info["mean"][2]) / info["std"][2]
    return image


def clamp_tensor(image, upper_bound, lower_bound):
    image = torch.where(image > upper_bound, upper_bound, image)
    image = torch.where(image < lower_bound, lower_bound, image)
    return image