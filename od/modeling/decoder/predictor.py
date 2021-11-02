import torch
import torch.nn as nn
import torch.nn.functional as F

class DecPredictor(nn.Module):
    def __init__(self, in_channels, out_channel,in_features_len,hidden_channels,output_image_size):
        super(DecPredictor, self).__init__()
        self.convs = nn.ModuleList()
        self.features_num=in_features_len
        for i in range(in_features_len):
            self.convs.append(nn.Conv2d(in_channels,  hidden_channels, kernel_size=1))
        self.final_conv = nn.Conv2d(hidden_channels*in_features_len , out_channel, kernel_size=1)
        self.out_image_size=(output_image_size,output_image_size)


    def forward(self, inputs):
        outs = []
        for i in range(self.features_num):
            x = self.convs[i](inputs[i])
            outs.append(
                F.interpolate(x, self.out_image_size, mode="bilinear", align_corners=False))

        all_outs = torch.cat(outs, dim=1)
        out = self.final_conv(all_outs)
        return out

def back_transform(image, cfg, scale=255.):
    image[:, 0, :, :] = (image[:, 0, :, :] *
                         cfg.INPUT.STANDARDIZATION_STDEV[0]) + cfg.INPUT.STANDARDIZATION_MEAN[0]
    image[:, 1, :, :] = (image[:, 1, :, :] *
                         cfg.INPUT.STANDARDIZATION_STDEV[1]) + cfg.INPUT.STANDARDIZATION_MEAN[1]
    image[:, 2, :, :] = (image[:, 2, :, :] *
                         cfg.INPUT.STANDARDIZATION_STDEV[2]) + cfg.INPUT.STANDARDIZATION_MEAN[2]
    return image * scale