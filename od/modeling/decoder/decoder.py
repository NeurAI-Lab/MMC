import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, hidden_dim,out_channel, norm_layer,solo_in_channel=None):
        super(DecoderBlock, self).__init__()
        self.projection = solo_in_channel!=None
        if self.projection:
            self.input_projection = nn.Conv2d(solo_in_channel, out_channel, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = norm_layer(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,padding=1, dilation=1, bias=False)
        self.bn2 = norm_layer(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, out_channel, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channel)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        inputs, skip = x

        inputs = F.interpolate(inputs, skip.size()[-2:], mode="bilinear", align_corners=False)
        if self.projection :
            residual = self.input_projection(inputs)
        else:
            residual = inputs

        x = torch.cat([inputs, skip], dim=1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        out += residual
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, in_channels,hidden_dim, out_channel):

        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.num_layers=len(in_channels)-1
        previous_in_channel=in_channels[self.num_layers]
        first_input=previous_in_channel
        for i in range(self.num_layers):
            input_channel=previous_in_channel+ in_channels[self.num_layers-i-1]
            self.blocks.append(DecoderBlock(input_channel, hidden_dim,out_channel, nn.BatchNorm2d,first_input))
            previous_in_channel=out_channel
            first_input=None



    def forward(self, inputs):
        x = inputs[-1]
        outs = []
        for i in range(self.num_layers):
            x = self.blocks[i]([x, inputs[self.num_layers - i - 1]])
            outs.append(x)
        outs.insert(0,torch.mean(outs[0],dim=(2,3),keepdim=True))
        return outs