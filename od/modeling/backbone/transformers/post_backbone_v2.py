import torch.nn as nn

class PostBackboneV2(nn.Module):
    def __init__(self,patch_size,inplanes,upsample):
        super().__init__()
        self.patch_size=patch_size
        self.conv=ExtraLayers(inplanes,upsample)
    def forward(self, x):
        batch=x[0].shape[0]
        outs=[]
        for i in range(len(self.conv)):
            data=x[-1*i].reshape((batch, self.patch_size, self.patch_size, -1))
            data=data.permute(0,3,1,2)
            outs.append(self.conv[-1*i](data))
        return x




class ConvBnReluLayer(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, stride, bias=False):
        super(ConvBnReluLayer, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding,
                              stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class UpsampleBnReluLayer(nn.Module):

    def __init__(self, inplanes, upsample, bias=False):
        super(UpsampleBnReluLayer, self).__init__()
        if upsample=="Conv":
            self.upsample_layer=nn.ConvTranspose2d(inplanes, inplanes, 4, stride=2, padding=1)
        elif upsample=="Upsample":
            self.upsample_layer=nn.Upsample(scale_factor=2)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.upsample_layer(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
class ExtraLayers(nn.Module):

    def __init__(self, inplanes,upsample=""):
        super(ExtraLayers, self).__init__()
        # self.upsample_layer = nn.ConvTranspose2d(inplanes, inplanes, 4, stride=2,padding=1)
        self.upsample=upsample
        if self.upsample!="":
            self.upsample_layer =UpsampleBnReluLayer(inplanes,upsample)
        self.convbnrelu1_1 = ConvBnReluLayer(inplanes, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu1_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
        self.convbnrelu2_1 = ConvBnReluLayer(512, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu2_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)
        self.convbnrelu3_1 = ConvBnReluLayer(512, 256, kernel_size=1, padding=0, stride=1)
        self.convbnrelu3_2 = ConvBnReluLayer(256, 512, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        outs=[]

        if self.upsample != "":
            out_0=self.upsample_layer(x)
            outs.append(out_0)

        outs.append(x)
        out1_1 = self.convbnrelu1_1(x)
        out1_2 = self.convbnrelu1_2(out1_1)
        outs.append(out1_2)

        out2_1 = self.convbnrelu2_1(out1_2)
        out2_2 = self.convbnrelu2_2(out2_1)
        outs.append(out2_2)

        out3_1 = self.convbnrelu3_1(out2_2)
        out3_2 = self.convbnrelu3_2(out3_1)
        outs.append(out3_2)

        return outs