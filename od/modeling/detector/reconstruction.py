import torch.nn as nn
from od.modeling.decoder.decoder import Decoder
from od.modeling.decoder.predictor import DecPredictor, back_transform

class Reconstruction(nn.Module):
    def __init__(self, cfg):

        super(Reconstruction, self).__init__()
        self.cfg = cfg
        self.mode = cfg.KD.AUX_RECON_MODE
        decoder_inchannels = []
        for i in cfg.MODEL.BACKBONE.OUT_CHANNELS:
            decoder_inchannels.append(i)
        self.decoder = Decoder(in_channels=decoder_inchannels, hidden_dim=cfg.MODEL.DECODER.DECODER_HIDDEN_DIM,
                               out_channel=cfg.MODEL.DECODER.DECODER_OUT_CHANNEL)
        self.decPredictor = DecPredictor(in_channels=cfg.MODEL.DECODER.DECODER_OUT_CHANNEL, out_channel=3,
                                         in_features_len=len(decoder_inchannels) - 1,
                                         hidden_channels=cfg.MODEL.DECODER.PREDICTOR_HIDDEN_DIM,
                                         output_image_size=cfg.INPUT.IMAGE_SIZE)

    def forward(self, features_bt, features_bs):
        if self.mode == "normal":
            features_decode_rgb = self.decoder(features_bt)
            recon_img_rgb = self.decPredictor(features_decode_rgb)
            recon_img_rgb = back_transform(recon_img_rgb, self.cfg, scale=1)
            return recon_img_rgb, None

        elif self.mode == "cross":
            features_decode_rgb = self.decoder(features_bt)
            features_decode_thm = self.decoder(features_bs)

            recon_img_rgb = self.decPredictor(features_decode_thm)
            recon_img_rgb = back_transform(recon_img_rgb, self.cfg, scale=1)
            recon_img_thm = self.decPredictor(features_decode_rgb)
            recon_img_thm = back_transform(recon_img_thm, self.cfg, scale=1)

            return recon_img_rgb, recon_img_thm
        else:
            assert False,"Wrong Recon Mode"
