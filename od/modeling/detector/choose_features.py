from torch import nn

"""Choose the specific features needed for respective OD head """
class choose_features(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, features):

        result_features = []
        """If multiscale feature is enabled, the feature sizes will be different each time.
           Hence, using feature index for this case"""
        if self.cfg.DATA_LOADER.MULTISCALE:
            assert len(self.cfg.MODEL.BACKBONE.FEATURE_INDEX) > 0, "Feature Index should be used while using Multiscale."
            result_features = [features[i] for i in self.cfg.MODEL.BACKBONE.FEATURE_INDEX]
        else:
            feature_index = self.cfg.MODEL.BACKBONE.FEATURE_INDEX
            feature_sizes = self.cfg.MODEL.BACKBONE.FEATURE_MAPS

            if feature_index and feature_sizes:
                if len(feature_index)!=len(feature_sizes):
                    assert "Mismatch between Feature Index and Feature Map"
                for ft_size,ft_idx in zip(feature_sizes,feature_index):
                    for i, ft in enumerate(features):
                        if ft.shape[2] == ft_size and i == ft_idx:
                            result_features.append(ft)
                        else:
                            assert "Mismatch between Feature Index and Feature Map"
            elif feature_index:
                result_features = [features[i] for i in self.cfg.MODEL.BACKBONE.FEATURE_INDEX]
            else:
                for ft in features:
                    if ft.shape[2] in feature_sizes:
                            result_features.append(ft)

        return result_features

"""Choose the specific features needed for respective backbone for KD (AT loss) """
class choose_features_backbone(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.at_layers = cfg.KD.DISTILL_AT_LAYERS

    def forward(self, features):
        # if self.at_layers == 1:
        #     layers = [2]
        # elif self.at_layers == 2:
        #     layers = [1,3]
        # elif self.at_layers == 3:
        #     layers = [1, 3, 5]
        result_features = []
        result_features = [features[layer] for layer in self.at_layers]

        return result_features