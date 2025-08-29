from mmpretrain.models import BACKBONES, NECKS
from my_tools.mmpretrain.models.resnet3d import ResNet3d


@BACKBONES.register_module()
class zwbCustomResNet3DBackbone_512(ResNet3d):
    def __init__(self, **kwargs):
        base_channels: int = 10,
        super().__init__(**kwargs)

    def forward(self, x):
        x_expand = x.unsqueeze(1)
        y = super().forward(x_expand)
        return tuple([y])
