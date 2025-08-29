from mmpretrain.models import BACKBONES, NECKS
from .resnet3d import ResNet3d


@BACKBONES.register_module()
class zwbCustomResNet3DBackbone(ResNet3d):
    def __init__(self, in_channels=1, **kwargs):  
        super().__init__(in_channels=in_channels, **kwargs)

    def forward(self, x):

        x_expand = x.unsqueeze(1)  # 光谱当作时间，那么原本的颜色通道就是只剩一通道了
        y = super().forward(x_expand)
        return tuple([y])
