import mmengine.fileio as fileio
from mmcv.transforms import LoadImageFromFile
import mmcv
import numpy as np
from mmpretrain.registry import TRANSFORMS

@TRANSFORMS.register_module()
class zwbLoadImageFromFile(LoadImageFromFile):
    a=1
