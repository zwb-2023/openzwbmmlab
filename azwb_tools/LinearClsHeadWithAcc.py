from mmpretrain.models import HEADS
from mmpretrain.models.heads.linear_head import LinearClsHead
import torch


@HEADS.register_module()
class LinearClsHeadWithAcc(LinearClsHead):

    def __init__(self, num_classes: int, in_channels: int, init_cfg: dict = dict(
        type='Normal', layer='Linear', std=0.01), **kwargs):
        super(LinearClsHeadWithAcc, self).__init__(num_classes, in_channels, init_cfg, **kwargs)
        self.features1 = []

    # def __del__(self):
    #     # 保存成numpy
    #     import numpy as np
    #     import pickle
    #     serialized_data = pickle.dumps(self.features1)
    #     np.save('data_list.npy', serialized_data)

    def loss(self, x, data_samples, **kwargs) -> dict:
        x = self.pre_logits(x)
        # for i in range(len(x)):
        #     self.features1.append({data_samples[i].img_path: x[i]})
        cls_score = self.fc(x)
        gt_label = torch.tensor([data.gt_label
                                 for data in data_samples]).to(x[0].device)
        # losses = self.loss(cls_score, gt_label, **kwargs)
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        acc = (cls_score.argmax(dim=-1) == gt_label).float().mean()
        result = {'acc': acc}
        result.update(losses)
        return result