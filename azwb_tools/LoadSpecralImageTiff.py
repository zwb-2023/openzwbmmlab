from mmcv.transforms import LoadImageFromFile

import numpy as np
from mmpretrain.registry import DATASETS, TRANSFORMS
import cv2


@TRANSFORMS.register_module()
class LoadSpecralImageTiff(LoadImageFromFile):

    def transform(self, results: dict):

        filename = results['img_path']
        img = cv2.imreadmulti(filename)
        img = np.array(img[1])
        img = img.transpose(1, 2, 0)
        # print(img.shape)


        #     if self.file_client_args is not None:
        #         file_client = fileio.FileClient.infer_client(
        #             self.file_client_args, filename)
        #         img_bytes = file_client.get(filename)
        #     else:
        #         img_bytes = fileio.get(
        #             filename, backend_args=self.backend_args)
        #     img = mmcv.imfrombytes(
        #         img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        # except Exception as e:
        #     if self.ignore_empty:
        #         return None
        #     else:
        #         raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        ### 因为当原始形状为 (C, H, W)，即：C: 通道数（例如，RGB的3个通道）。H: 图像的高度。W: 图像的宽度。
        ### 倘若图像为正常形状，如：(H, W, C)，则无需调整。



        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results
    