# ------------------------------------------------------------------
# 0️⃣ 基础组件：日志、hook、保存策略等公共配置
# ------------------------------------------------------------------
_base_ = ['../_base_/default_runtime.py']

# 引入自定义算子 / 数据集 / 网络
custom_imports = dict(
    imports=[
        'azwb_tools.resnet3d',          # 3D ResNet 主干
        'azwb_tools.zwbCustomResNet3DBackbone',  # 自定义 3D backbone
        'azwb_tools.LoadSpecralImageTiff',  # 读取 .tiff 光谱图
        'azwb_tools.TiffDataset',           # Dataset 封装
    ],
    allow_failed_imports=False,
)

# ------------------------------------------------------------------
# 1️⃣ 数据根目录
# ------------------------------------------------------------------
data_root_train = r'/home/shenjy/zwb/data/cigarette_cls/train'
data_root_test  = r'/home/shenjy/zwb/data/cigarette_cls/val'

dataset_type = 'TiffDataset'   # 自定义 Dataset，返回 (C=1, T=16, H, W)

# ------------------------------------------------------------------
# 2️⃣ 预处理器：仅做 Normalize
#      mean / std 用训练集统计 16 波段的全局值
# ------------------------------------------------------------------
data_preprocessor = dict(
    mean=[107.9] * 16,
    std=[86.3]  * 16,
    to_rgb=False,          # 单通道灰度，无需转 RGB
)

# ------------------------------------------------------------------
# 3️⃣ 训练流水线：只做 Resize + Normalize（如需增强，在此加）
# ------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadSpecralImageTiff', backend_args=None,
         imdecode_backend='tifffile'),
    dict(type='Resize', scale=(512, 256)),   # 统一空间尺寸
    dict(type='PackInputs'),                 # 打包成 mm 标准格式
]

test_pipeline = [
    dict(type='LoadSpecralImageTiff', backend_args=None,
         imdecode_backend='tifffile'),
    dict(type='Resize', scale=(512, 256)),
    dict(type='PackInputs'),
]

# ------------------------------------------------------------------
# 4️⃣ DataLoader
#      train 小 batch=6 省显存；val 大 batch=10 平滑指标
# ------------------------------------------------------------------
train_dataloader = dict(
    batch_size=6,
    num_workers=6,
    dataset=dict(type=dataset_type, data_root=data_root_train,
                 split='', pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=10,
    num_workers=6,
    dataset=dict(type=dataset_type, data_root=data_root_test,
                 split='', pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# ------------------------------------------------------------------
# 5️⃣ 模型结构：ImageClassifier 包装
#      backbone: 3D-ResNet50
#      neck:     全局 3D 池化
#      head:     2048 → 3 全连接
# ------------------------------------------------------------------
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='zwbCustomResNet3DBackbone',  # 自定义 ResNet3D-50
        depth=50,
        in_channels=1,        # 每帧 1 通道，就把光谱当作时间，那么原本的颜色通道就是只剩一通道了
        conv1_kernel=(4, 7, 7),  # 时间核 4，空间 7×7
        conv1_stride_t=1,        # 时间维 stride=1 → 保持 16 波段
        pool1_stride_t=1,        # 同上
        # 如果需要时间下采样，把 strides/dilations 打开即可
    ),
    neck=dict(type='GlobalAveragePooling', dim=3),  # (B,C,T,H,W)→(B,C,1,1,1)
    head=dict(
        type='LinearClsHead',
        num_classes=3,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss'),
        topk=(1,),
    ),
)

# ------------------------------------------------------------------
# 6️⃣ 训练策略
#      max_epochs=500，每 10 epoch 验证一次
# ------------------------------------------------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=500,
    val_interval=10,
)
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='Accuracy', topk=(1,))

# ------------------------------------------------------------------
# 7️⃣ 优化器 & 正则
#      SGD + momentum + weight decay
# ------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
)

# ------------------------------------------------------------------
# 8️⃣ 钩子：每 10 epoch 存一次，仅保留最近 1 个 ckpt
#      load_from 用于热启预训练权重
# ------------------------------------------------------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=1,
        save_best='auto',   # 同时保留 best（不占名额）
    )
)
load_from = r'/home/shenjy/zwb/mmpretrain/work_dirs/resnet3D2/epoch_500.pth'