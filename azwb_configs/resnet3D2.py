_base_ = [
    # '../_base_/models/resnet18.py',
    # '../_base_/datasets/imagenet_bs32.py',
    # '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=[

        'azwb_tools.resnet3d',
        'azwb_tools.LinearClsHeadWithAcc',

        'azwb_tools.zwbCustomResNet3DBackbone_512',
        'azwb_tools.LoadSpecralImageTiff',
        'azwb_tools.TiffDataset'
     ],
    allow_failed_imports=False)
# dataset settings
data_root_train = rf'/home/shenjy/zwb/data/cigarette_cls/train'
data_root_test = rf'/home/shenjy/zwb/data/cigarette_cls/val'

dataset_type = 'TiffDataset'
data_preprocessor = dict(
    mean=[107.9]*16,
    std=[86.3]*16,
    # convert image from BGR to RGB
    to_rgb=False,
)

train_pipeline = [
    dict(type='LoadSpecralImageTiff', backend_args=None
         , imdecode_backend='tifffile'
         ),
    # dict(type='RandomResizedCrop', scale=64),
    # dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='Resize', scale=(512, 256)),

    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadSpecralImageTiff', backend_args=None
         , imdecode_backend='tifffile'
         ),
    dict(type='Resize', scale=(512, 256)),

    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=6,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_train,
        split='',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=10,
    num_workers=6,
    dataset=dict(
        type=dataset_type,
        data_root=data_root_test,
        split='',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)




model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='zwbCustomResNet3DBackbone_512',
        depth=50,
        in_channels=1,  # 每个“时间帧”的通道数
        conv1_kernel=(4, 7, 7),  # 时间维度=5
        conv1_stride_t=1,
        pool1_stride_t=1,

        
        # strides=((1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2)),  # 时间维度不下采样
        # dilations=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ),
    neck=dict(type='GlobalAveragePooling', dim=3),
    head=dict(
        type='LinearClsHead',
        num_classes=3,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss'),
        topk=(1,),
    ),
)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=500,
    val_interval=10,
)

val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='Accuracy', topk=(1,))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
)


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,           # 每个 epoch 保存一次
        max_keep_ckpts=1,     # 只保留最近 1 个
        save_best='auto',     # 可选：同时保留 best（不占用 max_keep_ckpts 名额）
    )
)
load_from = rf'/home/shenjy/zwb/mmpretrain/work_dirs/resnet3D2/epoch_500.pth'