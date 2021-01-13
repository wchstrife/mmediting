# model settings
# indexnet 添加前景预测，clip不参与训练
# fg bg 局部算Loss
# 在新的绿幕数据上ft，4张卡训练
model = dict(
    type='IndexNetFG',
    backbone=dict(
        type='IndexnetEncoderDecoderFG',
        encoder=dict(type='IndexNetEncoder', in_channels=4, freeze_bn=True, norm_cfg=dict(type='SyncBN', requires_grad=True)),
        decoder=dict(type='IndexNetDecoderFG', norm_cfg=dict(type='SyncBN', requires_grad=True))),

    loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5, sample_wise=True),
    loss_comp=dict(
        type='CharbonnierCompLoss', loss_weight=1.5, sample_wise=True),

    loss_f_l1=dict(type='L1Loss', loss_weight=0.25, sample_wise=True),
    loss_b_l1=dict(type='L1Loss', loss_weight=0.25,  sample_wise=True),
    loss_fb_comp=dict(type='L1CompositionLoss', loss_weight=0.25, sample_wise=True),

    pretrained='work_dirs/indexnet/mobilenet_v2.pth')
# model training and testing settings
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = '/mnt/lustre/share/3darseg/segmentation/matting/green_matting/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', flag='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='LoadImageFromFile', key='bg'),
    dict(type='LoadImageFromFile', key='merged', save_original_img=True),
    dict(type='GenerateTrimapWithDistTransform', dist_thr=20),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg', 'trimap'],
        crop_sizes=[512, 640, 720],
        interpolations=[
            'bicubic', 'bicubic', 'bicubic', 'bicubic', 'bicubic', 'nearest'
        ]),
    dict(
        type='Resize',
        keys=['trimap'],
        scale=(512, 512),
        keep_ratio=False,
        interpolation='nearest'),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],
        scale=(512, 512),
        keep_ratio=False,
        interpolation='bicubic'),
    dict(
        type='Flip',
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg', 'trimap']),
    dict(
        type='RescaleToZeroOne',
        keys=['merged', 'alpha', 'ori_merged', 'fg', 'bg', 'trimap']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Collect',
        keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg'],
        meta_keys=[]),
    dict(
        type='ImageToTensor',
        keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg']),
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        flag='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        flag='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged',save_original_img=True),
    dict(type='RescaleToZeroOne', keys=['merged', 'trimap', 'ori_merged']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Resize',
        keys=['trimap'],
        size_factor=32,
        interpolation='nearest'),
    dict(
        type='Resize',
        keys=['merged', 'ori_merged'],
        size_factor=32,
        interpolation='bicubic'),
    dict(
        type='Collect',
        keys=['merged', 'trimap', 'ori_merged'],
        meta_keys=[
            'merged_path', 'interpolation', 'merged_ori_shape', 'ori_alpha',
            'ori_trimap'
        ]),
    dict(type='ImageToTensor', keys=['merged', 'trimap', 'ori_merged']),
]
data = dict(
    # train
    samples_per_gpu=4,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Data_adobe_new_gs_bg/list/train_change.json',
        data_prefix=data_root,
        pipeline=train_pipeline),
    # validation
    val_samples_per_gpu=1,
    val_workers_per_gpu=4,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Data_adobe_new_gs_bg/list/val_change.json',
        data_prefix=data_root,
        pipeline=test_pipeline),
    # test
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Data_adobe_new_gs_bg/list/val_change.json',
        data_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizers = dict(
    constructor='DefaultOptimizerConstructor',
    type='Adam',
    lr=1e-2,
    paramwise_cfg=dict(custom_keys={'encoder.layers': dict(lr_mult=0.01)}))
# learning policy
lr_config = dict(policy='Step', step=[52000, 67600], gamma=0.1, by_epoch=False)

# checkpoint saving
checkpoint_config = dict(interval=2600, by_epoch=False)
evaluation = dict(interval=2600, save_image=False)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='indexnet'))
    ])
# yapf:enable

# runtime settings
total_iters = 78000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/indexnet'
load_from = None
resume_from = None
workflow = [('train', 1)]
