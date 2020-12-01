# model settings

model = dict(
    type='InductiveFilter',
    backbone=dict(
        type='SimpleEncoderDecoder',
        encoder=dict(type='IndexNetEncoder', in_channels=4, freeze_bn=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        decoder=dict(
            type='IGFIndexNetDecoder',
            norm_cfg=dict(type='SyncBN', requires_grad=True))),
    loss_global=dict(type='L1Loss', loss_weight=10.0, sample_wise=True),
    loss_local=dict(type='L1Loss', loss_weight=1.0, sample_wise=True),
    loss_sync=dict(type='MSELoss', loss_weight=1.0, sample_wise=True),
    loss_gabor=dict(type='GradientLoss', loss_weight=1.0),
    pretrained='/mnt/lustre/chengjunqi/model_zoo/mobilenet_v2.pth')
#loss_gabor=None)
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = '/mnt/lustre/share/3darseg/segmentation/matting/'

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', flag='grayscale'),
    dict(type='LoadImageFromFile', key='merged', save_original_img=True),
    dict(type='GenerateTrimapWithDistTransform', dist_thr=20),
    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'ori_merged', 'trimap'],
        crop_sizes=[320, 480, 640, 720, 960],
        interpolations=[
            'bicubic', 'bicubic', 'bicubic', 'nearest'
        ]),
    dict(
        type='Resize',
        keys=['trimap'],
        scale=(320, 320),
        keep_ratio=False,
        interpolation='nearest'),
    dict(
        type='Resize',
        keys=['alpha', 'merged'],
        scale=(320, 320),
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='Flip', keys=['alpha', 'merged', 'trimap']),
    dict(type='RescaleToZeroOne', keys=['merged', 'alpha', 'trimap']),
    dict(type='GenerateMaskFromAlpha', kernel_size=(5, 30), mul=True),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Collect',
        keys=['merged', 'alpha', 'mask', 'local_mask', 'trimap', 'mask_2'],
        meta_keys=[]),
    dict(
        type='ImageToTensor',
        keys=['merged', 'alpha', 'mask', 'local_mask', 'trimap', 'mask_2']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', flag='grayscale', save_original_img=True),
    dict(type='LoadImageFromFile', key='mask', flag='grayscale', save_original_img=True),
    dict(type='LoadImageFromFile', key='merged', save_original_img=True),
    dict(type='RescaleToZeroOne', keys=['merged', 'alpha']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Resize',
        keys=['mask'],
        size_factor=32,
        interpolation='nearest'),
    dict(
        type='Resize',
        keys=['merged'],
        size_factor=32,
        interpolation='bicubic'),
    dict(
        type='Collect',
        keys=['merged', 'mask'],
        meta_keys=[
            'merged_path', 'interpolation', 'merged_ori_shape', 'ori_alpha',
            'ori_mask'
        ]),
    dict(type='ImageToTensor', keys=['merged', 'mask']),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    val_samples_per_gpu=1,
    val_workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'adobeclearhair_train.json',
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'hair_val.json',
        data_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'hair_val.json',
        data_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
#optimizers = dict(type='Adam', lr=1e-4, betas=[0.5, 0.999])
optimizers = dict(
    constructor='DefaultOptimizerConstructor',
    type='Adam',
    lr=2e-3,
    paramwise_cfg=dict(custom_keys={'encoder.layers': dict(lr_mult=0.01)}))
# learning policy
lr_config = dict(policy='Step', step=[52000, 67600], gamma=0.1, by_epoch=False)
#lr_config = dict(policy='Fixed')

# checkpoint saving
checkpoint_config = dict(interval=3000, by_epoch=False)
evaluation = dict(interval=3000, save_image=False, gpu_collect=False)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='gca'))
    ])
# yapf:enable

# runtime settings
total_iters = 78000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/gca'
load_from = None#'./work_dirs/igf17/iter_72000.pth'
resume_from = None#'./work_dirs/igf/iter_36000.pth'
workflow = [('train', 1)]