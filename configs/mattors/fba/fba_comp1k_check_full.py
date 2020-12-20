
# model settings
# 1. 去掉fushion
# 2. 单卡训练
model = dict(
    type='FBA',
    backbone=dict(
        type='FBAEncoderDecoder',
        encoder=dict(type='FBAEncoder', in_channels=11, block='resnet50_GN_WS'),
        decoder=dict(type='FBADecoder')),

    pretrained='work_dirs/fba/resnet_50_GN_WS_rename.pth',

    loss_alpha_l1=dict(type='L1Loss', loss_weight=1),
    loss_alpha_comp=dict(type='L1CompositionLoss', loss_weight=1),
    loss_alpha_grad=dict(type='GradientLoss', loss_weight=1),
    loss_alpha_lap=dict(type='LaplacianLoss', loss_weight=1),

    loss_f_l1=dict(type='L1Loss', loss_weight=0.25),
    loss_b_l1=dict(type='L1Loss', loss_weight=0.25),
    loss_fb_excl=dict(type='GradientExclusionLoss', loss_weight=0.25),
    loss_fb_comp=dict(type='L1CompositionLoss', loss_weight=0.25),
    loss_f_lap=dict(type='LaplacianLoss', loss_weight=0.25, channel=3),
    loss_b_lap=dict(type='LaplacianLoss', loss_weight=0.25, channel=3)
)
    
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = '/mnt/lustre/share/3darseg/segmentation/matting/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

img_norm_cfg_test =  dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], format='chw')

train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', flag='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='LoadImageFromFile', key='bg'),
    #dict(type='LoadImageFromFile', key='merged', save_original_img=True),

    dict(                       # 到时候换成更换后的FG
        type='CompositeFg',
        fg_dirs=[
            '/mnt/lustre/wangchenhao/code/gitlab/mmediting/data/adobe_train_fg_restimate'
        ],
        alpha_dirs=[
            '/mnt/lustre/wangchenhao/code/gitlab/mmediting/data/adobe_train_alpha'
        ]),    

    dict(type='Flip', keys=['alpha', 'fg', 'bg']),
    dict(type='RandomJitter'),  # 只针对fg
    dict(type='RandomGamma',keys=['fg', 'bg']),

    dict(type='MergeFgAndBg'),  # results['ori_merged']

    dict(
        type='CropAroundUnknown',
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],
        crop_sizes=[320, 480, 640]),
    dict(
        type='Resize',
        keys=['alpha', 'merged', 'ori_merged', 'fg', 'bg'],
        scale=(320, 320),
        keep_ratio=False),
    dict(type='GenerateTrimap', kernel_size=(3, 25)),

    dict(
        type='RescaleToZeroOne',
        keys=['merged', 'alpha', 'ori_merged', 'fg', 'bg', 'trimap']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),

    dict(type='FormatTrimap2Channel', key='trimap'), # results['trimap_1channel']
    dict(type='FormatTrimap6Channel', key='trimap'), # results['trimap_transformed']

    dict(
        type='Collect',
        keys=['merged', 'alpha', 'trimap', 'trimap_transformed', 'ori_merged', 'fg', 'bg', 'trimap_1channel'],
        meta_keys=[]),
    dict(
        type='ImageToTensor',
        keys=['merged', 'alpha', 'trimap', 'trimap_transformed', 'ori_merged', 'fg', 'bg', 'trimap_1channel']),    
    


]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        flag='grayscale',
        save_original_img=True),    # ori_alpha
    dict(
        type='LoadImageFromFile',
        key='trimap',
        flag='grayscale',
        save_original_img=True),    # ori_trimap
    dict(
        type='LoadImageFromFile', 
        key='merged', 
        #channel_order='rgb',
        save_original_img=True),    # ori_merged

    dict(type='CopyImage', key='trimap'),    # Copy a image for evaluate name: copy_trimap

    dict(type='RescaleToZeroOne', keys=['merged', 'trimap', 'ori_merged', 'ori_trimap']),

    dict(type='FormatTrimap2Channel', key='trimap'),
    dict(type='FormatTrimap2Channel', key='ori_trimap'),

    dict(
        type='ScaleInput',
        keys=['merged', 'trimap', 'ori_merged'],
        scale=1.0,
        scale_type=4),  # INTER_LANCZOS4=4

    dict(type='FormatTrimap6Channel', key='trimap'), # results['trimap_transformed']

    dict(type='Normalize', keys=['merged'], **img_norm_cfg),   # TODO: 删除自己实现的额GN，用统一的形式

    # dict(type='ImageToTensor', keys=['merged']),
    # dict(type='GroupNoraliseImage', keys=['merged'], **img_norm_cfg_test),

    
    dict(
        type='Collect',
        keys=['ori_merged','trimap' , 'merged', 'trimap_transformed'],
        meta_keys=[
            'merged_path', 'merged_ori_shape', 'ori_alpha', 'ori_trimap', 'copy_trimap'
        ]),
    
    # dict(type='ImageToTensor', keys=['ori_merged','trimap', 'trimap_transformed']),
    dict(type='ImageToTensor', keys=['ori_merged','trimap', 'trimap_transformed', 'merged']),

]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=False,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'adobe_restimate_train.json',
        data_prefix=data_root,
        pipeline=train_pipeline),
    # validation
    val_samples_per_gpu=1,
    val_workers_per_gpu=4,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'adobe/adobe_val.json',
        data_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'adobe/adobe_val.json',
        data_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
# optimizers_cfg = dict(type='Adam', lr=1e-5, momentun=0.9, weight_decay=0.0001)
# paramwise_cfg_1 = dict(custom_keys={'conv': dict(lr_mult=1, decay_mult=50), 'bn': dict(lr_mult=1, decay_mult=0.1})
# paramwise_cfg_2 = dict(custom_keys={)})

optimizers = dict(
    constructor='DefaultOptimizerConstructor',
    type='Adam',
    lr=1e-5,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={'conv':dict(lr_mult=1, decay_mult=50), 'bn':dict(lr_mult=1, decay_mult=0.1)})
    )

# learning policy
#lr_config = dict(policy='Fixed')
lr_config = dict(policy='Step', step=[43100*60], gamma=0.1, by_epoch=False)

# checkpoint saving
checkpoint_config = dict(interval=40000, by_epoch=False)
evaluation = dict(interval=40000, save_image=False)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='dim'))
    ])
# yapf:enable

# runtime settings
total_iters = 43100*100
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/fba/train'
load_from = None #'./work_dirs/fba/FBA_rename_pat.pth'
resume_from = 'work_dirs/fba/train-full/iter_2000000.pth'
workflow = [('train', 1)]
