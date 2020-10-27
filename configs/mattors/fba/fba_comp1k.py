# model settings
# model = dict(
#     type='DIM',
#     backbone=dict(
#         type='SimpleEncoderDecoder',
#         encoder=dict(type='VGG16', in_channels=4),
#         decoder=dict(type='PlainDecoder')),
#     refiner=dict(type='PlainRefiner'),
#     pretrained=None,
#     loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5),
#     loss_comp=dict(type='CharbonnierCompLoss', loss_weight=0.5),
#     loss_refine=dict(type='CharbonnierLoss'))
# train_cfg = dict(train_backbone=True, train_refiner=True)
# test_cfg = dict(refine=True, metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

model = dict(
    type='FBA',
    backbone=dict(
        type='FBAEncoderDecoder',
        encoder=dict(type='FBAEncoder', in_channels=11, block='resnet50_GN_WS'),
        decoder=dict(type='FBADecoder')),
    pretrained=None,
    loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5),
    loss_comp=dict(type='CharbonnierCompLoss', loss_weight=0.5)
)
    
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = '/mnt/lustre/share/3darseg/segmentation/matting/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

train_pipeline = [
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
    dict(
        type='LoadImageFromFile', 
        key='merged', 
        save_original_img=True),

    dict(type='FormatTrimap2Channel'),
    # dict(type='Pad', keys=['trimap', 'merged', 'ori_merged'], mode='reflect'),    # 要不要pad原始图
    dict(type='Pad', keys=['trimap', 'merged', 'ori_merged'], mode='reflect'),
    dict(type='RescaleToZeroOne', keys=['merged', 'trimap', 'ori_merged']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),   
    dict(type='FormatTrimap6Channel'), # results['trimap_transformed']
    dict(
        type='Collect',
        keys=['ori_merged','trimap' , 'merged', 'trimap_transformed'],
        meta_keys=[
            'merged_path', 'pad', 'merged_ori_shape', 'ori_alpha', 'ori_trimap'
        ]),
    dict(type='ImageToTensor', keys=['ori_merged','trimap' , 'merged', 'trimap_transformed']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=False,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'adobe/adobe_train.json',
        data_prefix=data_root,
        pipeline=train_pipeline),
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
optimizers = dict(type='Adam', lr=0.00001)
# learning policy
lr_config = dict(policy='Fixed')

# checkpoint saving
checkpoint_config = dict(interval=40000, by_epoch=False)
evaluation = dict(interval=40000, save_image=False)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='dim'))
    ])
# yapf:enable

# runtime settings
total_iters = 1000000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/dim_finetune'
load_from = './work_dirs/dim_stage2/latest.pth'
resume_from = None
workflow = [('train', 1)]