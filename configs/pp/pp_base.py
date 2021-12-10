_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'

classes=('ship',)

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=len(classes))),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='weights/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
    )
)

file_client_args = dict(backend='s3')
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'CocoDataset'
s3_data_root = 's3://pixelplus-s3/data/ObjectDetection/june2021collated/'
local_data_root = 'datasets/june2021collated/'

data = dict(
    train=dict(
        ann_file=local_data_root + 'train.json',
        img_prefix=s3_data_root + 'images/',
        pipeline=train_pipeline,
        classes=classes
        ),
    val=dict(
        ann_file=local_data_root + 'val.json',
        img_prefix=s3_data_root + 'images/',
        pipeline=test_pipeline,
        classes=classes,
        samples_per_gpu=2,
        ),
    test=dict(
        ann_file=local_data_root + 'val.json',
        img_prefix=s3_data_root + 'images/',
        pipeline=test_pipeline,
        classes=classes,
        samples_per_gpu=2,
        ))

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(_delete_=True)
work_dir = './work_dirs/output'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
