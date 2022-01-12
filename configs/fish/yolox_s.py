_base_ = '../yolox/yolox_s_8x8_300e_coco.py'
classes = ('vehicle',)

model = dict(
    bbox_head=dict(
        type='YOLOXHead', num_classes=len(classes), in_channels=128, feat_channels=128),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='weights/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
    )
)

dataset_type = 'CocoDataset'
data_root = 'datasets/'

data = dict(
    train=dict(
        dataset=dict(
            type='ClassBalancedDataset',
            ann_file=data_root + 'train/train.json',
            img_prefix=data_root + 'train/images/',
            classes=classes,
        )),
    val=dict(
        ann_file=data_root + 'val/val.json',
        img_prefix=data_root + 'val/images/',
        classes=classes,
        ),
    test=dict(
        ann_file=data_root + 'test/test.json',
        img_prefix=data_root + 'test/images/',
        classes=classes,
        ))

evaluation = dict(interval=1, metric='bbox', iou_thrs=[0.3], save_best='bbox_mAP_50')
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(_delete_=True)
work_dir = './work_dirs/output'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
