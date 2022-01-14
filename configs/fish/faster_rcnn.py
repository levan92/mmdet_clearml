_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py'
classes = ('vehicle',)

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=len(classes))),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='weights/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
    )
)

dataset_type = 'CocoDataset'
data_root = 'datasets/'
data = dict(
    train=dict(
        ann_file=data_root + 'train/train.json',
        img_prefix=data_root + 'train/images/',
        classes=classes,
        ),
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

evaluation = dict(interval=1, metric='bbox', iou_thrs=[0.3], save_best='bbox_mAP')
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(_delete_=True)
work_dir = './work_dirs/output'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
