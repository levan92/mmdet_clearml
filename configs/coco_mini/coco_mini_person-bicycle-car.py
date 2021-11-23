_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=3)),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='weights/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
    )
)
dataset_type = 'CocoDataset'
data_root = 'datasets/coco_mini/'
classes = ('person', 'bicycle', 'car')
data = dict(
    train=dict(
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'images/',
        classes=classes,
        ),
    val=dict(
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'images/',
        classes=classes,
        ),
    test=dict(
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'images/',
        classes=classes,
        ))

evaluation = dict(interval=500, metric='bbox', save_best='bbox_mAP_50')
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(_delete_=True)
work_dir = './work_dirs/output'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
