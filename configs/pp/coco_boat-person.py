_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=3)),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='weights/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
    )
)
dataset_type = 'CocoDataset'
data_root = 'datasets/coco2017/'
classes = ('person', 'boat')
data = dict(
    train=dict(
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        classes=classes,
        ),
    val=dict(
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        classes=classes,
        ),
    test=dict(
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        classes=classes,
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
