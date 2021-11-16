_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'

model = dict(
    backbone=dict (
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='weights/resnet50_msra-5891d200.pth')
    )
)
dataset_type = 'CocoDataset'
data_root = 'datasets/coco_mini/'
data = dict(
    train=dict(
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'images/',
        ),
    val=dict(
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'images/',
        ),
    test=dict(
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'images/',
        ))

evaluation = dict(interval=500, metric='bbox', save_best='bbox_mAP_50')
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(_delete_=True)
work_dir = './work_dirs/coco_mini'

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
