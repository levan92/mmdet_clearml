_base_ = './pp_modir_lsj.py'

s3_data_root = 's3://pixelplus-s3/data/ObjectDetection/june2021collated/'
local_data_root = 'datasets/june2021collated/'

data = dict(
    train=dict(
        ann_file=local_data_root + 'train.json',
        img_prefix=s3_data_root + 'images/',
        ),
    val=dict(
        ann_file=local_data_root + 'val.json',
        img_prefix=s3_data_root + 'images/',
        ),
    test=dict(
        ann_file=local_data_root + 'val.json',
        img_prefix=s3_data_root + 'images/',
        ))
