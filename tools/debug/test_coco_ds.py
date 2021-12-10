import os

from mmcv import Config

from mmdet.datasets import build_dataset, replace_ImageToTensor

import mmcv_custom.fileio.file_client
import mmdet_custom.datasets.coco

environs = {
    'AWS_ENDPOINT_URL':'https://play.min.io',
}

for k,v in environs.items():
    os.environ[k] = v

config = '../configs/coco_mini/coco_mini-person_boat.py'
cfg = Config.fromfile(config)

# in case the test dataset is concatenated
samples_per_gpu = 1
if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True
    samples_per_gpu = max(
        [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
    )
    if samples_per_gpu > 1:
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

trainds = build_dataset(cfg.data.train)
testds = build_dataset(cfg.data.test)


