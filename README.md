# mmdetection meets clearml
[mmdetection](https://github.com/open-mmlab/mmdetection) training with [clearml](https://github.com/allegroai/clearml) hooks

## Prerequisites

- ClearML API keys [set up](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps)

## Working

- [x] Training
- [x] Evaluation (imbalanced as well)
- [x] Direct S3 reading

## Main usage

`cd tools && ./dist_train_clearml-example.sh <config file> <num gpus>`


### Training example

#### locally with ClearML logging and S3 bulk downloading

```bash
CONFIG=../configs/coco_mini/coco_mini.py
GPUS=1
PORT=${PORT:-29500}

export AWS_ENDPOINT_URL=https://play.min.io
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export CERT_PATH=

python3 dist_run.py --nproc_per_node=$GPUS --master_port=$PORT \
    --clml-run-locally --clml-proj mmdet --clml-task-name coco_mini_train --download-models 'resnet50_msra-5891d200.pth' --s3-models-bucket mmdet-wts --s3-models-path '' --download-data coco_mini --s3-data-bucket coco --s3-data-path '' \
    $(dirname "$0")/train.py ../configs/coco_mini/coco_mini.py --launcher pytorch --clearml ${@:3} 
```

[ClearML example task](https://app.community.clear.ml/projects/90233d6aa54844a3b1b66eea7d952b26/experiments/56ffbc53039e49e181d5f8aba7c03b5a/output/log)


#### locally with ClearML logging and S3 direct reading of images

Expects following environment variables to be set: 
- AWS_ENDPOINT_URL
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- CERT_PATH (optional)
- CERT_DL_URL (optional)

See [coco_mini_s3_direct.py](./configs/coco_mini/coco_mini_s3_direct.py) for example config file. 

```bash
# with the same env var exports
python dist_run.py --nproc_per_node=$GPUS --master_port=$PORT \
    --clml-run-locally --clml-proj mmdet --clml-task-name coco_mini_train_s3_direct --download-models 'resnet50_msra-5891d200.pth' --s3-models-bucket mmdet-wts --s3-models-path '' --s3-direct-read --download-data coco_mini/train.json coco_mini/val.json --s3-data-bucket coco --s3-data-path '' \
    $(dirname "$0")/train.py ../configs/coco_mini/coco_mini_s3_direct.py --launcher pytorch --clearml ${@:3} 
```

[ClearML example task](https://app.community.clear.ml/projects/90233d6aa54844a3b1b66eea7d952b26/experiments/e548da1ac7234fc2ab61161a3569f65d/output/log)


### Evaluation example

#### locally without clearml

```
python3 test.py ../configs/coco_mini/coco_mini.py weights/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth --fuse-conv-bn --write-result --eval bbox --show-dir eval_viz --eval-options classwise=True
```

#### with clearml (still locally)

```
python3 dist_run.py --nproc_per_node=1 --master_port=29500 --clml-run-locally --clml-proj mmdet --clml-task-name coco_mini-test --skip-s3 \
    test.py ../configs/coco_mini/coco_mini.py weights/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth --clearml --fuse-conv-bn --write-result --eval bbox --show-dir eval_viz --eval-options classwise=True
```

[Example ClearML task page](https://app.community.clear.ml/projects/90233d6aa54844a3b1b66eea7d952b26/experiments/d5492c5fb6a64a38b30ef38253e460fb/output/log)

#### with sub evals

```
python3 dist_run.py --nproc_per_node=1 --master_port=29500 --clml-run-locally --clml-proj mmdet --clml-task-name coco_mini-test --skip-s3 \
    test.py ../configs/coco_mini/coco_mini.py weights/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth --clearml --fuse-conv-bn --write-result --eval bbox --sub-eval datasets/coco_mini/sampled/  --show-dir eval_viz
```

[Example ClearML task page](https://app.community.clear.ml/projects/90233d6aa54844a3b1b66eea7d952b26/experiments/108e6f2662b14557849167fa8a95fcb7/output/artifacts/other/test/output)
