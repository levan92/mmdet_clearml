# mmdetection meets clearml
[mmdetection](https://github.com/open-mmlab/mmdetection) training with [clearml](https://github.com/allegroai/clearml) hooks

## Prerequisites

- ClearML API keys set up: 

## main usage

`cd tools && ./dist_train_clearml-example.sh <config file> <num gpus>`

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
