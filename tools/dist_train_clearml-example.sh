#!/usr/bin/env bash

CONFIG=$1
GPUS=1
PORT=${PORT:-29500}


######## local ########
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python dist_run.py --nproc_per_node=$GPUS --master_port=$PORT \
#     --skip-clml --skip-s3 \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch  ${@:3} 


######## clearml ########
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python dist_run.py --nproc_per_node=$GPUS --master_port=$PORT \
#     --clml-run-locally --clml-proj mmdet --clml-task-name example-nos3-task --skip-s3 \
#     $(dirname "$0")/train.py $CONFIG --clearml --launcher pytorch ${@:3} 


######## clearml + s3 bulk dl ########
export AWS_ENDPOINT_URL=https://play.min.io
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export CERT_PATH=

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python dist_run.py --nproc_per_node=$GPUS --master_port=$PORT \
#     --clml-run-locally --clml-proj mmdet --clml-task-name coco_mini_train --download-models 'resnet50_msra-5891d200.pth' --s3-models-bucket mmdet-wts --s3-models-path '' --download-data coco_mini --s3-data-bucket coco --s3-data-path '' \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch --clearml ${@:3} 


######## clearml + data s3 direct ########
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python dist_run.py --nproc_per_node=$GPUS --master_port=$PORT \
    --clml-run-locally --clml-proj mmdet --clml-task-name coco_mini_train_s3_direct --clml-task-type training --download-models 'resnet50_msra-5891d200.pth' --s3-models-bucket mmdet-wts --s3-models-path '' --s3-direct-read --download-data coco_mini/train.json coco_mini/val.json --s3-data-bucket coco --s3-data-path '' \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --clearml ${@:3} 
