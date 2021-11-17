#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}


######## local ########
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python dist_train_clearml.py --nproc_per_node=$GPUS --master_port=$PORT \
#     --skip-clml --skip-s3 \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch  ${@:3} 


######## clearml ########
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python dist_train_clearml.py --nproc_per_node=$GPUS --master_port=$PORT \
    --clml-run-locally --clml-proj mmdet --clml-task-name example-nos3-task --skip-s3 \
    $(dirname "$0")/train.py $CONFIG --clearml --launcher pytorch ${@:3} 


######## clearml + s3 bulk dl ########
# export AWS_ENDPOINT_URL=https://play.min.io
# export AWS_ACCESS_KEY=$AWS_ACCESS_KEY
# export AWS_SECRET_ACCESS=$AWS_SECRET_ACCESS
# export CERT_PATH=

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python dist_train_clearml.py --nproc_per_node=$GPUS --master_port=$PORT \
#     --clml-run-locally --clml-proj mmdet --clml-task-name example-task --download-models 'resnet50_msra-5891d200.pth' --s3-models-bucket mmdet-wts --s3-models-path '' --download-data coco_mini --s3-data-bucket coco --s3-data-path '' \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch --clearml ${@:3} 


######## clearml + s3 direct ########
# export AWS_ENDPOINT_URL=https://play.min.io
# export AWS_ACCESS_KEY=$AWS_ACCESS_KEY
# export AWS_SECRET_ACCESS=$AWS_SECRET_ACCESS
# export CERT_PATH=

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python dist_train_clearml.py --nproc_per_node=$GPUS --master_port=$PORT \
#     --clml-run-locally --clml-proj mmdet --clml-task-name example-task --download-models 'resnet50_caffe-788b5fa3.pth' --s3-models-bucket mmdet-wts --s3-models-path '' --download-data coco_mini --s3-data-bucket coco --s3-data-path '' \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch --clearml ${@:3} 
