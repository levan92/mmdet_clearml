#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

export AWS_ENDPOINT_URL=https://play.min.io
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export CERT_PATH=

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python dist_test_clearml.py --nproc_per_node=$GPUS --master_port=$PORT \
--clml-run-locally --clml-proj mmdet --clml-task-name example-task --download-models 'resnet50_msra-5891d200.pth' --s3-models-bucket mmdet-wts --s3-models-path '' --download-data coco_mini --s3-data-bucket coco --s3-data-path '' \
$(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch \
    --clearml \
    --fuse-conv-bn \
    --write-result \
    --eval \
    ${@:3} 
