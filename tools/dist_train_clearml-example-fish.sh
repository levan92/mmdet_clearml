#!/usr/bin/env bash

export CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

# for FISH: Merging videos in DOTA format
export TRAIN_FOLDERS=collated
export VAL_FOLDERS=collated
export TEST_FOLDERS=collated

## coco_mode=0 for normal trg, 1 for vehicle training, 2 for split_classes_dict
export COCO_MODE=1
## coco_mode=2, map small-vehicle and large-vehicle into vehicle class
# export split_keys=vehicle,helicopter,plane
# export vehicle=small-vehicle,large-vehicle
######## local ########
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python dist_train_clearml.py --nproc_per_node=$GPUS --master_port=$PORT \
#     --skip-clml --skip-s3 \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch  ${@:3} 


######## clearml ########
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python dist_train_clearml.py --nproc_per_node=$GPUS --master_port=$PORT \
#     --clml-run-locally --clml-proj mmdet --clml-task-name example-nos3-task --skip-s3 \
#     $(dirname "$0")/train.py $CONFIG --clearml --launcher pytorch ${@:3} 


######## clearml + s3 bulk dl ########
export AWS_ENDPOINT_URL=https://play.min.io
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS
export CERT_PATH=$CERT_PATH
export CERT_DL_URL=$CERT_DL_URL
DOCKER_IMG='mmdet:latest'
S3_MODEL_BUCKET='mmdet-test'
S3_MODEL_PATH=
S3_DATA_BUCKET=$S3_MODEL_BUCKET
S3_DATA_PATH='raw_videos'

# Experimenting
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export FIND_UNUSED_PARAMETERS=True

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python dist_run.py --nproc_per_node=$GPUS --master_port=$PORT \
   --clml-run-locally --clml-proj mmdet --clml-task-name example-task --docker-img $DOCKER_IMG \
   --download-models 'yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth' \
   --s3-models-bucket $S3_MODEL_BUCKET --merge-videos --download-data collated.tar.gz --s3-data-bucket $S3_DATA_BUCKET --s3-data-path  $S3_DATA_PATH \
   $(dirname "$0")/train.py $CONFIG --launcher pytorch --clearml ${@:3}


######## clearml + s3 direct ########
# export AWS_ENDPOINT_URL=https://play.min.io
# export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
# export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
# export CERT_PATH=

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python dist_run.py --nproc_per_node=$GPUS --master_port=$PORT \
#     --clml-run-locally --clml-proj mmdet --clml-task-name example-task --download-models 'resnet50_caffe-788b5fa3.pth' --s3-models-bucket mmdet-wts --s3-models-path '' --download-data coco_mini --s3-data-bucket coco --s3-data-path '' \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch --clearml ${@:3} 
