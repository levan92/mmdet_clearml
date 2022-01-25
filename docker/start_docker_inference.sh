WORKSPACE=/media/data/mmdet_clearml
DATA=/media/data/downloads

docker run -it \
    --gpus all \
    -w $WORKSPACE \
    -v $WORKSPACE:$WORKSPACE \
    -v $DATA:$DATA \
    fish_inference
