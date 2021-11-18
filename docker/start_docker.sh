pardir="$(dirname "$PWD")"
tag=2.18.0-torch1.10.0
clearml=~/clearml.conf

docker run -it --gpus all -v $clearml:/root/clearml.conf -v $pardir:$pardir -w $pardir  --shm-size=8g mmdetection:$tag 