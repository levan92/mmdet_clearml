pardir="$(dirname "$PWD")"
tag=2.18.0-torch1.10.0
clearml=~/clearml.conf
cert_path=/usr/local/share/ca-certificates

docker run -it --gpus all -v $clearml:/root/clearml.conf -v $REQUESTS_CA_BUNDLE:$REQUESTS_CA_BUNDLE -v $cert_path:$cert_path -v $pardir:$pardir -w $pardir  --shm-size=8g \
--env AWS_ACCESS_KEY --env AWS_SECRET_ACCESS --env CERT_PATH --env CERT_DL_URL --env REQUESTS_CA_BUNDLE mmdetection:$tag
