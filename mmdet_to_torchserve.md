## torchserve for mmdet

- build mmdet-serve docker image
    - `docker build -t mmdet-serve:latest docker/serve/`
- `cd docker`
- `docker build -t fish_inference -f Dockerfile.inference .`
- `./start_docker_inference.sh`
- `cd tools/serve`
- change `score_thr` and `iou_threshold` in config file under model/test_cfg
- Convert model from MMDetection to TorchServe
    - `python3 mmdet2torchserve.py <config_file> <checkpoint_file> --output-folder <output_folder> --model-name <model_name> -f`

## [torchserve for timm](https://github.com/jinmingteo/serve/tree/master/examples/image_classifier/timm_classifier)

- convert pth to torchscript
    - `cd docker`
    - `./start_docker_inference.sh`
    - `cd tools/timm`
    - `python3 torchscript_model_converter.py --model <model_arch> --checkpoint <checkpoint_file> --num-classes <num_classes> --output <output_pt_file>`
- build serve docker image
    - `cd tools/timm`
    - `./build_image.sh -bt dev -g -cv cu101` for GPU based developer image with cuda version 10.2
- convert to torchserve
    - `cd docker`
    - `./start_docker_inference.sh`
    - change classnames in `index_to_name.json`
    - `torch-model-archiver --model-name <model_name> --version <version> --serialized-file <pt_file> --extra-files <path_to_index_to_name_json> --handler <handler_file> --export-path <output_folder> --force`

## Inference

- `docker-compose -f docker-compose.yml up`
