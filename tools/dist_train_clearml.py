from clearml import Task

from pathlib import Path
import os

from torch.distributed.run import get_args_parser

S3_ENVS = [
    "AWS_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "CERT_PATH",
    "CERT_DL_URL",
]


def get_default_parser():
    parser = get_args_parser()
    parser.add_argument(
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local_rank as argument, and will instead set LOCAL_RANK.",
    )
    return parser


def add_clearml_args(parser):
    clearml_parser = parser.add_argument_group("ClearML Args")

    clearml_parser.add_argument(
        "--skip-clml",
        help="flag to entirely skip any clearml action.",
        action="store_true",
    )
    clearml_parser.add_argument(
        "--clml-run-locally",
        help="flag to run job locally but keep clearml expt tracking.",
        action="store_true",
    )
    clearml_parser.add_argument(
        "--clml-proj", default="mmdet", help="ClearML Project Name"
    )
    clearml_parser.add_argument(
        "--clml-task-name", default="Task", help="ClearML Task Name"
    )
    clearml_parser.add_argument(
        "--clml-task-type",
        default="data_processing",
        help="ClearML Task Type, e.g. training, testing, inference, etc",
        choices=[
            "training",
            "testing",
            "inference",
            "data_processing",
            "application",
            "monitor",
            "controller",
            "optimizer",
            "service",
            "qc",
            "custom",
        ],
    )
    clearml_parser.add_argument(
        "--clml-output-uri",
        help="ClearML output uri",
    )
    clearml_parser.add_argument(
        "--docker-img",
        help="Base docker image to pull",
    )
    clearml_parser.add_argument("--queue", default="1gpu", help="ClearML Queue")


def add_s3_args(parser):
    s3_parser = parser.add_argument_group("S3 Args")

    s3_parser.add_argument(
        "--skip-s3", help="flag to entirely skip any s3 action.", action="store_true"
    )

    ## MODELS
    s3_parser.add_argument(
        "--download-models", help="List of models to download", nargs="+"
    )
    s3_parser.add_argument("--s3-models-bucket", help="S3 Bucket for models")
    s3_parser.add_argument("--s3-models-path", help="S3 Models Path", default='')

    ## DATA
    s3_parser.add_argument(
        "--download-data", help="List of dataset to download", nargs="+"
    )
    s3_parser.add_argument("--s3-data-bucket", help="S3 Bucket for data")
    s3_parser.add_argument("--s3-data-path", help="S3 Data Path", default='')
    parser.add_argument(
        "--s3-direct-read",
        help="direct reading of images from S3 bucket without initial bulk download.",
        action="store_true",
    )



def init_clearml(args, environs={}):
    if not args.skip_clml:
        cl_task = Task.init(
            project_name=args.clml_proj,
            task_name=args.clml_task_name,
            task_type=args.clml_task_type,
            output_uri=args.clml_output_uri,
        )
        if args.docker_img:
            env_strs = " ".join([f"--env {k}={v}" for k, v in environs.items()])
            cl_task.set_base_docker(
                f"{args.docker_img} --env GIT_SSL_NO_VERIFY=true {env_strs}"
            )
        if not args.clml_run_locally:
            cl_task.execute_remotely(queue_name=args.queue, exit_process=True)


def s3_download(
    args,
    environs={},
    local_weight_dir="weights",
    local_data_dir="datasets",
):
    if not args.skip_s3:
        from utils.s3_helper import S3_handler

        environs["CERT_PATH"] = environs["CERT_PATH"] if environs["CERT_PATH"] else None
        if (
            environs["CERT_DL_URL"]
            and environs["CERT_PATH"]
            and not Path(environs["CERT_PATH"]).is_file()
        ):
            import utils.wget as wget
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            print(f'Downloading from {environs["CERT_DL_URL"]}')
            wget.download(environs["CERT_DL_URL"])
            environs["CERT_PATH"] = Path(environs["CERT_DL_URL"]).name

        s3_handler = S3_handler(
            environs["AWS_ENDPOINT_URL"],
            environs["AWS_ACCESS_KEY_ID"],
            environs["AWS_SECRET_ACCESS_KEY"],
            environs["CERT_PATH"],
        )

        if args.download_models:
            local_weights_paths = s3_handler.dl_files(
                args.download_models,
                args.s3_models_bucket,
                args.s3_models_path,
                local_weight_dir,
                unzip=True,
            )

        if args.download_data:
            if args.s3_direct_read:
                local_data_dirs = s3_handler.dl_files(
                    args.download_data,
                    args.s3_data_bucket,
                    args.s3_data_path,
                    local_data_dir,
                    unzip=True,
                )
            else:
                local_data_dirs = s3_handler.dl_dirs(
                    args.download_data,
                    args.s3_data_bucket,
                    args.s3_data_path,
                    local_data_dir,
                    unzip=True,
                )


def main(args=None):
    parser = get_default_parser()
    add_clearml_args(parser)
    add_s3_args(parser)
    args = parser.parse_args(args)

    environs = {var: os.environ.get(var) for var in S3_ENVS}
    init_clearml(args, environs=environs)
    s3_download(args, environs=environs)

    from torchrun import run

    run(args)


if __name__ == "__main__":
    main()
