# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
from pathlib import Path

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector

import mmcv_custom.fileio.file_client
import mmdet_custom.datasets.coco

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--clearml",
        action="store_true",
        help="whether not to include clearml logging",
    )
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--write-result", action="store_true", help="write inference results into file"
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument(
        "--sub-eval",
        type=str,
        help="Only applies for bbox eval. Give a folder of coco jsons which are subsets to the original test set. Will run evaluation on each subset json.",
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )
    parser.add_argument(
        "--show-score-thr",
        type=float,
        default=0.3,
        help="score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.write_result or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--write-result", "--show" or "--show-dir"'
    )

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get("neck"):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get("rfp_backbone"):
            if cfg.model.neck.rfp_backbone.get("pretrained"):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    write_res_prefix = None
    if rank == 0:
        work_dir = args.work_dir if args.work_dir is not None else cfg.work_dir
        if work_dir is not None:
            mmcv.mkdir_or_exist(osp.abspath(work_dir))
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            json_file = osp.join(work_dir, f"eval_{timestamp}.json")
            write_res_prefix = osp.join(work_dir, f"res_{timestamp}")

    if args.clearml and (not distributed or rank == 0):
        from clearml import Task

        cl_task = Task.current_task()
        cl_task.connect_configuration(name="config", configuration=dict(cfg))

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model, data_loader, args.show, args.show_dir, args.show_score_thr
        )
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options

        if args.write_result:
            res_files, _ = dataset.format_results(
                outputs, jsonfile_prefix=write_res_prefix, **kwargs
            )
            if args.clearml:
                for key, path in res_files.items():
                    cl_task.upload_artifact(
                        name=f"res-{key}",
                        artifact_object=path,
                    )
        else:
            res_files = None

        if args.eval:
            # Overall Evaluation
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs.copy(), **eval_kwargs)
            print(metric)

            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None:
                mmcv.dump(metric_dict, json_file)

            if args.clearml:
                cl_task.upload_artifact(
                    name="test",
                    artifact_object=metric,
                )
                cl_logger = cl_task.get_logger()
                for key, value in metric.items():
                    if "_copypaste" in key:
                        continue
                    cl_logger.report_scalar(
                        title="test",
                        series=key,
                        value=value,
                        iteration=0,
                    )

            if args.sub_eval:
                # Subset evaluation
                jsons = []
                val_str = "val"
                if isinstance(args.sub_eval, list):
                    jsons = args.sub_eval
                elif isinstance(args.sub_eval, str):
                    sub_eval_path = Path(args.sub_eval)
                    if sub_eval_path.is_file():
                        jsons.append(args.sub_eval)
                    elif sub_eval_path.is_dir():
                        jsons = [
                            json_path
                            for json_path in sub_eval_path.glob(f"{val_str}*.json")
                        ]
                    else:
                        raise OSError("Given sub eval file/folder not found")
                if len(jsons) == 0:
                    warnings.warn("No valid json for sub-eval found/given")
                else:
                    if res_files is None:
                        res_files, tmp_dir = dataset.format_results(outputs, None)
                    else:
                        tmp_dir = None

                    for json_path in jsons:
                        val_set_name = json_path.stem
                        print()
                        print(f"Evaluating {val_set_name}")
                        test_data_cfg = cfg.data.test.copy()
                        test_data_cfg.type = "CocoDatasetImba"
                        test_data_cfg.ann_file = str(json_path)
                        subdataset = build_dataset(test_data_cfg)
                        metric = subdataset.evaluate_imba(res_files, **eval_kwargs)

                        if args.clearml:
                            cl_task.upload_artifact(
                                name=val_set_name,
                                artifact_object=metric,
                            )
                            cl_logger = cl_task.get_logger()
                            for key, value in metric.items():
                                if "_copypaste" in key:
                                    continue
                                cl_logger.report_scalar(
                                    title=val_set_name,
                                    series=key,
                                    value=value,
                                    iteration=0,
                                )

                    if tmp_dir is not None:
                        tmp_dir.cleanup()


if __name__ == "__main__":
    main()
