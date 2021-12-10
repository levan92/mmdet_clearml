"""
S3 downloading and uploading
"""
import os
from pathlib import Path
import tarfile
import zipfile
from warnings import warn

import boto3
from botocore.client import Config


def master_unzip(local_fp):
    local_fp = Path(local_fp)
    if local_fp.suffix in [".tar", ".gz", ".tgz"]:
        print("Untarring..")
        with tarfile.open(local_fp) as tar:
            tar.extractall(local_fp.parent)
    elif local_fp.suffix in [".zip"]:
        print("Unzipping..")
        with zipfile.ZipFile(local_fp, "r") as zip_ref:
            zip_ref.extractall(local_fp.parent)


def download_dir_from_s3(
    s3_resource, bucket_name, remote_dir_name, local_dir, unzip=True
):
    buck = s3_resource.Bucket(bucket_name)
    for obj in buck.objects.filter(Prefix=str(remote_dir_name) + "/"):
        remote_rel_path = Path(obj.key).relative_to(remote_dir_name)
        if str(remote_rel_path) == ".":
            continue
        local_fp = local_dir / remote_rel_path
        local_fp.parent.mkdir(parents=True, exist_ok=True)
        if obj.key[-1] == "/":
            download_dir_from_s3(
                s3_resource, bucket_name, obj.key, local_fp, unzip=unzip
            )
            continue
        if not local_fp.is_file():
            print(f"Downloading {obj.key} from S3 to {local_fp}..")
            buck.download_file(obj.key, str(local_fp))
            if unzip:
                master_unzip(local_fp)


def upload_dir_to_s3(s3_resource, bucket_name, local_dir, remote_dir):
    buck = s3_resource.Bucket(bucket_name)

    for root, dirs, files in os.walk(str(local_dir)):
        for file in files:
            local_fp = Path(root) / file
            rel_path = Path(root).relative_to(local_dir)
            remote_fp = Path(remote_dir) / rel_path / file
            print(f"Uploading {local_fp} to S3 {remote_dir}")
            buck.upload_file(str(local_fp), str(remote_fp))


class S3_handler:
    def __init__(
        self,
        endpoint_url,
        aws_access_key_id,
        aws_secret_access_key,
        cert_path,
        local_weight_dir=None,
        local_data_dir=None,
    ):
        assert endpoint_url,'Endpoint URL is empty'
        self.s3_resource = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
            verify=cert_path,
        )

        self.local_weight_dir = local_weight_dir or "weights"
        self.local_weight_dir = Path(self.local_weight_dir)
        self.local_weight_dir.mkdir(parents=True, exist_ok=True)

        self.local_data_dir = local_data_dir or "datasets"
        self.local_data_dir = Path(self.local_data_dir)
        self.local_data_dir.mkdir(parents=True, exist_ok=True)

    def dl_files(self, files, s3_bucket, s3_parent_path, local_parent, unzip=True):
        local_dled_files = []
        if files:
            assert s3_bucket
            buck = self.s3_resource.Bucket(s3_bucket)
            s3_parent_path = Path(s3_parent_path)
            assert local_parent
            local_parent = Path(local_parent)
            for file in files:
                s3_file_path = s3_parent_path / file
                local_file_path = local_parent / file
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                print(
                    f"Downloading from {s3_file_path} from bucket {s3_bucket} to {local_file_path}.."
                )
                buck.download_file(str(s3_file_path), str(local_file_path))
                assert local_file_path.is_file()
                local_dled_files.append(local_file_path)
                print(f"File: {file} downloaded from S3 & stored at {local_file_path}!")
                if unzip:
                    master_unzip(local_file_path)
        return local_dled_files

    def dl_dirs(self, dirs, s3_bucket, s3_parent_path, local_parent, unzip=True):
        local_dled_dirs = []
        if dirs:
            assert s3_bucket
            s3_parent_path = Path(s3_parent_path)
            assert local_parent
            local_parent = Path(local_parent)
            for folder in dirs:
                local_folder_path = local_parent / folder
                s3_dataset_path = s3_parent_path / folder
                download_dir_from_s3(
                    self.s3_resource,
                    s3_bucket,
                    s3_dataset_path,
                    local_folder_path,
                    unzip=unzip,
                )
                local_dled_dirs.append(local_folder_path)
        else:
            warn('No dirs downloading')
        return local_dled_dirs

    def ul_dir(self, local_dir, s3_bucket, s3_parent_path, s3_dir_name):
        s3_output_path = Path(s3_parent_path) / f"{s3_dir_name}"
        upload_dir_to_s3(self.s3_resource, s3_bucket, local_dir, s3_output_path)
