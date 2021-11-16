import io
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union

from mmcv.fileio import FileClient, BaseStorageBackend


class Boto3Handler:
    def __init__(
        self,
        endpoint_url,
        aws_access_key_id,
        aws_secret_access_key,
        service_name="s3",
        region_name="us-east-1",
        verify=None,
        bucket=None,
    ):
        import boto3

        if bucket:
            self.bucket = bucket

        self.client = boto3.client(
            service_name=service_name,
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            verify=verify,
        )

    def download_file(self, filepath, bucket=None):
        bucket = bucket or self.bucket
        with io.BytesIO() as f:
            return self._client.download_fileobj(bucket, filepath, f)


@FileClient.register_backend("boto3")
class Boto3Backend(BaseStorageBackend):
    """Boto3 storage backend.

    Boto3Backend supports reading and writing data to an AWS services (s3, ecs, etc.).

    Args:
        path_mapping (dict, optional): Path mapping dict from local path to
            s3 path. When ``path_mapping={'src': 'dst'}``, ``src`` in
            ``filepath`` will be replaced by ``dst``. Default: None.

    Examples:
        >>> filepath1 = 's3://path/of/file'
        >>> client = Boto3Backend()
        >>> client.get(filepath1)  # get data from default cluster
    """

    def __init__(
        self,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        bucket: str,
        region_name: Optional[str] = "us-east-1",
        verify: Optional[str] = None,
        path_mapping: Optional[dict] = None,
    ):
        try:
            self._client = Boto3Handler(
                endpoint_url,
                aws_access_key_id,
                aws_secret_access_key,
                region_name,
                verify,
            )
        except ImportError:
            raise ImportError("Please install boto3 to enable " "Boto3Backend.")

        self.bucket = bucket

        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.

        Args:
            filepath (str): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """
        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r"\\+", "/", filepath)

    def get(self, filepath: Union[str, Path]) -> memoryview:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            memoryview: A memory view of expected bytes object to avoid
                copying. The memoryview object can be converted to bytes by
                ``value_buf.tobytes()``.
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)

        value = self._client.download_file(filepath, self.bucket)
        value_buf = memoryview(value)

        return value_buf

