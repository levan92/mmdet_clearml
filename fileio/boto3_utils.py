def get_s3(s3_info):
    if s3_info is None:
        return None
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_info["endpoint_url"],
        aws_access_key_id=s3_info["aws_access_key_id"],
        aws_secret_access_key=s3_info["aws_secret_access_key"],
        region_name=s3_info["region_name"],
        verify=s3_info["verify"],
    )
    return s3
