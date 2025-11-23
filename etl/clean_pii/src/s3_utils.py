from pathlib import Path
from time import sleep
import boto3

def _create_s3_client():
    return boto3.client("s3")

def _parse_s3_path(s3_path: str):
    s3_path_parts = Path(s3_path).parts
    s3_bucket = s3_path_parts[1]
    s3_path_dir = "/".join(s3_path_parts[2:])
    return s3_bucket, s3_path_dir


def list_files(s3_path: str):
    s3_client = _create_s3_client()
    s3_bucket, s3_path_dir = _parse_s3_path(s3_path)
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_path_dir)

    # Iterate through the objects and print their keys
    if 'Contents' in response:
        for obj in response['Contents']:
            yield f"s3://{s3_bucket}/{obj['Key']}"

    # Handle pagination for large buckets
    while response.get('IsTruncated'):
        response = s3_client.list_objects_v2(
            Bucket=s3_bucket,
            Prefix=s3_path_dir, Delimiter="/",
            ContinuationToken=response['NextContinuationToken']
        )
        if 'Contents' in response:
            for obj in response['Contents']:
                yield f"s3://{s3_bucket}/{obj['Key']}"


def download_file(s3_path: str, local_path: Path):
    s3_bucket, s3_file_path = _parse_s3_path(s3_path)
    s3 = boto3.client('s3')
    with open(local_path, 'wb') as f:
        s3.download_fileobj(s3_bucket, s3_file_path, f)
    return local_path