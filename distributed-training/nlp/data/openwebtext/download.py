import os
import boto3
import argparse

s3 = boto3.client("s3")
parser = argparse.ArgumentParser()
parser.add_argument("--bucket", type=str, default=None)
args = parser.parse_args()

BUCKET = "open-web-text"
if args.bucket is not None:
    BUCKET = args.bucket

PROCESSED_S3_URI = "processed"  # <- path inside S3 bucket

for split in ["train", "val"]:
    print(f"Downloading {split}.bin . . . . ")
    out_filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
    s3.download_file(BUCKET, f"{PROCESSED_S3_URI}/{split}.bin", out_filename)
