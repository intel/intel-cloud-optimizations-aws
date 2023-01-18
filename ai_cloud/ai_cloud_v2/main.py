"""entrypoint"""
import argparse
import os
import sys

from model import Model
from server import serve


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", type=str, default="disk", choices=["disk", "s3"])
    parser.add_argument("kind", type=str, default="infer", choices=["train", "infer"])
    args = parser.parse_args()
    if args.backend == "s3":
        try:
            os.environ.get("MODEL_BUCKET_NAME")
        except KeyError:
            sys.exit("bucket name for s3 should be specified as ane env variable.")
    return args


def main():
    args = cli()
    bucket = None
    if args.backend == "s3":
        bucket = os.environ.get("MODEL_BUCKET_NAME")
    clf = Model(bucket=bucket)
    if args.kind == "train":
        clf.train()
        clf.save()
    elif args.kind == "infer":
        serve()


if __name__ == "__main__":
    main()
