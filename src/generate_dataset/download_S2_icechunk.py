import os
import argparse
import yaml
import glob
import shutil
import sys
from datetime import datetime
from loguru import logger
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

from upath import UPath
import s3fs
from s3fs import S3FileSystem

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.icechunk_utils import safe_to_icechunk

# ----------------------
# CONFIG & ENVIRONMENT
# ----------------------

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def setup_environment(config):
    ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

    ENDPOINT_URL = config["endpoint_url"]
    BUCKET_NAME = config["bucket_name"]
    DATASET_VERSION = config["dataset_version"]
    BASE_DIR = config["base_dir"]
    DATASET_DIR = f"{BASE_DIR}/{DATASET_VERSION}"
    BANDS = config["bands"]

    s3_fs = S3FileSystem(
        key=ACCESS_KEY_ID,
        secret=SECRET_ACCESS_KEY,
        client_kwargs={"endpoint_url": ENDPOINT_URL, "region_name": "default"},
    )

    return {
        "ACCESS_KEY_ID": ACCESS_KEY_ID,
        "SECRET_ACCESS_KEY": SECRET_ACCESS_KEY,
        "ENDPOINT_URL": ENDPOINT_URL,
        "BUCKET_NAME": BUCKET_NAME,
        "DATASET_DIR": DATASET_DIR,
        "BANDS": BANDS,
        "s3_fs": s3_fs,
    }

def setup_logger(log_path, filename_prefix):
    log_filename = f"{log_path}/{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.remove()
    logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")
    return log_filename

def prepare_paths(path_dir):
    logger.info(f"Preparing paths from directory: {path_dir}")
    df_output = pd.read_csv(f"{path_dir}/output_l2a.csv")
    logger.info(f"Paths prepared: {len(df_output)} output files")
    return df_output


def convert_safe_to_icechunk(product_url, repo_url, fs, bands):
    """
    Convert a SAFE product on S3 into an Icechunk repo on S3.
    """
    # Open a new Icechunk repo at the target location
    repo = Repository.create(repo_url, storage_options={"anon": False, "client": fs})
    logger.info(f"Created Icechunk repo: {repo_url}")

    # Convert SAFE -> Zarr -> Icechunk (abstracted in a utility)
    write_zarr_from_safe(product_url, repo, bands=bands)

    logger.success(f"Finished writing to {repo_url}")
    repo.close()


def download_sentinel_data(df_output, base_dir, s3_fs, bucket_name, bands):
    output_dir = os.path.join(base_dir, "target")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Created local directories: {output_dir}")

    for _, row in tqdm(df_output.iterrows(), total=len(df_output), desc="Target files"):
        try:
            product_url_base = row["S3Path"]   # e.g. /eodata/...SAFE
            product_url = "s3://" + product_url_base.lstrip("/")
            icechunk_repo_name = os.path.basename(product_url).replace(".SAFE", ".icechunk")
            repo_url = f"s3://{bucket_name}/sentinel2/{icechunk_repo_name}"
            logger.info(f"Converting SAFE -> Icechunk: {product_url} -> {repo_url}")
            convert_safe_to_icechunk(product_url, repo_url, s3_fs, bands)

        except Exception as e:
            logger.error(f"Error converting {product_url if product_url else 'unknown'}: {e}")
        finally:
            # Cleanup tmp .SAFE directories if any
            tmp_safes = glob.glob("/tmp/*.SAFE")
            for safe_dir in tmp_safes:
                shutil.rmtree(safe_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary SAFE: {safe_dir}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Download Sentinel data as Icechunk repos")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--l2a-csv", type=str, required=True, help="Path to L2A CSV file")
    args = parser.parse_args()

    config = load_config(args.config)
    env = setup_environment(config)
    df = prepare_paths(env["DATASET_DIR"])
    bands = env["BANDS"]
    setup_logger(env["DATASET_DIR"], "sentinel_download_log")

    logger.info("Starting SAFE -> Icechunk conversion...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing SAFEs"):
        s3_url = row['S3Path']  # or build full S3 URL if needed
        try:
            safe_to_icechunk(s3_url, repo_path, bands=bands)
        except Exception as e:
            print(f"Failed to process {s3_url}: {e}")


if __name__ == "__main__":
    main()