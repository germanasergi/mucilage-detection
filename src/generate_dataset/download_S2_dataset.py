# src/download_sentinel_data.py

import os
import yaml
import sys
import pandas as pd
import boto3
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from eopf.common.constants import OpeningMode
from eopf.common.file_utils import AnyPath
from eopf.store.convert import convert
import shutil
import glob
from datetime import datetime
from sklearn.utils import shuffle

# Import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auth.auth import S3Connector
from utils.cdse_utils import (download_bands, extract_s3_path_from_url, create_rgb_image)

#os.environ["ECCODES_DISABLE_WARNINGS"] = "1"

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_environment(config):
    """Set up environment variables and directories for the dataset"""
    # Keep these from environment variables
    ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")

    # Get other parameters from config
    ENDPOINT_URL = config['endpoint_url']
    BUCKET_NAME = config['bucket_name']
    DATASET_VERSION = config['dataset_version']
    BASE_DIR = config['base_dir']
    DATASET_DIR = f"{BASE_DIR}/{DATASET_VERSION}"
    BANDS = config['bands']

    # Setup connector
    connector = S3Connector(
        endpoint_url=ENDPOINT_URL,
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        region_name='default')

    s3 = connector.get_s3_resource()
    s3_client = connector.get_s3_client()

    return {
        'ACCESS_KEY_ID': ACCESS_KEY_ID,
        'SECRET_ACCESS_KEY': SECRET_ACCESS_KEY,
        'ENDPOINT_URL': ENDPOINT_URL,
        'BUCKET_NAME': BUCKET_NAME,
        'DATASET_DIR': DATASET_DIR,
        'BANDS': BANDS,
        's3_client': connector.get_s3_client(),
        's3': connector.get_s3_resource()
    }

def setup_logger(log_path, filename_prefix):
    """Setup logger with specified path and prefix"""
    log_filename = f"{log_path}/{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.remove()
    logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")
    return log_filename



def prepare_paths(path_dir):
    logger.info(f"Preparing paths from directory: {path_dir}")

    #df_input = pd.read_csv(f"{path_dir}/input_l1c.csv")
    df_output = pd.read_csv(f"{path_dir}/output_l2a.csv")

    logger.info(f"Paths prepared: {len(df_output)} output files")
    return df_output



def download_sentinel_data(df_output, base_dir, access_key, secret_key, endpoint_url):
    """Download Sentinel data from S3 to local directories"""

    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Created local directories: {input_dir}, {output_dir}")

    S3_CONFIG = {
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {
            "endpoint_url": endpoint_url,
            "region_name": "default"
        }
    }

    target_store_config = dict(mode=OpeningMode.CREATE_OVERWRITE)

    logger.info("Starting target file downloads...")
    for _, row in tqdm(df_output.iterrows(), total=len(df_output), desc="Target files"):
        try:
            product_url_base = row['S3Path']
            product_url = "s3://" + product_url_base.lstrip("/")
            zarr_filename = os.path.basename(product_url).replace('.SAFE', '.zarr')
            zarr_path = os.path.join(output_dir, zarr_filename)

            # Skip if file already exists
            if os.path.exists(zarr_path):
                logger.info(f"Skipping download, .zarr already exists: {zarr_path}")
                continue
            
            logger.info(f"Downloading target: {product_url} -> {zarr_path}")
            convert(AnyPath(product_url, **S3_CONFIG), zarr_path, target_store_kwargs=target_store_config)
        except Exception as e:
            logger.error(f"Error downloading {product_url if product_url else 'unknown URL'}: {e}")
        finally:
            # Cleanup SAFE temp dirs left in /tmp
            tmp_safes = glob.glob("/tmp/*.SAFE")
            for safe_dir in tmp_safes:
                try:
                    shutil.rmtree(safe_dir, ignore_errors=True)
                    logger.info(f"Cleaned up temporary SAFE: {safe_dir}")
                except Exception as ce:
                    logger.warning(f"Could not clean temp SAFE {safe_dir}: {ce}")


        
def main():
    # Load environment variables
    load_dotenv()

    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Download Sentinel data based on provided config and CSV files')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    #parser.add_argument('--l1c-csv', type=str, required=True, help='Path to L1C CSV file')
    parser.add_argument('--l2a-csv', type=str, required=True, help='Path to L2A CSV file')
    parser.add_argument('--create-rgb', action='store_true', help='Optionally create RGB images after download')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup environment
    env = setup_environment(config)
    df_output = prepare_paths(env['DATASET_DIR'])

    # Setup logger
    setup_logger(env['DATASET_DIR'], "sentinel_download_log")

    logger.info("Starting download process...")
    download_sentinel_data(
        df_output,
        env['DATASET_DIR'],
        env['ACCESS_KEY_ID'],
        env['SECRET_ACCESS_KEY'],
        env['ENDPOINT_URL']
    )

    logger.success("All downloads completed.")

if __name__ == "__main__":
    main()