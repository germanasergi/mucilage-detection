import os
import time
import yaml
import shutil
import sys
from datetime import datetime, timedelta
import pandas as pd
import requests
from loguru import logger
from dotenv import load_dotenv
from shapely.geometry import shape, Polygon

# Load environment variables
load_dotenv()

# Import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import remove_last_segment_rsplit
from utils.cdse_utils import (create_cdse_query_url, download_bands)
from auth.auth import S3Connector

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def save_config_copy(config, config_path, dataset_dir):
    """Save a copy of the config file to the dataset directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_filename = os.path.basename(config_path)
    config_name, config_ext = os.path.splitext(config_filename)
    target_path = os.path.join(dataset_dir, f"{config_name}_{timestamp}{config_ext}")

    # Save a copy of the config to the dataset directory
    with open(target_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    logger.info(f"Saved configuration copy to {target_path}")
    return target_path


def setup_environment(config):
    """Set up environment variables and directories for the dataset"""
    # Keep these from environment variables
    ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")
    PAT = os.environ.get("PAT")

    # Get other parameters from config
    ENDPOINT_URL = config['endpoint_url']
    ENDPOINT_STAC = config['endpoint_stac']
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
    bucket = s3.Bucket(BUCKET_NAME)

    return {
        'ENDPOINT_URL': ENDPOINT_URL,
        'ENDPOINT_STAC': ENDPOINT_STAC,
        'PAT': PAT,
        'BUCKET_NAME': BUCKET_NAME,
        'DATASET_VERSION': DATASET_VERSION,
        'BASE_DIR': BASE_DIR,
        'DATASET_DIR': DATASET_DIR,
        'BANDS': BANDS,
        's3': s3,
        's3_client': s3_client,
        'bucket': bucket
    }


def setup_logger(log_path, filename_prefix):
    """Setup logger with specified path and prefix"""
    log_filename = f"{log_path}/{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.remove()
    logger.add(log_filename, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="{message}")
    return log_filename


def query_sentinel_data(bbox, start_date, end_date, max_items, max_cloud_cover, num_days=10):
    """Query Sentinel data for the specified parameters"""
    # Generate the polygon string from bbox [minx, miny, maxx, maxy]
    polygon = f"POLYGON (({bbox[0]} {bbox[1]}, {bbox[0]} {bbox[3]}, {bbox[2]} {bbox[3]}, {bbox[2]} {bbox[1]}, {bbox[0]} {bbox[1]}))"

    # Initialize empty lists to store all results
    all_l1c_results = []
    all_l2a_results = []

    # Loop through the date range with a step of 10 days
    current_date = start_date
    while current_date < end_date:
        # Calculate the end of the current interval
        next_date = min(current_date + timedelta(days=num_days), end_date)

        # Format the dates for the OData query
        start_interval = f"{current_date.strftime('%Y-%m-%dT00:00:00.000Z')}"
        end_interval = f"{next_date.strftime('%Y-%m-%dT23:59:59.999Z')}"

        date_interval = f"{current_date.strftime('%Y-%m-%d')}/{next_date.strftime('%Y-%m-%d')}"

        try:
            # Query for L2A products
            l2a_query_url = create_cdse_query_url(
                collection_name="SENTINEL-3", #SENTINEL-2
                product_type="SL_2_WST__", #MSIL2A
                polygon=polygon,
                start_interval=start_interval,
                end_interval=end_interval,
                #max_cloud_cover=max_cloud_cover,
                max_items=max_items,
                orderby="ContentDate/Start"
            )
            l2a_json = requests.get(l2a_query_url).json()
            l2a_results = l2a_json.get('value', [])

            # Add interval metadata
            for item in l2a_results:
                item['query_interval'] = date_interval

            # # Query for L1C products
            # l1c_query_url = create_cdse_query_url(
            #     product_type="MSIL1C",
            #     polygon=polygon,
            #     start_interval=start_interval,
            #     end_interval=end_interval,
            #     max_cloud_cover=max_cloud_cover,
            #     max_items=max_items,
            #     orderby="ContentDate/Start"
            # )
            # l1c_json = requests.get(l1c_query_url).json()
            # l1c_results = l1c_json.get('value', [])

            # # Add interval metadata
            # for item in l1c_results:
            #     item['query_interval'] = date_interval

            # Log counts
            # l1c_count = len(l1c_results)
            l2a_count = len(l2a_results)

            # if l1c_count != l2a_count:
            #     logger.warning(f"Mismatch in counts for {date_interval}: L1C={l1c_count}, L2A={l2a_count}")

            # # Append results
            # all_l1c_results.extend(l1c_results)
            all_l2a_results.extend(l2a_results)

            # logger.info(f"L1C Items for {date_interval}: {l1c_count}")
            logger.info(f"L2A Items for {date_interval}: {l2a_count}")
            logger.info("####")

        except Exception as e:
            logger.error(f"Error processing interval {date_interval}: {str(e)}")

        # Move to the next interval
        current_date = next_date

    return all_l2a_results


def queries_curation(all_l2a_results):
    """Process and align L1C and L2A data to ensure they match"""
    # Create DataFrames
    # df_l1c = pd.DataFrame(all_l1c_results)
    df_l2a = pd.DataFrame(all_l2a_results)

    # Select required columns
    df_l2a = df_l2a[["Name", "S3Path", "Footprint", "GeoFootprint", "Attributes"]]
    # df_l1c = df_l1c[["Name", "S3Path", "Footprint", "GeoFootprint", "Attributes"]]

    # Extract cloud cover
    # df_l1c['cloud_cover'] = df_l1c['Attributes'].apply(lambda x: x[2]["Value"])
    df_l2a['cloud_cover'] = df_l2a['Attributes'].apply(lambda x: x[2]["Value"])
    # Drop the Attributes column (note: inplace=True needed or need to reassign)
    # df_l1c = df_l1c.drop(columns=['Attributes'], axis=1)
    df_l2a = df_l2a.drop(columns=['Attributes'], axis=1)
    # Create id_key for matching
    df_l2a['id_key'] = df_l2a['Name'].apply(remove_last_segment_rsplit)
    df_l2a['id_key'] = df_l2a['id_key'].str.replace('MSIL2A_', 'MSIL1C_')
    # df_l1c['id_key'] = df_l1c['Name'].apply(remove_last_segment_rsplit)

    # Remove duplicates
    df_l2a = df_l2a.drop_duplicates(subset='id_key', keep='first')
    # df_l1c = df_l1c.drop_duplicates(subset='id_key', keep='first')

    # Align both datasets
    # df_l2a = df_l2a[df_l2a['id_key'].isin(df_l1c['id_key'])]
    # df_l1c = df_l1c[df_l1c['id_key'].isin(df_l2a['id_key'])]

    # Make sure the order is the same
    df_l2a = df_l2a.set_index('id_key')
    # df_l1c = df_l1c.set_index('id_key')

    # df_l2a = df_l2a.loc[df_l1c.index].reset_index()
    # df_l1c = df_l1c.reset_index()

    return df_l2a


def validate_data_alignment(df_l1c, df_l2a):
    """Validate that the data is properly aligned"""
    mismatches = 0
    for i in range(min(len(df_l1c), len(df_l2a))):
        if df_l1c['id_key'][i] != df_l2a['id_key'][i]:
            logger.error(f"Mismatch: {df_l1c['id_key'][i]} != {df_l2a['id_key'][i]}")
            mismatches += 1

    if mismatches == 0:
        logger.info(f"All {len(df_l1c)} records are properly aligned")
    else:
        logger.warning(f"Found {mismatches} mismatches in data alignment")


def retrieve_tile_name(df_l1c, df_l2a):
    """Retrieve the tile name for each row in the df"""
    df_l1c['tile_name'] = df_l1c["Name"].apply(lambda x: x.split("_")[-2])
    df_l2a['tile_name'] = df_l2a["Name"].apply(lambda x: x.split("_")[-2])

    df_l1c['single_tile_name'] = df_l1c['tile_name'].apply(lambda x: x[1:])
    df_l2a['single_tile_name'] = df_l2a['tile_name'].apply(lambda x: x[1:])
    return df_l1c, df_l2a


def parse_geofootprint(geofootprint):
    if isinstance(geofootprint, dict):
        # Already a dict, return as is
        return geofootprint
    elif isinstance(geofootprint, str):
        # Replace single quotes with double quotes for JSON parsing
        geofootprint_fixed = geofootprint.replace("'", '"')
        # Convert string to dict
        return ast.literal_eval(geofootprint_fixed)
    else:
        raise ValueError(f"Unsupported GeoFootprint type: {type(geofootprint)}")


def compute_coverage_ratio(geofootprint_str, bbox):
    try:
        geofootprint_dict = parse_geofootprint(geofootprint_str)
        tile_poly = shape(geofootprint_dict)
    except Exception as e:
        print(f"Error parsing GeoFootprint: {e}")
        return 0.0

    # Define AOI polygon from bbox
    aoi_polygon = Polygon([
        (bbox[0], bbox[1]),
        (bbox[0], bbox[3]),
        (bbox[2], bbox[3]),
        (bbox[2], bbox[1]),
        (bbox[0], bbox[1])
    ])
    
    intersection = tile_poly.intersection(aoi_polygon)
    if intersection.is_empty:
        return 0.0
    return intersection.area / aoi_polygon.area

def remove_last_segment_rsplit(sentinel_id):
    # Split from the right side, max 1 split
    parts = sentinel_id.rsplit('_', 1)
    return parts[0]