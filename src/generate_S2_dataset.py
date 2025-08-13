from datetime import datetime, timedelta
from loguru import logger 
import os 
import ast

from generate_ds import load_config, setup_environment, save_config_copy, setup_logger, query_sentinel_data, queries_curation, validate_data_alignment, retrieve_tile_name, parse_geofootprint, compute_coverage_ratio

def main():
    config_path = '/home/ubuntu/mucilage_pipeline/mucilage-detection/src/cfg/config_dataset.yaml'

    # Load configuration from YAML
    config = load_config(config_path)

    # Initialize environment using config
    env = setup_environment(config)

    # Save a copy of the config file to the dataset directory
    saved_config_path = save_config_copy(config, config_path, env['DATASET_DIR'])

    # Get query parameters from config
    query_config = config['query']
    bbox = query_config['bbox']
    start_date = datetime.strptime(query_config['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(query_config['end_date'], '%Y-%m-%d')
    max_items = query_config['max_items']
    max_cloud_cover = query_config['max_cloud_cover']
    label = config['label'] # added

    # Set up logger for query
    setup_logger(env['DATASET_DIR'], "sentinel_query_log")

    # Log query parameters
    logger.info(f"Using configuration from: {saved_config_path}")
    logger.info(f"Query parameters:")
    logger.info(f"Bounding box: {bbox}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Max items per request: {max_items}")
    logger.info(f"Max cloud cover: {max_cloud_cover}%")
    logger.info(f"Label: {label}")

    # Query Sentinel data
    all_l1c_results, all_l2a_results = query_sentinel_data(
        bbox, start_date, end_date, max_items, max_cloud_cover
    )

    # Process and align data
    df_l1c, df_l2a = queries_curation(all_l1c_results, all_l2a_results)

    # Retrieve tile names
    df_l1c, df_l2a = retrieve_tile_name(df_l1c, df_l2a)

    # Retrieve labels
    if label is not None:
        df_l1c["label"] = label
        df_l2a["label"] = label

    # Save full datasets
    # df_l1c.to_csv(f"{env['DATASET_DIR']}/input_l1c.csv")
    df_l2a.to_csv(f"{env['DATASET_DIR']}/output_l2a.csv")

    # Compute coverage ratio of image
    df_l2a['coverage_ratio'] = df_l2a['GeoFootprint'].apply(lambda gf: compute_coverage_ratio(gf, bbox))

    # Filter by threshold
    coverage_threshold = 0.7
    df_l2a_filtered = df_l2a[df_l2a['coverage_ratio'] > coverage_threshold].copy()
    df_l2a_filtered.to_csv(f"{env['DATASET_DIR']}/output_l2a_filtered.csv")

    # Validate alignment
    # validate_data_alignment(df_l1c, df_l2a)

    # Set up logger for download
    setup_logger(env['DATASET_DIR'], "sentinel_download_log")


if __name__ == "__main__":
    main()